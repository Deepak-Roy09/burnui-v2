"""Authentication, RBAC, clearance workflow, and audit support for BurnUI.

This module is intentionally separate from the production anomaly pipeline.
SQLite is used for this prototype behind a repository boundary so it can be
replaced with PostgreSQL without changing route-level business logic.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import socket
import smtplib
import ssl
import sqlite3
from email.message import EmailMessage
from functools import lru_cache
from typing import Callable, Iterator, Literal
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator

try:
    import bcrypt
except ImportError:
    bcrypt = None

try:  # Prefer Argon2id when a deployment provides it.
    from argon2 import PasswordHasher
    from argon2.exceptions import VerificationError
    from argon2.low_level import Type

    ARGON2_HASHER: PasswordHasher | None = PasswordHasher(type=Type.ID)
except ImportError:  # bcrypt is the declared secure fallback dependency.
    ARGON2_HASHER = None
    VerificationError = ValueError


# Load the backend-local configuration once before get_settings is decorated
# and evaluated. Host-supplied environment values still take precedence.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

Role = Literal["INSPECTOR", "ENGINEER", "ADMIN"]
ROLES: tuple[Role, ...] = ("INSPECTOR", "ENGINEER", "ADMIN")
REQUEST_ROLES: tuple[Role, ...] = ("INSPECTOR", "ENGINEER")
SESSION_COOKIE_NAME = "burnui_session"
LOGIN_FAILURE_LIMIT = 5
LOGIN_COOLDOWN = timedelta(minutes=15)
SESSION_LIFETIME = timedelta(hours=8)
ACTIVATION_LIFETIME = timedelta(hours=24)
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
SMTP_PLACEHOLDER_VALUES = {
    "smtp.example.com",
    "smtp.example.test",
    "replace-with-smtp-password",
    "your_google_app_password",
    "your-google-app-password",
    "burnui@example.com",
    "your-gmail-address@example.com",
}

# Equalize rejected-login work for unknown, pending, and inactive accounts.
if ARGON2_HASHER is not None:
    DUMMY_PASSWORD_HASH = ARGON2_HASHER.hash("burnui-login-timing-placeholder")
elif bcrypt is not None:
    DUMMY_PASSWORD_HASH = bcrypt.hashpw(
        b"burnui-login-timing-placeholder",
        bcrypt.gensalt(rounds=12),
    ).decode("utf-8")
else:
    DUMMY_PASSWORD_HASH = None


class AuthConfigurationError(RuntimeError):
    """Raised for a missing security or SMTP configuration."""


class DatabaseUnavailable(RuntimeError):
    """Raised when the configured identity database cannot be accessed."""


class SMTPDeliveryError(RuntimeError):
    """Sanitized SMTP failure category; never contains credentials or tokens."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class Settings:
    database_url: str
    session_secret: str | None
    cookie_secure: bool
    smtp_host: str | None
    smtp_port: int | None
    smtp_username: str | None
    smtp_password: str | None
    smtp_from_email: str | None
    admin_contact_email: str | None
    admin_email: str | None
    admin_initial_password: str | None
    public_base_url: str

    @property
    def database_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise AuthConfigurationError("Only sqlite URLs are supported by this prototype repository.")
        return Path(self.database_url.removeprefix(prefix)).expanduser()

    @property
    def smtp_configured(self) -> bool:
        return smtp_configuration_error(self) is None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read environment configuration after local dotenv initialization."""
    project_backend = Path(__file__).resolve().parents[1]
    default_database = project_backend / "data" / "burnui.sqlite3"
    port_value = os.getenv("SMTP_PORT")
    try:
        smtp_port = int(port_value) if port_value else None
    except ValueError as exc:
        raise AuthConfigurationError("SMTP_PORT must be a valid integer.") from exc
    return Settings(
        database_url=os.getenv("BURNUI_DATABASE_URL", f"sqlite:///{default_database}"),
        session_secret=os.getenv("BURNUI_SESSION_SECRET"),
        cookie_secure=os.getenv("BURNUI_COOKIE_SECURE", "true").strip().lower() in {"1", "true", "yes"},
        smtp_host=os.getenv("SMTP_HOST"),
        smtp_port=smtp_port,
        smtp_username=os.getenv("SMTP_USERNAME"),
        smtp_password=os.getenv("SMTP_PASSWORD"),
        smtp_from_email=os.getenv("SMTP_FROM_EMAIL"),
        admin_contact_email=os.getenv("BURNUI_ADMIN_CONTACT_EMAIL"),
        admin_email=os.getenv("BURNUI_ADMIN_EMAIL"),
        admin_initial_password=os.getenv("BURNUI_ADMIN_INITIAL_PASSWORD"),
        public_base_url=os.getenv("BURNUI_PUBLIC_BASE_URL", "http://127.0.0.1:5173").rstrip("/"),
    )


def reset_settings_cache() -> None:
    """Test-only helper for re-reading environment configuration."""
    get_settings.cache_clear()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_timestamp(value: datetime | None = None) -> str:
    return (value or utcnow()).isoformat()


def parse_timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class User:
    id: int
    full_name: str
    email: str
    role: Role
    department: str | None
    password_hash: str | None
    is_active: bool
    session_version: int
    failed_login_count: int
    locked_until: datetime | None
    activation_token_hash: str | None
    activation_expires_at: datetime | None
    activation_used_at: datetime | None
    created_at: str

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "department": self.department,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ClearanceRequest:
    id: int
    full_name: str
    email: str
    requested_role: Role
    department: str
    reason: str
    status: Literal["PENDING", "APPROVED", "DECLINED"]
    created_at: str
    reviewed_at: str | None
    reviewed_by_user_id: int | None

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "requested_role": self.requested_role,
            "department": self.department,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "reviewed_by_user_id": self.reviewed_by_user_id,
        }


class ClearanceRequestInput(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: str = Field(max_length=254)
    requested_role: Literal["INSPECTOR", "ENGINEER"]
    department: str = Field(min_length=2, max_length=120)
    reason: str = Field(min_length=5, max_length=1000)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)


class LoginInput(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)


class ActivationInput(BaseModel):
    token: str = Field(min_length=20, max_length=512)
    password: str = Field(min_length=12, max_length=256)


class ForgotPasswordInput(BaseModel):
    email: str = Field(max_length=254)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)


class ApprovalInput(BaseModel):
    confirm: Literal[True]


class DeclineInput(BaseModel):
    confirm: Literal[True]


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address.")
    return email


def _require_password_hasher() -> None:
    if ARGON2_HASHER is None and bcrypt is None:
        raise AuthConfigurationError("No supported password hasher is installed. Install the bcrypt dependency from requirements.txt.")


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if ARGON2_HASHER is not None:
        return ARGON2_HASHER.hash(password)
    _require_password_hasher()
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False
    try:
        if stored_hash.startswith("$argon2"):
            return bool(ARGON2_HASHER and ARGON2_HASHER.verify(stored_hash, password))
        if bcrypt is None:
            return False
        return bool(bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")))
    except (ValueError, VerificationError):
        return False


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthRepository:
    """SQLite repository boundary; routes and services do not issue SQL directly."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield one transaction-scoped SQLite connection and always close it.

        The sqlite3 connection context manager only commits or rolls back; it
        does not close its operating-system file handle. Explicit closure is
        required before Windows can remove a temporary test database.
        """
        connection: sqlite3.Connection | None = None
        try:
            path = self.settings.database_path
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path, timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
        except (OSError, sqlite3.Error, AuthConfigurationError) as exc:
            if connection is not None:
                connection.close()
            raise DatabaseUnavailable("The BurnUI identity database is unavailable.") from exc
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            role TEXT NOT NULL CHECK (role IN ('INSPECTOR', 'ENGINEER', 'ADMIN')),
            department TEXT,
            password_hash TEXT,
            is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
            activation_token_hash TEXT,
            activation_expires_at TEXT,
            activation_used_at TEXT,
            failed_login_count INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            session_version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS clearance_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL COLLATE NOCASE,
            requested_role TEXT NOT NULL CHECK (requested_role IN ('INSPECTOR', 'ENGINEER')),
            department TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('PENDING', 'APPROVED', 'DECLINED')),
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewed_by_user_id INTEGER REFERENCES users(id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_pending_clearance_per_email
            ON clearance_requests(email) WHERE status = 'PENDING';
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            actor_user_id INTEGER REFERENCES users(id),
            target_user_id INTEGER REFERENCES users(id),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS audit_logs_created_at ON audit_logs(created_at DESC);
        """
        try:
            with self._connection() as connection:
                connection.executescript(schema)
        except sqlite3.Error as exc:
            raise DatabaseUnavailable("The BurnUI identity database could not be initialized.") from exc

    def create_user(
        self,
        *,
        full_name: str,
        email: str,
        role: Role,
        department: str | None,
        password_hash: str | None,
        is_active: bool,
        activation_token_hash: str | None = None,
        activation_expires_at: datetime | None = None,
    ) -> User:
        now = to_timestamp()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """INSERT INTO users (
                        full_name, email, role, department, password_hash, is_active,
                        activation_token_hash, activation_expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        full_name.strip(), normalize_email(email), role, department.strip() if department else None,
                        password_hash, int(is_active), activation_token_hash,
                        to_timestamp(activation_expires_at) if activation_expires_at else None, now, now,
                    ),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account already exists for this email address.") from exc
        user = self.get_user_by_id(user_id)
        if user is None:
            raise DatabaseUnavailable("The newly created BurnUI user could not be read.")
        return user

    def get_user_by_email(self, email: str) -> User | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (normalize_email(email),)).fetchone()
        return _row_to_user(row) if row else None

    def get_user_by_id(self, user_id: int) -> User | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None

    def update_user_password(self, user_id: int, password_hash: str) -> User | None:
        """Replace only an existing user's already-hashed password value."""
        if not password_hash:
            raise ValueError("password_hash must not be empty.")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id),
            )
        if cursor.rowcount != 1:
            return None
        return self.get_user_by_id(user_id)

    def create_clearance_request(self, request: ClearanceRequestInput) -> ClearanceRequest:
        now = to_timestamp()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """INSERT INTO clearance_requests
                    (full_name, email, requested_role, department, reason, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'PENDING', ?)""",
                    (request.full_name.strip(), request.email, request.requested_role, request.department.strip(), request.reason.strip(), now),
                )
                request_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError("A clearance request for this email is already pending.") from exc
        result = self.get_clearance_request(request_id)
        if result is None:
            raise DatabaseUnavailable("The clearance request could not be read.")
        return result

    def get_clearance_request(self, request_id: int) -> ClearanceRequest | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM clearance_requests WHERE id = ?", (request_id,)).fetchone()
        return _row_to_clearance_request(row) if row else None

    def list_clearance_requests(self) -> list[ClearanceRequest]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM clearance_requests ORDER BY created_at DESC, id DESC").fetchall()
        return [_row_to_clearance_request(row) for row in rows]

    def review_clearance_request(self, request_id: int, *, status_value: Literal["APPROVED", "DECLINED"], reviewer_id: int) -> ClearanceRequest:
        now = to_timestamp()
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE clearance_requests
                   SET status = ?, reviewed_at = ?, reviewed_by_user_id = ?
                   WHERE id = ? AND status = 'PENDING'""",
                (status_value, now, reviewer_id, request_id),
            )
        if cursor.rowcount != 1:
            raise ValueError("Only a pending clearance request can be reviewed.")
        reviewed = self.get_clearance_request(request_id)
        if reviewed is None:
            raise DatabaseUnavailable("The reviewed clearance request could not be read.")
        return reviewed

    def activate_user(self, token_hash: str, password_hash: str) -> User | None:
        now_string = to_timestamp()
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM users
                   WHERE activation_token_hash = ? AND activation_used_at IS NULL
                     AND activation_expires_at > ? AND is_active = 0""",
                (token_hash, now_string),
            ).fetchone()
            if row is None:
                return None
            user_id = int(row["id"])
            connection.execute(
                """UPDATE users SET password_hash = ?, is_active = 1, activation_used_at = ?,
                   activation_token_hash = NULL, activation_expires_at = NULL, failed_login_count = 0,
                   locked_until = NULL, updated_at = ?, session_version = session_version + 1 WHERE id = ?""",
                (password_hash, now_string, now_string, user_id),
            )
        return self.get_user_by_id(user_id)

    def delete_unactivated_user(self, user_id: int) -> None:
        """Remove a just-created inactive account if activation delivery failed."""
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM users WHERE id = ? AND is_active = 0 AND activation_used_at IS NULL",
                (user_id,),
            )

    def record_login_failure(self, user: User | None) -> None:
        if user is None:
            return
        next_count = user.failed_login_count + 1
        locked_until = utcnow() + LOGIN_COOLDOWN if next_count >= LOGIN_FAILURE_LIMIT else None
        with self._connection() as connection:
            connection.execute(
                "UPDATE users SET failed_login_count = ?, locked_until = ?, updated_at = ? WHERE id = ?",
                (next_count, to_timestamp(locked_until) if locked_until else None, to_timestamp(), user.id),
            )

    def clear_login_failures(self, user_id: int) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE users SET failed_login_count = 0, locked_until = NULL, updated_at = ? WHERE id = ?",
                (to_timestamp(), user_id),
            )

    def invalidate_sessions(self, user_id: int) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE users SET session_version = session_version + 1, updated_at = ? WHERE id = ?",
                (to_timestamp(), user_id),
            )

    def list_users(self) -> list[User]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY created_at DESC, id DESC").fetchall()
        return [_row_to_user(row) for row in rows]

    def has_users(self) -> bool:
        """Return whether the identity store already contains any account."""
        with self._connection() as connection:
            row = connection.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        return row is not None

    def log_event(
        self,
        event_type: str,
        *,
        actor_user_id: int | None = None,
        target_user_id: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        safe_metadata = metadata or {}
        forbidden = {"password", "password_hash", "token", "activation_token", "smtp_password", "secret"}
        if any(key.lower() in forbidden for key in safe_metadata):
            raise ValueError("Audit metadata must not include secrets.")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO audit_logs (event_type, actor_user_id, target_user_id, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (event_type, actor_user_id, target_user_id, json.dumps(safe_metadata, sort_keys=True), to_timestamp()),
            )

    def list_audit_logs(self, limit: int = 250) -> list[dict[str, object]]:
        """Return safe administrative audit rows in reverse chronological order."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id, event_type, actor_user_id, target_user_id, metadata_json, created_at
                   FROM audit_logs
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "event_type": str(row["event_type"]),
                "actor_user_id": row["actor_user_id"],
                "target_user_id": row["target_user_id"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        full_name=str(row["full_name"]),
        email=str(row["email"]),
        role=str(row["role"]),
        department=row["department"],
        password_hash=row["password_hash"],
        is_active=bool(row["is_active"]),
        session_version=int(row["session_version"]),
        failed_login_count=int(row["failed_login_count"]),
        locked_until=parse_timestamp(row["locked_until"]),
        activation_token_hash=row["activation_token_hash"],
        activation_expires_at=parse_timestamp(row["activation_expires_at"]),
        activation_used_at=parse_timestamp(row["activation_used_at"]),
        created_at=str(row["created_at"]),
    )


def _row_to_clearance_request(row: sqlite3.Row) -> ClearanceRequest:
    return ClearanceRequest(
        id=int(row["id"]),
        full_name=str(row["full_name"]),
        email=str(row["email"]),
        requested_role=str(row["requested_role"]),
        department=str(row["department"]),
        reason=str(row["reason"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        reviewed_at=row["reviewed_at"],
        reviewed_by_user_id=row["reviewed_by_user_id"],
    )


def get_repository(settings: Settings = Depends(get_settings)) -> AuthRepository:
    repository = AuthRepository(settings)
    repository.initialize()
    return repository


def bootstrap_admin(repository: AuthRepository, settings: Settings) -> None:
    """Create the configured administrator once, with only a secure password hash."""
    if not settings.admin_email and not settings.admin_initial_password:
        return
    if not settings.admin_email or not settings.admin_initial_password:
        raise AuthConfigurationError("Set both BURNUI_ADMIN_EMAIL and BURNUI_ADMIN_INITIAL_PASSWORD to bootstrap an admin.")
    if repository.has_users():
        return
    admin = repository.create_user(
        full_name="BurnUI Administrator",
        email=settings.admin_email,
        role="ADMIN",
        department="Administration",
        password_hash=hash_password(settings.admin_initial_password),
        is_active=True,
    )
    repository.log_event("ACCOUNT_ACTIVATED", actor_user_id=admin.id, target_user_id=admin.id, metadata={"bootstrap": True})


def _require_session_secret(settings: Settings) -> str:
    if not settings.session_secret or len(settings.session_secret) < 32:
        raise AuthConfigurationError("BURNUI_SESSION_SECRET must be set to at least 32 random characters.")
    return settings.session_secret


def create_session_token(user: User, settings: Settings) -> str:
    secret = _require_session_secret(settings).encode("utf-8")
    payload = {
        "sub": user.id,
        "role": user.role,
        "sv": user.session_version,
        "exp": int((utcnow() + SESSION_LIFETIME).timestamp()),
        "nonce": secrets.token_urlsafe(12),
    }
    encoded = _base64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _base64url(hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def verify_session_token(token: str, settings: Settings) -> dict[str, object] | None:
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = _base64url(
            hmac.new(_require_session_secret(settings).encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        payload = json.loads(_base64url_decode(encoded))
        if not isinstance(payload, dict) or int(payload["exp"]) <= int(utcnow().timestamp()):
            return None
        if not isinstance(payload.get("sub"), int) or payload.get("role") not in ROLES or not isinstance(payload.get("sv"), int):
            return None
        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, AuthConfigurationError):
        return None


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def set_session_cookie(response: Response, user: User, settings: Settings) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_token(user, settings),
        max_age=int(SESSION_LIFETIME.total_seconds()),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


def require_authenticated_user(
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    repository: AuthRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> User:
    if not session_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"message": "Authentication is required."})
    payload = verify_session_token(session_cookie, settings)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"message": "Your session is invalid or has expired."})
    user = repository.get_user_by_id(int(payload["sub"]))
    if not user or not user.is_active or user.session_version != int(payload["sv"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"message": "Your session is invalid or has expired."})
    return user


def require_role(*roles: Role) -> Callable[[User], User]:
    def dependency(user: User = Depends(require_authenticated_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"message": "You do not have permission for this action."})
        return user

    return dependency


def ensure_smtp_configured(settings: Settings) -> None:
    error_category = smtp_configuration_error(settings)
    if error_category:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "SMTP is not configured. Set valid SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, and SMTP_FROM_EMAIL values before approving clearance requests."},
        )


def smtp_configuration_error(settings: Settings) -> str | None:
    """Return a safe SMTP configuration category without exposing any secret."""
    required_values = (
        settings.smtp_host,
        settings.smtp_username,
        settings.smtp_password,
        settings.smtp_from_email,
    )
    if not all(required_values):
        return "smtp_configuration_missing"
    if not isinstance(settings.smtp_port, int) or not 1 <= settings.smtp_port <= 65535:
        return "smtp_port_invalid"
    if any(str(value).strip().lower() in SMTP_PLACEHOLDER_VALUES for value in required_values):
        return "smtp_configuration_placeholder"
    return None


def build_activation_url(activation_token: str, settings: Settings) -> str:
    """Build the one-time activation URL from the configured public base URL."""
    separator = "&" if "?" in settings.public_base_url else "?"
    return f"{settings.public_base_url}{separator}{urlencode({'activate': activation_token})}"


def _close_smtp_client(client: smtplib.SMTP) -> None:
    """Close an SMTP client without replacing a more useful delivery error."""
    try:
        client.quit()
    except (OSError, smtplib.SMTPException):
        try:
            client.close()
        except OSError:
            pass


@contextmanager
def _smtp_client(settings: Settings) -> Iterator[smtplib.SMTP]:
    configuration_error = smtp_configuration_error(settings)
    if configuration_error:
        raise SMTPDeliveryError(configuration_error)
    try:
        client = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15)
    except (TimeoutError, socket.timeout) as exc:
        raise SMTPDeliveryError("smtp_timeout") from exc
    except (OSError, smtplib.SMTPConnectError) as exc:
        raise SMTPDeliveryError("smtp_connection_failed") from exc

    try:
        try:
            client.ehlo()
        except (OSError, smtplib.SMTPException) as exc:
            raise SMTPDeliveryError("smtp_connection_failed") from exc
        try:
            client.starttls(context=ssl.create_default_context())
            client.ehlo()
        except (OSError, smtplib.SMTPException) as exc:
            raise SMTPDeliveryError("starttls_failed") from exc
        try:
            client.login(settings.smtp_username, settings.smtp_password)
        except smtplib.SMTPAuthenticationError as exc:
            raise SMTPDeliveryError("smtp_authentication_failed") from exc
        except (OSError, smtplib.SMTPException) as exc:
            raise SMTPDeliveryError("smtp_authentication_failed") from exc
    except SMTPDeliveryError:
        _close_smtp_client(client)
        raise

    try:
        yield client
    finally:
        _close_smtp_client(client)


def test_smtp_connectivity(settings: Settings) -> dict[str, object]:
    """Authenticate to SMTP without sending an email; response is safe for admins."""
    try:
        with _smtp_client(settings):
            pass
    except SMTPDeliveryError as exc:
        return {
            "success": False,
            "category": exc.category,
            "message": "SMTP connectivity test failed. Check the server configuration and administrator diagnostics.",
        }
    return {
        "success": True,
        "category": "smtp_ready",
        "message": "SMTP connection, STARTTLS, and authentication succeeded.",
    }


def send_activation_email(
    *,
    recipient: str,
    activation_token: str,
    role: Role,
    settings: Settings,
) -> None:
    """Deliver an activation link without persisting or logging its plaintext token."""
    activation_url = build_activation_url(activation_token, settings)
    message = EmailMessage()
    message["Subject"] = "SafeHighR/BurnUI clearance approved — activate your account"
    message["From"] = settings.smtp_from_email
    message["To"] = recipient
    message.set_content(
        "Your SafeHighR/BurnUI clearance request has been approved.\n\n"
        f"Your {role.title()} account is ready to activate.\n\n"
        "Use this secure activation link to create your password:\n"
        f"{activation_url}\n\n"
        "This link expires in 24 hours and can be used once. Do not share it."
    )
    try:
        with _smtp_client(settings) as client:
            client.send_message(message)
    except smtplib.SMTPSenderRefused as exc:
        raise SMTPDeliveryError("sender_rejected") from exc
    except smtplib.SMTPRecipientsRefused as exc:
        raise SMTPDeliveryError("recipient_rejected") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise SMTPDeliveryError("smtp_timeout") from exc
    except (OSError, smtplib.SMTPException) as exc:
        raise SMTPDeliveryError("unexpected_smtp_response") from exc


def new_activation_token() -> tuple[str, str, datetime]:
    token = secrets.token_urlsafe(32)
    return token, hash_secret(token), utcnow() + ACTIVATION_LIFETIME


def login_user(payload: LoginInput, repository: AuthRepository, settings: Settings) -> User:
    _require_password_hasher()
    user = repository.get_user_by_email(payload.email)
    invalid_credentials = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"message": "Invalid email or password."})
    if not user or not user.is_active or not user.password_hash:
        if DUMMY_PASSWORD_HASH:
            verify_password(payload.password, DUMMY_PASSWORD_HASH)
        repository.log_event("LOGIN_FAILURE", metadata={"reason": "invalid_credentials"})
        raise invalid_credentials
    if user.locked_until and user.locked_until > utcnow():
        repository.log_event("LOGIN_FAILURE", actor_user_id=user.id, target_user_id=user.id, metadata={"reason": "cooldown"})
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail={"message": "Too many failed login attempts. Please try again later."})
    if not verify_password(payload.password, user.password_hash):
        repository.record_login_failure(user)
        repository.log_event("LOGIN_FAILURE", actor_user_id=user.id, target_user_id=user.id, metadata={"reason": "invalid_credentials"})
        raise invalid_credentials
    repository.clear_login_failures(user.id)
    repository.log_event("LOGIN_SUCCESS", actor_user_id=user.id, target_user_id=user.id, metadata={})
    return repository.get_user_by_id(user.id) or user
