import os
import smtplib
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.auth import (
    AuthRepository,
    SMTPDeliveryError,
    get_settings,
    hash_password,
    hash_secret,
    new_activation_token,
    reset_settings_cache,
    send_activation_email,
    test_smtp_connectivity as check_smtp_connectivity,
    utcnow,
    verify_password,
)
from app.main import create_app


class AuthenticationAndRBACTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "burnui.sqlite3"
        self.environment = {
            "BURNUI_DATABASE_URL": f"sqlite:///{self.database_path}",
            "BURNUI_SESSION_SECRET": "test-session-secret-with-more-than-thirty-two-characters",
            "BURNUI_COOKIE_SECURE": "false",
            "BURNUI_ADMIN_CONTACT_EMAIL": "admin@example.com",
            "BURNUI_ADMIN_EMAIL": "admin@example.com",
            "BURNUI_ADMIN_INITIAL_PASSWORD": "strong-admin-password",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "burnui@example.test",
            "SMTP_PASSWORD": "smtp-test-password",
            "SMTP_FROM_EMAIL": "burnui@example.test",
        }
        self.previous = {name: os.environ.get(name) for name in self.environment}
        os.environ.update(self.environment)
        reset_settings_cache()
        self.client_context = TestClient(create_app())
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        for name, previous in self.previous.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        reset_settings_cache()
        self.directory.cleanup()

    def admin_login(self) -> None:
        response = self.client.post("/api/auth/login", json={"email": "admin@example.com", "password": "strong-admin-password"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["role"], "ADMIN")
        self.assertIn("HttpOnly", response.headers["set-cookie"])

    def create_request(self, email: str = "inspector@example.com") -> int:
        response = self.client.post("/api/auth/clearance-requests", json={
            "full_name": "Inspector One", "email": email, "requested_role": "INSPECTOR", "department": "Reliability", "reason": "Need to review approved screening datasets.",
        })
        self.assertEqual(response.status_code, 201)
        return int(response.json()["request"]["id"])

    def test_clearance_request_is_pending_and_duplicate_pending_is_rejected(self) -> None:
        request_id = self.create_request()
        duplicate = self.client.post("/api/auth/clearance-requests", json={
            "full_name": "Inspector One", "email": "inspector@example.com", "requested_role": "INSPECTOR", "department": "Reliability", "reason": "Need to review approved screening datasets.",
        })
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(self.client.post("/api/auth/login", json={"email": "inspector@example.com", "password": "not-a-password"}).status_code, 401)
        self.admin_login()
        requests = self.client.get("/api/admin/clearance-requests").json()["requests"]
        self.assertEqual(next(item for item in requests if item["id"] == request_id)["status"], "PENDING")

    def test_approval_delivers_activation_then_activation_is_single_use(self) -> None:
        request_id = self.create_request()
        self.admin_login()
        with patch("app.api.routes.admin.send_activation_email") as send_email:
            approved = self.client.post(f"/api/admin/clearance-requests/{request_id}/approve", json={"confirm": True})
        self.assertEqual(approved.status_code, 200)
        self.assertTrue(approved.json()["activation_email_sent"])
        token = send_email.call_args.kwargs["activation_token"]
        self.assertNotIn(token, str(approved.json()))
        self.assertEqual(self.client.post("/api/auth/login", json={"email": "inspector@example.com", "password": "unactivated-password"}).status_code, 401)
        activated = self.client.post("/api/auth/activate", json={"token": token, "password": "valid-activated-password"})
        self.assertEqual(activated.status_code, 200)
        self.assertEqual(self.client.post("/api/auth/activate", json={"token": token, "password": "valid-activated-password"}).status_code, 400)
        inspector_login = self.client.post("/api/auth/login", json={"email": "inspector@example.com", "password": "valid-activated-password"})
        self.assertEqual(inspector_login.status_code, 200)
        self.assertEqual(inspector_login.json()["user"]["role"], "INSPECTOR")

    def test_approval_stays_approved_and_exposes_link_when_smtp_is_not_configured(self) -> None:
        request_id = self.create_request()
        self.admin_login()
        smtp_values = {name: os.environ.pop(name) for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL")}
        reset_settings_cache()
        try:
            result = self.client.post(f"/api/admin/clearance-requests/{request_id}/approve", json={"confirm": True})
            self.assertEqual(result.status_code, 200)
            payload = result.json()
            self.assertFalse(payload["activation_email_sent"])
            self.assertEqual(payload["email_delivery"]["category"], "smtp_configuration_missing")
            self.assertIn("?activate=", payload["activation_url"])
            requests = self.client.get("/api/admin/clearance-requests").json()["requests"]
            self.assertEqual(next(item for item in requests if item["id"] == request_id)["status"], "APPROVED")
            user = AuthRepository(get_settings()).get_user_by_email("inspector@example.com")
            self.assertIsNotNone(user)
            assert user is not None
            self.assertFalse(user.is_active)
            self.assertIsNotNone(user.activation_token_hash)
            logs = self.client.get("/api/admin/audit-logs").json()["audit_logs"]
            failure = next(item for item in logs if item["event_type"] == "CLEARANCE_EMAIL_DELIVERY_FAILED")
            self.assertEqual(failure["metadata"]["category"], "smtp_configuration_missing")
            self.assertNotIn("activate=", str(logs))
            self.assertNotIn(payload["activation_url"].split("?activate=", 1)[1], str(logs))
            self.assertNotIn(self.environment["BURNUI_SESSION_SECRET"], str(logs))
        finally:
            os.environ.update(smtp_values)
            reset_settings_cache()

    def test_gmail_smtp_uses_ehlo_starttls_authentication_and_clean_close(self) -> None:
        settings = replace(
            get_settings(),
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_username="mailer@example.org",
            smtp_password="test-app-password",
            smtp_from_email="mailer@example.org",
        )
        with patch("app.auth.smtplib.SMTP") as smtp:
            client = smtp.return_value
            send_activation_email(
                recipient="inspector@example.org",
                activation_token="test-activation-token",
                role="INSPECTOR",
                settings=settings,
            )
        smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=15)
        self.assertEqual(client.ehlo.call_count, 2)
        client.starttls.assert_called_once()
        client.login.assert_called_once_with("mailer@example.org", "test-app-password")
        client.send_message.assert_called_once()
        client.quit.assert_called_once()

    def test_smtp_failure_categories_are_sanitized(self) -> None:
        settings = replace(
            get_settings(),
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_username="mailer@example.org",
            smtp_password="test-app-password",
            smtp_from_email="mailer@example.org",
        )
        with patch("app.auth.smtplib.SMTP") as smtp:
            smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(535, b"authentication failed")
            with self.assertRaises(SMTPDeliveryError) as failed:
                send_activation_email(recipient="inspector@example.org", activation_token="test-activation-token", role="INSPECTOR", settings=settings)
        self.assertEqual(failed.exception.category, "smtp_authentication_failed")

        with patch("app.auth.smtplib.SMTP", side_effect=OSError("offline")):
            result = check_smtp_connectivity(settings)
        self.assertFalse(result["success"])
        self.assertEqual(result["category"], "smtp_connection_failed")
        self.assertNotIn("test-app-password", str(result))

    def test_smtp_missing_configuration_is_reported_without_a_connection_attempt(self) -> None:
        settings = replace(get_settings(), smtp_host=None)
        with patch("app.auth.smtplib.SMTP") as smtp:
            result = check_smtp_connectivity(settings)
        self.assertFalse(result["success"])
        self.assertEqual(result["category"], "smtp_configuration_missing")
        smtp.assert_not_called()

    def test_activation_token_expiry_remains_enforced_after_delivery_fallback(self) -> None:
        repository = AuthRepository(get_settings())
        token, token_hash, _ = new_activation_token()
        repository.create_user(
            full_name="Expired Activation",
            email="expired@example.com",
            role="INSPECTOR",
            department="Reliability",
            password_hash=None,
            is_active=False,
            activation_token_hash=token_hash,
            activation_expires_at=utcnow() - timedelta(seconds=1),
        )
        self.assertIsNone(repository.activate_user(hash_secret(token), hash_password("valid-activated-password")))

    def test_smtp_connectivity_endpoint_requires_administrator(self) -> None:
        self.assertEqual(self.client.post("/api/admin/smtp-test").status_code, 401)
        self.admin_login()
        with patch("app.api.routes.admin.test_smtp_connectivity", return_value={"success": True, "category": "smtp_ready", "message": "ok"}) as check:
            result = self.client.post("/api/admin/smtp-test")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()["success"])
        check.assert_called_once()

    def test_declined_request_cannot_log_in_and_is_audited(self) -> None:
        request_id = self.create_request("declined@example.com")
        self.admin_login()
        self.assertEqual(self.client.post(f"/api/admin/clearance-requests/{request_id}/decline", json={"confirm": True}).status_code, 200)
        self.assertEqual(self.client.post("/api/auth/login", json={"email": "declined@example.com", "password": "valid-activated-password"}).status_code, 401)
        self.admin_login()
        events = [item["event_type"] for item in self.client.get("/api/admin/audit-logs").json()["audit_logs"]]
        self.assertIn("CLEARANCE_DECLINED", events)

    def test_login_throttles_repeated_failures(self) -> None:
        for _ in range(5):
            self.assertEqual(self.client.post("/api/auth/login", json={"email": "admin@example.com", "password": "wrong-password"}).status_code, 401)
        self.assertEqual(self.client.post("/api/auth/login", json={"email": "admin@example.com", "password": "strong-admin-password"}).status_code, 429)

    def test_logout_invalidates_the_cookie_session(self) -> None:
        self.admin_login()
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_rbac_returns_401_without_session_and_403_for_inspector_admin_access(self) -> None:
        anonymous = TestClient(create_app())
        self.assertEqual(anonymous.get("/api/admin/users").status_code, 401)
        request_id = self.create_request()
        self.admin_login()
        with patch("app.api.routes.admin.send_activation_email") as send_email:
            self.client.post(f"/api/admin/clearance-requests/{request_id}/approve", json={"confirm": True})
        token = send_email.call_args.kwargs["activation_token"]
        self.client.post("/api/auth/activate", json={"token": token, "password": "valid-activated-password"})
        self.client.post("/api/auth/login", json={"email": "inspector@example.com", "password": "valid-activated-password"})
        self.assertEqual(self.client.get("/api/admin/users").status_code, 403)
        self.assertEqual(self.client.post("/api/admin/smtp-test").status_code, 403)

    def test_password_hash_and_activation_token_are_not_plaintext(self) -> None:
        repository = AuthRepository(get_settings())
        request_id = self.create_request()
        self.admin_login()
        with patch("app.api.routes.admin.send_activation_email") as send_email:
            self.client.post(f"/api/admin/clearance-requests/{request_id}/approve", json={"confirm": True})
        token = send_email.call_args.kwargs["activation_token"]
        user = repository.get_user_by_email("inspector@example.com")
        self.assertIsNotNone(user)
        self.assertEqual(user.activation_token_hash, hash_secret(token))
        self.assertNotEqual(user.activation_token_hash, token)
        self.assertIsNone(user.password_hash)

    def test_update_user_password_replaces_only_an_existing_password_hash(self) -> None:
        repository = AuthRepository(get_settings())
        admin = repository.get_user_by_email("admin@example.com")
        self.assertIsNotNone(admin)
        assert admin is not None
        original_hash = admin.password_hash
        replacement_hash = hash_password("replacement-admin-password")

        updated = repository.update_user_password(admin.id, replacement_hash)
        missing = repository.update_user_password(999999, replacement_hash)

        self.assertIsNotNone(updated)
        self.assertIsNone(missing)
        reloaded = repository.get_user_by_id(admin.id)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertNotEqual(reloaded.password_hash, original_hash)
        self.assertTrue(verify_password("replacement-admin-password", reloaded.password_hash))
        self.assertFalse(verify_password("strong-admin-password", reloaded.password_hash))
