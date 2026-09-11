"""Public authentication, activation, and clearance-request routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.auth import (
    ActivationInput,
    AuthConfigurationError,
    AuthRepository,
    ClearanceRequestInput,
    DatabaseUnavailable,
    ForgotPasswordInput,
    LoginInput,
    Settings,
    User,
    clear_session_cookie,
    get_repository,
    get_settings,
    hash_password,
    hash_secret,
    login_user,
    new_activation_token,
    require_authenticated_user,
    send_activation_email,
    set_session_cookie,
)


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/public-config")
def public_config(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    """Expose only the administrator contact address needed by the login UI."""
    return {
        "admin_contact_email": settings.admin_contact_email,
        "smtp_configured": settings.smtp_configured,
    }


@router.post("/login")
def login(
    payload: LoginInput,
    response: Response,
    repository: AuthRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    try:
        user = login_user(payload, repository, settings)
        set_session_cookie(response, user, settings)
        return {"user": user.public_dict()}
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": str(exc)}) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": "Authentication is temporarily unavailable."}) from exc


@router.post("/logout")
def logout(
    response: Response,
    user: User = Depends(require_authenticated_user),
    repository: AuthRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    repository.invalidate_sessions(user.id)
    repository.log_event("LOGOUT", actor_user_id=user.id, target_user_id=user.id, metadata={})
    clear_session_cookie(response, settings)
    return {"message": "Logged out."}


@router.get("/me")
def current_user(user: User = Depends(require_authenticated_user)) -> dict[str, object]:
    return {"user": user.public_dict()}


@router.post("/clearance-requests", status_code=status.HTTP_201_CREATED)
def request_clearance(
    payload: ClearanceRequestInput,
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, object]:
    try:
        request = repository.create_clearance_request(payload)
        repository.log_event("CLEARANCE_REQUEST", metadata={"request_id": request.id, "requested_role": request.requested_role})
        return {"request": request.public_dict()}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"message": str(exc)}) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": "Clearance requests are temporarily unavailable."}) from exc


@router.post("/activate")
def activate_account(
    payload: ActivationInput,
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, object]:
    try:
        user = repository.activate_user(hash_secret(payload.token), hash_password(payload.password))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"message": str(exc)}) from exc
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": "Account activation is temporarily unavailable."}) from exc
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"message": "The activation link is invalid, expired, or has already been used."})
    repository.log_event("ACCOUNT_ACTIVATED", actor_user_id=user.id, target_user_id=user.id, metadata={})
    return {"message": "Account activated. You can now sign in."}


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
def forgot_password(
    _: ForgotPasswordInput,
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, str]:
    """Intentionally non-enumerating: recovery is handled by an administrator."""
    repository.log_event("PASSWORD_RESET_REQUEST", metadata={})
    return {"message": "For security, password recovery is handled by the administrator."}
