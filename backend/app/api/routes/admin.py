"""Admin-only clearance, user, and audit-management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import (
    ApprovalInput,
    AuthRepository,
    ClearanceRequest,
    DatabaseUnavailable,
    DeclineInput,
    SMTPDeliveryError,
    Settings,
    User,
    build_activation_url,
    get_repository,
    get_settings,
    new_activation_token,
    require_role,
    send_activation_email,
    test_smtp_connectivity,
)


router = APIRouter(prefix="/admin", tags=["administration"])
require_admin = require_role("ADMIN")


@router.get("/overview")
def overview(
    _: User = Depends(require_admin),
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, int]:
    requests = repository.list_clearance_requests()
    users = repository.list_users()
    return {
        "pending_clearance_requests": sum(request.status == "PENDING" for request in requests),
        "approved_clearance_requests": sum(request.status == "APPROVED" for request in requests),
        "user_count": len(users),
        "active_user_count": sum(user.is_active for user in users),
    }


@router.get("/clearance-requests")
def clearance_requests(
    _: User = Depends(require_admin),
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, object]:
    return {"requests": [request.public_dict() for request in repository.list_clearance_requests()]}


@router.get("/users")
def users(
    _: User = Depends(require_admin),
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, object]:
    return {"users": [user.public_dict() for user in repository.list_users()]}


@router.get("/audit-logs")
def audit_logs(
    _: User = Depends(require_admin),
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, object]:
    return {"audit_logs": repository.list_audit_logs()}


@router.post("/smtp-test")
def smtp_test(
    _: User = Depends(require_admin),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Check SMTP + STARTTLS authentication without sending any email."""
    return test_smtp_connectivity(settings)


@router.post("/clearance-requests/{request_id}/approve")
def approve_clearance(
    request_id: int,
    _: ApprovalInput,
    admin: User = Depends(require_admin),
    repository: AuthRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    request = repository.get_clearance_request(request_id)
    if request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Clearance request not found."})
    if request.status != "PENDING":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"message": "Only pending requests can be approved."})
    if repository.get_user_by_email(request.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"message": "An account already exists for this clearance request email."})

    token, token_hash, expires_at = new_activation_token()
    user = repository.create_user(
        full_name=request.full_name,
        email=request.email,
        role=request.requested_role,
        department=request.department,
        password_hash=None,
        is_active=False,
        activation_token_hash=token_hash,
        activation_expires_at=expires_at,
    )
    reviewed = repository.review_clearance_request(request_id, status_value="APPROVED", reviewer_id=admin.id)
    repository.log_event("CLEARANCE_APPROVED", actor_user_id=admin.id, target_user_id=user.id, metadata={"request_id": request_id})
    try:
        send_activation_email(
            recipient=request.email,
            activation_token=token,
            role=request.requested_role,
            settings=settings,
        )
    except SMTPDeliveryError as exc:
        repository.log_event(
            "CLEARANCE_EMAIL_DELIVERY_FAILED",
            actor_user_id=admin.id,
            target_user_id=user.id,
            metadata={"request_id": request_id, "category": exc.category},
        )
        return {
            "request": reviewed.public_dict(),
            "activation_email_sent": False,
            "email_delivery": {
                "status": "failed",
                "category": exc.category,
                "message": "Email delivery failed. The account was approved and a secure activation link is available.",
            },
            "activation_url": build_activation_url(token, settings),
        }
    return {
        "request": reviewed.public_dict(),
        "activation_email_sent": True,
        "email_delivery": {"status": "sent", "category": "smtp_sent", "message": "Activation email sent."},
    }


@router.post("/clearance-requests/{request_id}/decline")
def decline_clearance(
    request_id: int,
    _: DeclineInput,
    admin: User = Depends(require_admin),
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, object]:
    request = repository.get_clearance_request(request_id)
    if request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Clearance request not found."})
    try:
        reviewed = repository.review_clearance_request(request_id, status_value="DECLINED", reviewer_id=admin.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"message": str(exc)}) from exc
    repository.log_event("CLEARANCE_DECLINED", actor_user_id=admin.id, metadata={"request_id": request_id})
    return {"request": reviewed.public_dict()}
