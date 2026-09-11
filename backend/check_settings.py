from app.auth import get_settings

settings = get_settings()

print("ADMIN EMAIL:", settings.admin_email)
print("PASSWORD SET:", bool(settings.admin_initial_password))