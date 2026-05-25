from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os, httpx, smtplib, secrets
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from database import get_db
from models import User, RefreshToken, PasswordResetToken
from schemas import (
    RegisterRequest, LoginRequest, TokenResponse,
    RefreshRequest, ForgotPasswordRequest, GoogleCallbackRequest,
    UserOut, MessageResponse, ResetPasswordRequest
)
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    verify_access_token, get_current_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

#  Config 
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:5173")
REFRESH_EXPIRE_DAYS  = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))
RESET_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", 60))

# Gmail SMTP config
EMAIL_HOST     = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER     = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_FROM     = os.getenv("EMAIL_FROM", EMAIL_USER)


#  Helper 
def _save_refresh_token(db: Session, user_id: int, token: str) -> None:
    """Persist a refresh token in the database."""
    rt = RefreshToken(
        token=token,
        user_id=user_id,
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_EXPIRE_DAYS)
    )
    db.add(rt)
    db.commit()


def _build_token_response(db: Session, user: User, remember_me: bool = False) -> TokenResponse:
    """Create access + refresh tokens and return the full TokenResponse."""
    access  = create_access_token({"sub": str(user.id)}, remember_me=remember_me)
    refresh = create_refresh_token()
    _save_refresh_token(db, user.id, refresh)

    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user=UserOut.model_validate(user)
    )


#  1. Register 
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    try:
        # Block duplicate email
        if db.query(User).filter(User.email == body.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        # Block duplicate username
        if db.query(User).filter(User.username == body.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")

        user = User(
            email=body.email,
            username=body.username,
            full_name=body.full_name,
            hashed_password=hash_password(body.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return _build_token_response(db, user)
    except Exception as e:
        db.rollback()
        return {"access_token": "", "refresh_token": "", "user": {"id": 0, "email": str(e), "full_name": str(type(e)), "username": "", "avatar_url": None, "is_active": False, "is_google_user": False}}


#  2. Login 
@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()

    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    return _build_token_response(db, user, remember_me=body.remember_me)


#  2b. Swagger UI Token Login 
@router.post("/token", include_in_schema=False)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Swagger UI sends the email in the 'username' field
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    
    response = _build_token_response(db, user)
    return {"access_token": response.access_token, "token_type": "bearer"}


#  3. Refresh token 
@router.post("/refresh", response_model=TokenResponse)
def refresh_token(body: RefreshRequest, db: Session = Depends(get_db)):
    rt = db.query(RefreshToken).filter(RefreshToken.token == body.refresh_token).first()

    if not rt:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    if rt.expires_at < datetime.utcnow():
        db.delete(rt)
        db.commit()
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = db.query(User).filter(User.id == rt.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    # Rotate the refresh token (delete old, issue new)
    db.delete(rt)
    db.commit()
    return _build_token_response(db, user)


#  4. Logout 
@router.post("/logout", response_model=MessageResponse)
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    rt = db.query(RefreshToken).filter(RefreshToken.token == body.refresh_token).first()
    if rt:
        db.delete(rt)
        db.commit()
    return MessageResponse(message="Logged out successfully")


#  5. Get current user 
@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ─── Email Helper ────────────────────────────────────────────────────────────
def send_reset_email(to_email: str, reset_url: str) -> None:
    """Send a password reset email via Gmail SMTP (runs in a background thread)."""
    if not EMAIL_USER or not EMAIL_PASSWORD:
        print(f"[EMAIL] SMTP not configured — reset URL: {reset_url}")
        return

    subject = "VocalArmor — Password Reset Request"
    html = f"""
    <html><body style="font-family:sans-serif;background:#050e10;color:#dfe8e6;padding:32px">
      <div style="max-width:520px;margin:auto;background:#0d1f24;border:1px solid rgba(255,255,255,0.1);
                  border-radius:16px;padding:40px">
        <h1 style="font-size:28px;color:#1dcfcf;margin:0 0 8px">VocalArmor</h1>
        <p style="color:#7ea8a4;margin:0 0 28px;font-size:14px">Deepfake Voice Detection</p>
        <h2 style="font-size:20px;margin:0 0 16px">Reset your password</h2>
        <p style="font-size:14px;color:#b0c4c0;line-height:1.6">
          We received a request to reset the password for your VocalArmor account.
          Click the button below to choose a new password. This link expires in
          <strong style="color:#1dcfcf">60 minutes</strong>.
        </p>
        <a href="{reset_url}"
           style="display:inline-block;margin:24px 0;padding:14px 32px;
                  background:linear-gradient(135deg,rgba(255,92,43,0.8),rgba(29,207,207,0.8));
                  color:#fff;text-decoration:none;border-radius:10px;
                  font-weight:700;font-size:14px;letter-spacing:0.05em">
          Reset Password →
        </a>
        <p style="font-size:12px;color:#5a7a76;margin:16px 0 0">
          If you didn't request this, you can safely ignore this email.
          Your password will not change.
        </p>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:24px 0">
        <p style="font-size:11px;color:#3a5a56">VocalArmor Security Team</p>
      </div>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"VocalArmor <{EMAIL_FROM}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        print(f"[EMAIL] Reset email sent to {to_email}")
    except Exception as e:
        print(f"[EMAIL] Failed to send reset email: {e}")


#  6. Forgot password 
@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Always return success — prevents email enumeration
    user = db.query(User).filter(User.email == body.email).first()
    if user:
        # Invalidate any existing unused tokens for this user
        db.query(PasswordResetToken).filter(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used == False
        ).delete()
        db.commit()

        # Generate a new secure token
        raw_token = secrets.token_urlsafe(48)
        prt = PasswordResetToken(
            token      = raw_token,
            user_id    = user.id,
            expires_at = datetime.utcnow() + timedelta(minutes=RESET_EXPIRE_MINUTES),
        )
        db.add(prt)
        db.commit()

        reset_url = f"{FRONTEND_URL}/reset-password?token={raw_token}"
        print(f"[AUTH] Password reset requested for: {user.email} | URL: {reset_url}")
        background_tasks.add_task(send_reset_email, user.email, reset_url)

    return MessageResponse(message="If that email is registered, a reset link has been sent")


#  6b. Reset password (verify token + update password) 
@router.post("/reset-password", response_model=MessageResponse)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    prt = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == body.token
    ).first()

    if not prt:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    if prt.used:
        raise HTTPException(status_code=400, detail="This reset link has already been used")
    if prt.expires_at < datetime.utcnow():
        db.delete(prt)
        db.commit()
        raise HTTPException(status_code=400, detail="Reset link has expired. Please request a new one")

    user = db.query(User).filter(User.id == prt.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=400, detail="User not found or inactive")

    # Update the password and mark token as used
    user.hashed_password = hash_password(body.new_password)
    prt.used = True
    db.commit()

    print(f"[AUTH] Password successfully reset for: {user.email}")
    return MessageResponse(message="Password updated successfully. You can now sign in.")


#  7. Google OAuth — initiate 
@router.get("/google")
def google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    params = "&".join([
        "response_type=code",
        f"client_id={GOOGLE_CLIENT_ID}",
        f"redirect_uri={GOOGLE_REDIRECT_URI}",
        "scope=openid%20email%20profile",
        "access_type=offline",
        "prompt=select_account consent",
    ])
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


#  8. Google OAuth — callback 
@router.get("/google/callback")
async def google_callback(code: str, db: Session = Depends(get_db)):
    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=no_code")

    async with httpx.AsyncClient() as client:
        # Step A: Exchange code for tokens
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":GOOGLE_CLIENT_ID,
                "client_secret":GOOGLE_CLIENT_SECRET,
                "redirect_uri":GOOGLE_REDIRECT_URI,
                "grant_type":"authorization_code",
            }
        )
        if token_res.status_code != 200:
            return RedirectResponse(f"{FRONTEND_URL}/login?error=token_exchange_failed")

        google_tokens = token_res.json()
        id_token = google_tokens.get("id_token", "")

        # Step B: Verify the ID token to get user info
        verify_res = await client.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
        )
        if verify_res.status_code != 200:
            return RedirectResponse(f"{FRONTEND_URL}/login?error=token_verify_failed")

        info = verify_res.json()

    google_id = info.get("sub")
    email = info.get("email")
    name = info.get("name", "")
    avatar = info.get("picture", "")

    if not email or not google_id:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=missing_user_info")

    # Step C: Find or create user
    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        # Check if email already exists (link Google to existing account)
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.google_id = google_id
            user.is_google_user = True
            user.avatar_url = avatar
        else:
            # Brand new user via Google
            user = User(
                email=email,
                full_name=name,
                google_id=google_id,
                is_google_user=True,
                avatar_url=avatar,
            )
            db.add(user)
        db.commit()
        db.refresh(user)

    # Step D: Issue tokens and redirect to frontend callback
    response = _build_token_response(db, user)
    redirect_url = (
        f"{FRONTEND_URL}/auth/callback"
        f"?token={response.access_token}"
        f"&refresh={response.refresh_token}"
    )
    return RedirectResponse(redirect_url)