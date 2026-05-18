from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os, httpx

from database import get_db
from models import User, RefreshToken
from schemas import (
    RegisterRequest, LoginRequest, TokenResponse,
    RefreshRequest, ForgotPasswordRequest, GoogleCallbackRequest,
    UserOut, MessageResponse
)
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    verify_access_token, get_current_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

#  Config 
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
REFRESH_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))


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


#  6. Forgot password 
@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    # Always return success to prevent email enumeration attacks
    user = db.query(User).filter(User.email == body.email).first()
    if user:
        # TODO: Send real password reset email via SendGrid / Resend / SMTP
        print(f"[AUTH] Password reset requested for: {user.email}")
    return MessageResponse(message="If that email exists, a reset link has been sent")


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