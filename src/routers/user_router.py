from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from auth import get_current_user
from database import get_db
from models import User
from schemas import UserOut, UserUpdate
import shutil, os, uuid
from pathlib import Path

# Absolute path to uploads/avatars, relative to this file's location
BASE_DIR = Path(__file__).resolve().parent.parent
AVATAR_DIR = BASE_DIR / "uploads" / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/users", tags=["Users"])

@router.put("/me", response_model=UserOut)
def update_user_profile(
    user_update: UserUpdate, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name
    if user_update.email is not None:
        if user_update.email != current_user.email:
            existing = db.query(User).filter(User.email == user_update.email).first()
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = user_update.email
    if user_update.phone is not None:
        current_user.phone = user_update.phone
        
    db.commit()
    db.refresh(current_user)
    return current_user

@router.post("/me/avatar", response_model=UserOut)
def upload_avatar(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    
    # Store avatars using absolute path so location is consistent
    filepath = AVATAR_DIR / filename
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Build the public URL served via the /uploads static mount
    base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    current_user.avatar_url = f"{base_url}/uploads/avatars/{filename}"
    
    db.commit()
    db.refresh(current_user)
    return current_user
