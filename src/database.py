from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./vocal_armor.db"
)

# SQLite needs check_same_thread=False, PostgreSQL does not
# We detect which one is being used and set connect_args accordingly
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def create_tables():
    from models import User, RefreshToken, PasswordResetToken
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _lazy_import():
    from models import User, RefreshToken, PasswordResetToken
    return User, RefreshToken, PasswordResetToken

try:
    from models import User, RefreshToken, PasswordResetToken
except Exception:
    User = None
    RefreshToken = None
    PasswordResetToken = None