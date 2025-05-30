# api/user_api.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from db.db import SessionLocal, User, init_db
from models.schemas import UserCreate, UserLogin, UserInfo
from passlib.context import CryptContext
import jwt
import datetime
import os
import requests
import uuid

router = APIRouter()
init_db()

# --- Password hashing utility ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "REPLACE_ME_WITH_A_STRONG_KEY")
JWT_ALGORITHM = "HS256"

# --- Dependency to get DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def hash_password(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: int = 60*24):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ==============================
# Local registration and login
# ==============================

@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new local user account.
    """
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pw = hash_password(user.password)
    new_user = User(username=user.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully", "id": new_user.id}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate a local user and return JWT token.
    """
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(data={"sub": db_user.username, "id": db_user.id})
    return {"access_token": token, "token_type": "bearer"}

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@router.get("/me", response_model=UserInfo)
def get_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Get current logged-in user's info (protected route).
    """
    payload = decode_access_token(token)
    db_user = db.query(User).filter(User.username == payload["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")
    return UserInfo(
        id=db_user.id,
        username=db_user.username,
        created_at=db_user.created_at.strftime("%Y-%m-%d %H:%M:%S")
    )