# api/user_api.py

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from db.db import SessionLocal, User, init_db
from models.schemas import UserCreate, UserLogin, UserInfo
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
import jwt
import datetime
import os
import random
import string
import time
import smtplib
from email.mime.text import MIMEText
from typing import Dict

router = APIRouter()
init_db()

# --- Password hashing context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- JWT configuration ---
JWT_SECRET = os.getenv("JWT_SECRET", "REPLACE_ME_WITH_A_STRONG_KEY")
JWT_ALGORITHM = "HS256"

# --- OAuth2 scheme for token dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Email verification code cache ---
VERIFICATION_CODES: Dict[str, tuple] = {}  # {email: (code, expire_time)}
CODE_EXPIRE_SECONDS = 300  # 5 minutes

# --- SMTP Email config (replace with your real info in production) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "unav.nyu@gmail.com"
SMTP_PASS = "fpse qert cfil wkhi"  # Use an app password if needed


def get_db():
    """Dependency to provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_minutes: int = 60*24) -> str:
    """
    Create a JWT token with the provided data.
    """
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises HTTPException if token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_user_id_from_token(token: str = Depends(oauth2_scheme)) -> str:
    """
    Extract the user ID from a valid JWT token.
    """
    payload = decode_access_token(token)
    return str(payload["id"])


def send_email_code(email: str, code: str):
    """
    Send a 6-digit verification code to the user's email.
    This function uses SMTP for actual sending.
    """
    msg = MIMEText(f"Your UNav verification code is: {code}", "plain", "utf-8")
    msg['Subject'] = "UNav Verification Code"
    msg['From'] = SMTP_USER
    msg['To'] = email

    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [email], msg.as_string())
        server.quit()
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        raise RuntimeError("Failed to send verification email")


def generate_code(length=6):
    """Generate a random 6-digit verification code as string."""
    return ''.join(random.choices(string.digits, k=length))


@router.post("/send_verification_code")
def send_verification_code(data: dict, background_tasks: BackgroundTasks):
    """
    Endpoint to send a verification code to the specified email.
    Input: {"email": "..."}
    Returns: {"msg": "verification_code_sent"} or {"error": ...}
    """
    email = data.get("email", "").strip().lower()
    if not email or "@" not in email or len(email) > 60:
        return {"error": "invalid_email"}

    code = generate_code()
    expire_at = time.time() + CODE_EXPIRE_SECONDS
    VERIFICATION_CODES[email] = (code, expire_at)

    background_tasks.add_task(send_email_code, email, code)
    print(f"[DEBUG] Sent verification code {code} to {email}")

    return {"msg": "verification_code_sent"}


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: dict, db: Session = Depends(get_db)):
    """
    Register a new user with email, password, and verification code.
    Input: {"username": email, "password": "...", "code": "..."}
    Returns: {"msg": "..."} or {"error": "..."}
    """
    username = user.get("username", "").strip().lower()
    password = user.get("password", "")
    code = user.get("code", "")

    if not username or not password or not code:
        return {"error": "empty_field"}

    # Check verification code
    code_pair = VERIFICATION_CODES.get(username)
    now = time.time()
    if not code_pair or code_pair[0] != code or code_pair[1] < now:
        return {"error": "invalid_or_expired_code"}

    VERIFICATION_CODES.pop(username, None)

    # Check duplicate
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return {"error": "user_exists"}

    hashed_pw = hash_password(password)
    new_user = User(username=username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully", "id": new_user.id}


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user credentials and return JWT access token.
    Input: {"username": email, "password": "..."}
    Returns: {"access_token": "...", "token_type": "bearer"} or {"error": "..."}
    """
    if not user.username.strip() or not user.password.strip():
        return {"error": "empty_field"}
    db_user = db.query(User).filter(User.username == user.username.lower()).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        return {"error": "incorrect_credentials"}
    token = create_access_token(data={"sub": db_user.username, "id": db_user.id})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/logout")
def logout(user_id: str = Depends(get_user_id_from_token)):
    """
    Clear user session state in memory upon logout.
    """
    from core.unav_state import user_sessions
    user_sessions.pop(user_id, None)
    return {"msg": "User session cleared"}


@router.get("/me", response_model=UserInfo)
def get_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Retrieve information about the currently authenticated user.
    Returns: UserInfo or HTTP_401_UNAUTHORIZED if not found.
    """
    payload = decode_access_token(token)
    db_user = db.query(User).filter(User.username == payload["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return UserInfo(
        id=db_user.id,
        username=db_user.username,
        created_at=db_user.created_at.strftime("%Y-%m-%d %H:%M:%S")
    )
