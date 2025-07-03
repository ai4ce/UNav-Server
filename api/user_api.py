from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Form, Response
from sqlalchemy.orm import Session
from db.db import UserSessionLocal, User, init_user_db
from models.schemas import UserRegister, UserLogin, UserInfo
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
import jwt
import datetime
import os
import secrets
import time
import smtplib
from email.mime.text import MIMEText
from typing import Dict

router = APIRouter()
init_user_db()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "REPLACE_ME_WITH_A_STRONG_KEY")
JWT_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

VERIFICATION_CODES: Dict[str, tuple] = {}
CODE_EXPIRE_SECONDS = 300

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "unav.nyu@gmail.com"
SMTP_PASS = "fpse qert cfil wkhi"


def get_db():
    db = UserSessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_minutes: int = 60*24) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_user_id_from_token(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_access_token(token)
    return str(payload["id"])


def send_email_code(email: str, code: str):
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
    return ''.join(secrets.choice('0123456789') for _ in range(length))


@router.post("/send_verification_code")
def send_verification_code(data: dict, background_tasks: BackgroundTasks):
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
def register(user: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user with email, nickname, password, and verification code.
    """
    email = user.email.strip().lower()
    nickname = user.nickname.strip()
    password = user.password
    code = user.code

    if not email or not nickname or not password or not code:
        return {"error": "empty_field"}

    # Check verification code
    code_pair = VERIFICATION_CODES.get(email)
    now = time.time()
    if not code_pair or code_pair[0] != code or code_pair[1] < now:
        return {"error": "invalid_or_expired_code"}

    VERIFICATION_CODES.pop(email, None)

    # Check duplicate email
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        return {"error": "user_exists"}

    hashed_pw = hash_password(password)
    new_user = User(email=email, nickname=nickname, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully", "id": new_user.id, "nickname": nickname}


@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user credentials and return JWT access token.
    """
    if not user.email.strip() or not user.password.strip():
        return {"error": "empty_field"}
    db_user = db.query(User).filter(User.email == user.email.lower()).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        return {"error": "incorrect_credentials"}
    token = create_access_token(data={"sub": db_user.email, "id": db_user.id})
    avatar_url = f"/api/avatar/{db_user.id}" if db_user.avatar_blob else None
    return {
        "access_token": token,
        "token_type": "bearer",
        "nickname": db_user.nickname,
        "avatar_url": avatar_url,
    }


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
    Returns: {"id": ..., "email": ..., "nickname": ..., "avatar_url": ..., "created_at": ...}
    avatar_url will be the download endpoint: /api/avatar/{user_id} if avatar exists, else null.
    """
    payload = decode_access_token(token)
    db_user = db.query(User).filter(User.email == payload["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    avatar_url = f"/api/avatar/{db_user.id}" if db_user.avatar_blob else None
    return UserInfo(
        id=db_user.id,
        email=db_user.email,
        nickname=db_user.nickname,
        avatar_url=avatar_url,
        created_at=db_user.created_at.strftime("%Y-%m-%d %H:%M:%S")
    )


@router.post("/update_profile")
def update_profile(
    nickname: str = Form(None),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """
    Update the user's nickname.
    """
    payload = decode_access_token(token)
    db_user = db.query(User).filter(User.email == payload["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if nickname:
        db_user.nickname = nickname
        db.commit()
    return {"msg": "Profile updated", "nickname": db_user.nickname}


@router.post("/upload_avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Upload a new avatar for a user. The image content is stored in the database.
    """
    db_user = db.query(User).filter(User.email == email.lower()).first()
    if not db_user:
        return {"error": "User not found"}
    content = await file.read()
    db_user.avatar_blob = content
    db.commit()
    return {"msg": "Avatar uploaded"}


@router.get("/avatar/{user_id}")
def get_avatar(user_id: int, db: Session = Depends(get_db)):
    """
    Returns the user's avatar as an image (binary).
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user or db_user.avatar_blob is None:
        raise HTTPException(status_code=404, detail="Avatar not found")
    # By default, assume JPEG. Optionally add filetype check.
    return Response(content=db_user.avatar_blob, media_type="image/jpeg")
