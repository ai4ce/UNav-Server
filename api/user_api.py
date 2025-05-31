# api/user_api.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.db import SessionLocal, User, init_db
from models.schemas import UserCreate, UserLogin, UserInfo
from passlib.context import CryptContext
import jwt
import datetime
import os
from fastapi.security import OAuth2PasswordBearer

router = APIRouter()
init_db()

# --- Password hashing context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- JWT configuration ---
JWT_SECRET = os.getenv("JWT_SECRET", "REPLACE_ME_WITH_A_STRONG_KEY")
JWT_ALGORITHM = "HS256"

# --- OAuth2 scheme for token dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    """
    Dependency to provide a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against its hash.
    """
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """
    Generate a bcrypt hash of the given password.
    """
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_minutes: int = 60*24) -> str:
    """
    Create a JWT token encoding the provided data with expiration.
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

# -----------------------------------
# User registration endpoint
# -----------------------------------
@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user with hashed password.
    Raises 400 if username is already taken.
    """
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    hashed_pw = hash_password(user.password)
    new_user = User(username=user.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully", "id": new_user.id}

# -----------------------------------
# User login endpoint
# -----------------------------------
@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user credentials and return JWT access token.
    Raises 401 on invalid credentials.
    """
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_access_token(data={"sub": db_user.username, "id": db_user.id})
    return {"access_token": token, "token_type": "bearer"}

# -----------------------------------
# User logout endpoint
# -----------------------------------
@router.post("/logout")
def logout(user_id: str = Depends(get_user_id_from_token)):
    """
    Clear user session state in memory upon logout.
    """
    from core.unav_state import user_sessions
    user_sessions.pop(user_id, None)
    return {"msg": "User session cleared"}

# -----------------------------------
# Protected endpoint to get current user info
# -----------------------------------
@router.get("/me", response_model=UserInfo)
def get_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Retrieve information about the currently authenticated user.
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
