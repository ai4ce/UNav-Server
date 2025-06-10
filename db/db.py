from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATA_ROOT
import os
import datetime

# Database connection string (SQLite local file)
DATABASE_URL = f"sqlite:///{os.path.join(DATA_ROOT, 'users.db')}"

# Create SQLAlchemy engine (for SQLite, disable thread check)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

class User(Base):
    """
    ORM model for the 'users' table.

    Fields:
        id: Primary key, unique user ID.
        email: User's unique email address (used for login).
        nickname: User's display name, can be changed and duplicated.
        hashed_password: Hashed user password.
        avatar_url: URL to user's avatar image, can be null.
        created_at: Timestamp when the user was created.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)   # Unique email address
    nickname = Column(String, nullable=False)                            # User's nickname (not unique)
    hashed_password = Column(String, nullable=False)                     # Hashed password
    avatar_blob = Column(LargeBinary, nullable=True)                       # Avatar blob (optional)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)      # User creation time

def init_db():
    """
    Initialize database tables based on ORM models.
    """
    Base.metadata.create_all(bind=engine)