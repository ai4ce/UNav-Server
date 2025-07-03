import os
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, LargeBinary, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATA_ROOT
import numpy as np
import cv2

# =========================== Ensure DATA_ROOT Exists ===========================
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT, exist_ok=True)

# =================== User Database (users.db, for user info only) ===================
USER_DB_URL = f"sqlite:///{os.path.join(DATA_ROOT, 'users.db')}"
user_engine = create_engine(USER_DB_URL, connect_args={"check_same_thread": False})
UserBase = declarative_base()
UserSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=user_engine)

class User(UserBase):
    """
    ORM model for the 'users' table.

    Fields:
        id: Primary key, unique user ID.
        email: User's unique email address (used for login).
        nickname: User's display name, can be changed and duplicated.
        hashed_password: Hashed user password.
        avatar_blob: Optional user avatar stored as binary data.
        created_at: Timestamp when the user was created.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    nickname = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    avatar_blob = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_user_db():
    """
    Initialize the users database tables according to ORM models.
    Should be called at application startup.
    """
    UserBase.metadata.create_all(bind=user_engine)

# ============== Navigation Log Database (log.db, for user navigation records) ==============
LOG_DB_URL = f"sqlite:///{os.path.join(DATA_ROOT, 'log.db')}"
log_engine = create_engine(LOG_DB_URL, connect_args={"check_same_thread": False})
LogBase = declarative_base()
LogSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=log_engine)

class UserNavigationRecord(LogBase):
    """
    ORM model for the 'user_navigation_records' table.
    Stores every navigation request for every user (including both success and failure).

    Fields:
        id: Primary key.
        user_id: Foreign key, links to 'users.id'.
        timestamp: Operation timestamp (UTC).
        query_image_path: Relative path of the navigation query image.
        floorplan_pose: floorplan pose.
        navigation_commands: JSON-serialized list of navigation commands.
        path: JSON-serialized navigation trajectory (list of waypoints, etc.).
        source_place_id, source_building_id, source_floor_id: Start point context.
        dest_place_id, dest_building_id, dest_floor_id: Destination context.
        destination_id: Final navigation destination node ID.
        status: Navigation result status ('success', 'failed', etc.).
        extra_info: Optional JSON field for diagnostics, timings, reasons, etc.
    """
    __tablename__ = "user_navigation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    query_image_path = Column(String, nullable=True)
    floorplan_pose = Column(JSON, nullable=False)
    navigation_commands = Column(JSON, nullable=True)
    path = Column(JSON, nullable=True)
    source_place_id = Column(String, nullable=True)
    source_building_id = Column(String, nullable=True)
    source_floor_id = Column(String, nullable=True)
    dest_place_id = Column(String, nullable=True)
    dest_building_id = Column(String, nullable=True)
    dest_floor_id = Column(String, nullable=True)
    destination_id = Column(String, nullable=True)
    status = Column(String, default="success", nullable=False)  # "success" or "failed"
    extra_info = Column(JSON, nullable=True)

def init_log_db():
    """
    Initialize the log database tables according to ORM models.
    Should be called at application startup.
    """
    LogBase.metadata.create_all(bind=log_engine)

# ===================== Navigation Data Logging Function =====================
def save_query_image(
    image: np.ndarray,
    data_root: str,
    user_id: int,
    source_place_id: str,
    source_building_id: str,
    source_floor_id: str,
    timestamp: datetime = None
) -> str:
    """
    Save user's query image to a structured path and return the relative path.

    Args:
        image (np.ndarray): Query image (BGR format).
        data_root (str): Root data directory.
        user_id (int): User's unique ID.
        source_place_id, source_building_id, source_floor_id: Navigation context.
        timestamp (datetime, optional): Filename timestamp. Defaults to UTC now.

    Returns:
        str: Relative file path, e.g.
        'test/123456/A12/B1/3F/20250630T154256123456.jpg'
    """
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()
    ts_str = timestamp.strftime("%Y%m%dT%H%M%S%f")
    rel_dir = os.path.join("test", str(user_id), str(source_place_id), str(source_building_id), str(source_floor_id))
    rel_path = os.path.join(rel_dir, f"{ts_str}.jpg")
    abs_dir = os.path.join(data_root, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    abs_path = os.path.join(data_root, rel_path)
    cv2.imwrite(abs_path, image)
    return rel_path

def log_navigation_record(
    user_id,
    query_image_path,
    floorplan_pose,
    navigation_commands,
    path,
    source_place_id,
    source_building_id,
    source_floor_id,
    dest_place_id,
    dest_building_id,
    dest_floor_id,
    destination_id,
    status="success",
    extra_info=None,
    timestamp=None,
):
    """
    Insert a new navigation record into the log database.

    Args:
        user_id (int): User's unique ID.
        query_image_path (str): Path to the query image.
        localization_result (dict): Localization results as JSON-serializable dict.
        navigation_commands (list): List of navigation commands (JSON-serializable).
        path (list): List of waypoints or coordinates (JSON-serializable).
        source_place_id (str): Source place identifier.
        source_building_id (str): Source building identifier.
        source_floor_id (str): Source floor identifier.
        dest_place_id (str): Destination place identifier.
        dest_building_id (str): Destination building identifier.
        dest_floor_id (str): Destination floor identifier.
        destination_id (str): Destination identifier.
        status (str): Navigation result status ("success", "failed", etc.).
        extra_info (dict, optional): Any additional information (JSON-serializable).
        timestamp (datetime, optional): Operation timestamp. If None, will use now (UTC).
    """
    db = LogSessionLocal()
    try:
        record = UserNavigationRecord(
            user_id=user_id,
            timestamp=timestamp or datetime.datetime.utcnow(),
            query_image_path=query_image_path,
            floorplan_pose=floorplan_pose,
            navigation_commands=navigation_commands,
            path=path,
            source_place_id=source_place_id,
            source_building_id=source_building_id,
            source_floor_id=source_floor_id,
            dest_place_id=dest_place_id,
            dest_building_id=dest_building_id,
            dest_floor_id=dest_floor_id,
            destination_id=destination_id,
            status=status,
            extra_info=extra_info,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()
