import json
import os
import sqlite3
from typing import Dict

from models.agent_schemas import UserProfile

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "db", "agent_profiles.sqlite3")
DB_PATH = os.path.abspath(DB_PATH)


class ProfileStore:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_user_profile(self, user_id: str) -> UserProfile:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM agent_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return UserProfile(user_id=user_id)
        return UserProfile.model_validate(json.loads(row[0]))

    def upsert_user_profile(self, user_id: str, patch: Dict) -> UserProfile:
        current = self.get_user_profile(user_id)
        merged = current.model_dump()
        merged.update(patch)
        profile = UserProfile.model_validate(merged)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_profiles (user_id, profile_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, json.dumps(profile.model_dump())),
            )
            conn.commit()
        return profile
