from __future__ import annotations

import json
import os
import secrets
import time
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, Index, select, or_
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ----------------- CONFIG -----------------
def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


DB_URL = os.environ.get("DATABASE_URL", "").strip()
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is required")

# Some providers give postgres://, SQLAlchemy wants postgresql://
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

DB_CONNECT_TIMEOUT = _get_env_int("DB_CONNECT_TIMEOUT", 5)  # seconds
DB_SSLMODE = os.environ.get("DB_SSLMODE", "require").strip()  # "require" is safe for Render external URLs


# ----------------- DATABASE -----------------
# psycopg2 connect args support connect_timeout + sslmode
connect_args = {"connect_timeout": DB_CONNECT_TIMEOUT}
if DB_SSLMODE:
    connect_args["sslmode"] = DB_SSLMODE

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ----------------- FASTAPI APP -----------------
app = FastAPI(title="DeniTask Macro Hub", version="1.0")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    created_utc = Column(Float, nullable=False, default=lambda: time.time())


class MacroRow(Base):
    __tablename__ = "macros"
    id = Column(Integer, primary_key=True)

    name = Column(String(128), nullable=False, index=True)
    creator = Column(String(64), nullable=False, index=True)
    visibility = Column(String(16), nullable=False, index=True)  # public|private

    description = Column(Text, nullable=False, default="")
    instructions = Column(Text, nullable=False, default="")

    macro_json = Column(Text, nullable=False)  # Macro.to_dict() JSON

    created_utc = Column(Float, nullable=False, default=lambda: time.time())
    updated_utc = Column(Float, nullable=False, default=lambda: time.time())


Index("ix_macros_visibility_name", MacroRow.visibility, MacroRow.name)
Index("ix_macros_creator_visibility", MacroRow.creator, MacroRow.visibility)


@app.on_event("startup")
def _startup():
    """
    Create tables at startup.
    If DB is unreachable, fail fast instead of hanging.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # Raising here will make Render show a clear runtime error instead of hanging forever
        raise RuntimeError(f"Database init failed: {e}") from e


# ----------------- API MODELS -----------------
class RegisterIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)


class RegisterOut(BaseModel):
    username: str
    token: str


class MacroUploadIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    visibility: str = Field(pattern="^(public|private)$")
    description: str = Field(default="", max_length=4000)
    instructions: str = Field(default="", max_length=8000)
    macro: dict  # Macro.to_dict()


class MacroMetaOut(BaseModel):
    id: int
    name: str
    creator: str
    visibility: str
    description: str
    instructions: str
    updated_utc: float


class MacroGetOut(MacroMetaOut):
    macro: dict


def _require_user(token: Optional[str]) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Missing X-DeniTask-Token")

    with SessionLocal() as db:
        u = db.execute(select(User).where(User.token == token)).scalar_one_or_none()
        if not u:
            raise HTTPException(status_code=401, detail="Invalid token")
        return u


# ----------------- ROUTES -----------------
@app.get("/health")
def health():
    # keep /health lightweight (no DB query). If you want DB check, tell me.
    return {"ok": True, "service": "denitask-macro-hub"}


@app.post("/register", response_model=RegisterOut)
def register(payload: RegisterIn):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username required")

    with SessionLocal() as db:
        existing = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if existing:
            return RegisterOut(username=existing.username, token=existing.token)

        token = secrets.token_hex(24)
        u = User(username=username, token=token)
        db.add(u)
        db.commit()
        return RegisterOut(username=username, token=token)


@app.get("/macros", response_model=List[MacroMetaOut])
def list_macros(
    query: str = "",
    scope: str = "all",  # all|public|mine
    limit: int = 200,
    x_denitask_token: Optional[str] = Header(default=None, alias="X-DeniTask-Token"),
):
    user = _require_user(x_denitask_token)
    q = (query or "").strip()
    limit = max(1, min(500, int(limit)))

    with SessionLocal() as db:
        stmt = select(MacroRow)

        if scope == "public":
            stmt = stmt.where(MacroRow.visibility == "public")
        elif scope == "mine":
            stmt = stmt.where(MacroRow.creator == user.username)
        else:
            stmt = stmt.where(or_(MacroRow.visibility == "public", MacroRow.creator == user.username))

        if q:
            stmt = stmt.where(MacroRow.name.ilike(f"%{q}%"))

        stmt = stmt.order_by(MacroRow.updated_utc.desc()).limit(limit)

        rows = db.execute(stmt).scalars().all()
        return [
            MacroMetaOut(
                id=r.id,
                name=r.name,
                creator=r.creator,
                visibility=r.visibility,
                description=r.description,
                instructions=r.instructions,
                updated_utc=r.updated_utc,
            )
            for r in rows
        ]


@app.get("/macros/{macro_id}", response_model=MacroGetOut)
def get_macro(
    macro_id: int,
    x_denitask_token: Optional[str] = Header(default=None, alias="X-DeniTask-Token"),
):
    user = _require_user(x_denitask_token)

    with SessionLocal() as db:
        r = db.execute(select(MacroRow).where(MacroRow.id == int(macro_id))).scalar_one_or_none()
        if not r:
            raise HTTPException(status_code=404, detail="Macro not found")

        if r.visibility != "public" and r.creator != user.username:
            raise HTTPException(status_code=403, detail="Not allowed")

        return MacroGetOut(
            id=r.id,
            name=r.name,
            creator=r.creator,
            visibility=r.visibility,
            description=r.description,
            instructions=r.instructions,
            updated_utc=r.updated_utc,
            macro=json.loads(r.macro_json),
        )


@app.post("/macros/upload", response_model=MacroMetaOut)
def upload_macro(
    payload: MacroUploadIn,
    x_denitask_token: Optional[str] = Header(default=None, alias="X-DeniTask-Token"),
):
    user = _require_user(x_denitask_token)

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Macro name required")

    macro_str = json.dumps(payload.macro)
    if len(macro_str.encode("utf-8")) > 2_000_000:  # 2MB safety
        raise HTTPException(status_code=413, detail="Macro too large (max ~2MB)")

    now = time.time()

    with SessionLocal() as db:
        r = MacroRow(
            name=name,
            creator=user.username,
            visibility=payload.visibility,
            description=payload.description or "",
            instructions=payload.instructions or "",
            macro_json=macro_str,
            created_utc=now,
            updated_utc=now,
        )
        db.add(r)
        db.commit()
        db.refresh(r)

        return MacroMetaOut(
            id=r.id,
            name=r.name,
            creator=r.creator,
            visibility=r.visibility,
            description=r.description,
            instructions=r.instructions,
            updated_utc=r.updated_utc,
        )
