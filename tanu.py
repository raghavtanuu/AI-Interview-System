"""
AI-Based Interview System (Backend)
-----------------------------------
Single-file FastAPI backend that includes:
- JWT auth (register/login)
- SQLAlchemy models (MySQL-ready; SQLite fallback)
- Interview lifecycle APIs
- AI scoring with TF-IDF + cosine similaimport pytest"""
from __future__ import annotations

import os
import random
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

try:
    import jwt  # PyJWT
except Exception:  # pragma: no cover
    jwt = None

try:
    from passlib.context import CryptContext
except Exception:  # pragma: no cover
    CryptContext = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover
    SentenceTransformer = None
    util = None


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
APP_NAME = "AI Interview System API"
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# MySQL example:
# mysql+pymysql://user:password@localhost:3306/interview_ai
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_interview.db")
PROJECT_ROOT = Path(__file__).resolve().parent
# Portable background lookup order:
# 1) BACKGROUND_IMAGE_PATH env var
# 2) ./assets/interview-bg.png
# 3) this shared local asset (if present)
# 4) No image (route returns 404, CSS gradient still shows)
_bg_candidates = [
    str(PROJECT_ROOT / "assets" / "interview-bg.png"),
    "/Users/aarav/.cursor/projects/Users-aarav-NIRF/assets/c0d78e9d-0755-4c3c-a72d-4caf976cf09d-f1ab80c0-a85f-48f4-9878-0fafce182ebc.png",
]
BACKGROUND_IMAGE_PATH = os.getenv("BACKGROUND_IMAGE_PATH", "")
if not BACKGROUND_IMAGE_PATH:
    for _candidate in _bg_candidates:
        if os.path.exists(_candidate):
            BACKGROUND_IMAGE_PATH = _candidate
            break

if CryptContext is None:
    raise RuntimeError("Missing dependency: passlib. Install with `pip install passlib[bcrypt]`.")

# Use pbkdf2_sha256 for stable local compatibility (avoids bcrypt backend/version issues).
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
recruiter_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/recruiter/login")


# -----------------------------------------------------------------------------
# Database setup
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    interviews: Mapped[list["Interview"]] = relationship(back_populates="user")


class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="in_progress")
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    selection_decision: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="interviews")
    answers: Mapped[list["Answer"]] = relationship(back_populates="interview", cascade="all, delete-orphan")


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_answer: Mapped[str] = mapped_column(Text, nullable=False)
    tfidf_score: Mapped[float] = mapped_column(Float, default=0.0)
    semantic_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, default=0.0)
    feedback: Mapped[str] = mapped_column(Text, default="")

    interview: Mapped["Interview"] = relationship(back_populates="answers")


class Recruiter(Base):
    __tablename__ = "recruiters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    company: Mapped[str] = mapped_column(String(120), nullable=False)
    domain_focus: Mapped[str] = mapped_column(String(120), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class InterviewCreateRequest(BaseModel):
    role: str = Field(min_length=2, max_length=100)


class InterviewCreateResponse(BaseModel):
    interview_id: int
    role: str
    status: str


class AnswerSubmitRequest(BaseModel):
    question: str
    expected_answer: str
    candidate_answer: str


class AnswerSubmitResponse(BaseModel):
    answer_id: int
    tfidf_score: float
    semantic_score: float
    final_score: float
    feedback: str


class InterviewResultResponse(BaseModel):
    interview_id: int
    role: str
    status: str
    overall_score: float
    total_answers: int
    selection_decision: str
    chart_payload: dict
    detailed_feedback: list[dict]


class QuestionItem(BaseModel):
    question: str
    expected_answer: str


class InterviewStartResponse(BaseModel):
    interview_id: int
    total_questions: int
    questions: list[QuestionItem]


class RecruiterRegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    company: str = Field(min_length=2, max_length=120)
    domain_focus: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8, max_length=72)


class RecruiterCandidateRow(BaseModel):
    candidate_email: str
    domain: str
    score_pct: float
    decision: str
    completed_at: Optional[str]


class CandidateQuestionAnswerRow(BaseModel):
    question: str
    answer: str
    score_pct: float
    feedback: str


class RecruiterCandidateDetailRow(BaseModel):
    candidate_email: str
    domain: str
    score_pct: float
    decision: str
    completed_at: Optional[str]
    question_answers: list[CandidateQuestionAnswerRow]


# -----------------------------------------------------------------------------
# Security helpers
# -----------------------------------------------------------------------------
def hash_password(raw_password: str) -> str:
    if len(raw_password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password too long. Please use up to 72 characters."
        )
    try:
        return pwd_context.hash(raw_password)
    except Exception:
        raise HTTPException(status_code=500, detail="Password hashing failed. Please try again.")


def verify_password(raw_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(raw_password, hashed_password)
    except Exception:
        return False


def create_access_token(subject: str, actor_type: str = "candidate") -> str:
    if jwt is None:
        raise HTTPException(status_code=500, detail="PyJWT not installed. Install with `pip install PyJWT`.")
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "type": actor_type, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    if jwt is None:
        raise HTTPException(status_code=500, detail="PyJWT not installed. Install with `pip install PyJWT`.")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_access_token(token)
    if payload.get("type") != "candidate":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token for candidate APIs")
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.scalar(select(User).where(User.email == email))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_recruiter(token: str = Depends(recruiter_oauth2_scheme), db: Session = Depends(get_db)) -> Recruiter:
    payload = decode_access_token(token)
    if payload.get("type") != "recruiter":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token for recruiter APIs")
    email = payload.get("sub")
    recruiter = db.scalar(select(Recruiter).where(Recruiter.email == email))
    if not recruiter:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Recruiter not found")
    return recruiter


# -----------------------------------------------------------------------------
# AI scoring helpers
# -----------------------------------------------------------------------------
_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if SentenceTransformer is None:
        return None
    if _semantic_model is None:
        # Lightweight, good quality sentence embedding model.
        _semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _semantic_model


def tfidf_similarity(reference: str, answer: str) -> float:
    if not reference.strip() or not answer.strip():
        return 0.0
    if TfidfVectorizer is None or cosine_similarity is None:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([reference, answer])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(max(0.0, min(score, 1.0)))


def semantic_similarity(reference: str, answer: str) -> float:
    if not reference.strip() or not answer.strip():
        return 0.0
    model = _get_semantic_model()
    if model is None or util is None:
        return 0.0
    emb_ref = model.encode(reference, convert_to_tensor=True)
    emb_ans = model.encode(answer, convert_to_tensor=True)
    score = util.cos_sim(emb_ref, emb_ans).item()
    score = (score + 1) / 2  # Normalize from [-1,1] to [0,1]
    return float(max(0.0, min(score, 1.0)))


def keyword_overlap_score(reference: str, answer: str) -> float:
    """
    Lightweight semantic fallback when transformer model is unavailable.
    """
    ref_tokens = {t for t in re.findall(r"[a-zA-Z0-9_+#.]+", reference.lower()) if len(t) > 2}
    ans_tokens = {t for t in re.findall(r"[a-zA-Z0-9_+#.]+", answer.lower()) if len(t) > 2}
    if not ref_tokens or not ans_tokens:
        return 0.0
    intersection = len(ref_tokens & ans_tokens)
    # Use capped reference denominator so longer candidate answers are not penalized too harshly.
    denom = max(1, min(len(ref_tokens), 12))
    overlap = intersection / denom
    return float(max(0.0, min(overlap, 1.0)))


def key_concept_hits(reference: str, answer: str) -> tuple[int, int]:
    """
    Count concept hits between expected and candidate answers.
    Used to reward concise but conceptually correct 60s answers.
    """
    ref_tokens = [t for t in re.findall(r"[a-zA-Z0-9_+#.]+", reference.lower()) if len(t) > 2]
    ans_tokens = {t for t in re.findall(r"[a-zA-Z0-9_+#.]+", answer.lower()) if len(t) > 2}
    if not ref_tokens or not ans_tokens:
        return 0, 0

    # Keep unique tokens while preserving order.
    unique_ref = list(dict.fromkeys(ref_tokens))
    # Top concepts from expected answer, capped to avoid over-penalizing short references.
    top_concepts = unique_ref[: min(len(unique_ref), 12)]
    hits = len([t for t in top_concepts if t in ans_tokens])
    return hits, len(top_concepts)


def lexical_similarity(reference: str, answer: str) -> float:
    ratio = SequenceMatcher(None, reference.lower().strip(), answer.lower().strip()).ratio()
    return float(max(0.0, min(ratio, 1.0)))


def generate_feedback(final_score: float) -> str:
    pct = round(final_score * 100, 1)
    if final_score >= 0.82:
        return f"Excellent answer ({pct}%). Strong alignment with expected concepts."
    if final_score >= 0.62:
        return f"Good answer ({pct}%). Covers most required points; add more depth/examples."
    if final_score >= 0.42:
        return f"Average answer ({pct}%). Partially correct but missing key concepts."
    return f"Weak answer ({pct}%). Needs clearer structure and core concept coverage."


def compute_final_score(expected: str, candidate: str) -> tuple[float, float, float, str]:
    tfidf_score = tfidf_similarity(expected, candidate)
    sem_score = semantic_similarity(expected, candidate)
    concept_overlap = keyword_overlap_score(expected, candidate)
    concept_hits, concept_total = key_concept_hits(expected, candidate)

    # Fallback semantics to avoid overly harsh scores without transformer dependencies.
    if sem_score == 0:
        sem_score = (0.78 * concept_overlap) + (0.22 * lexical_similarity(expected, candidate))
    else:
        # Blend model similarity with concept overlap for short-but-correct answers.
        sem_score = (0.65 * sem_score) + (0.35 * concept_overlap)

    # Hybrid weighted scoring
    if sem_score > 0:
        final_score = (0.25 * tfidf_score) + (0.75 * sem_score)
    else:
        final_score = tfidf_score

    # Concept-hit boost: if 5+ key concepts are present, treat it as a strong response
    # even when the answer is concise due to strict interview timer.
    if concept_hits >= 5:
        final_score = max(final_score, 0.74)
    elif concept_hits >= 4:
        final_score = max(final_score, 0.66)
    elif concept_hits >= 3:
        final_score = max(final_score, 0.56)

    # Small boost for substantial answers with reasonable alignment.
    answer_len = len(candidate.strip().split())
    if answer_len >= 25 and final_score >= 0.40:
        final_score = min(1.0, final_score + 0.08)
    elif answer_len >= 40 and final_score >= 0.35:
        final_score = min(1.0, final_score + 0.10)
    elif answer_len >= 12 and concept_hits >= 5 and final_score >= 0.60:
        final_score = min(1.0, final_score + 0.04)

    # Guard against invalid range.
    final_score = float(max(0.0, min(final_score, 1.0)))

    feedback = generate_feedback(final_score)
    return tfidf_score, sem_score, final_score, feedback


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    _ensure_schema_updates()


def _ensure_schema_updates():
    """
    Small auto-migration for local/dev databases without Alembic.
    Ensures newly added columns exist in pre-existing SQLite/MySQL tables.
    """
    with engine.begin() as conn:
        try:
            cols = {row[1] for row in conn.execute(text("PRAGMA table_info(interviews)"))}
            if "selection_decision" not in cols:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN selection_decision VARCHAR(40)"))
            if "completed_at" not in cols:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN completed_at DATETIME"))
        except Exception:
            # For non-SQLite backends where PRAGMA is unavailable, try generic ALTER TABLE.
            try:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN selection_decision VARCHAR(40)"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE interviews ADD COLUMN completed_at DATETIME"))
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AI Interview Platform</title>
        <style>
          body {
            margin: 0;
            background:
              linear-gradient(rgba(6, 10, 26, 0.82), rgba(6, 10, 26, 0.82)),
              url('/assets/interview-bg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: Inter, -apple-system, sans-serif;
            color: #e5e7eb;
          }
          .wrap { width: min(980px, 95vw); margin: 18px auto; }
          .card { background: rgba(15, 23, 42, 0.82); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 16px; padding: 18px; margin-bottom: 12px; }
          h1, h2, h3 { margin: 0 0 8px; }
          p { margin: 0 0 8px; color: #cbd5e1; }
          .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
          .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
          .input, textarea, select { width: 100%; box-sizing: border-box; background: #0b1220; color: #e2e8f0; border: 1px solid #334155; border-radius: 10px; padding: 10px; margin: 6px 0; }
          .btn { background: #2563eb; color: #fff; border: none; border-radius: 10px; padding: 10px 14px; font-weight: 700; cursor: pointer; }
          .btn.secondary { background: #475569; }
          .btn.success { background: #16a34a; }
          .hidden { display: none; }
          .page.hidden { display: none; }
          .muted { color: #94a3b8; font-size: 13px; }
          .home-hero { padding: 22px; border-radius: 14px; border: 1px solid rgba(96,165,250,.35); background: linear-gradient(120deg, rgba(37,99,235,.25), rgba(15,23,42,.75)); }
          .home-hero h2 { margin: 0 0 8px; font-size: 28px; }
          .feature-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
          .feature-card { background: rgba(2,6,23,.58); border: 1px solid rgba(148,163,184,.22); border-radius: 12px; padding: 12px; }
          .feature-card h4 { margin: 0 0 4px; font-size: 14px; color: #bfdbfe; }
          .feature-card p { margin: 0; font-size: 12px; color: #cbd5e1; }
          .role-split { border-top: 1px solid #334155; margin-top: 14px; padding-top: 14px; }
          #timer { color: #fbbf24; font-weight: 700; }
          #log { white-space: pre-wrap; min-height: 64px; background: #020617; border: 1px solid #1f2937; border-radius: 10px; padding: 10px; }
          video { width: 240px; height: 150px; border-radius: 10px; border: 1px solid #334155; background: #000; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="card">
            <h1>AI Interview Platform</h1>
            <p>Login/Register → Integrity Guidelines → Camera/Mic Check → Random 10 Questions (timed) → AI Evaluation + Selection decision.</p>
          </div>

          <div id="authPage" class="page">
          <div class="card">
            <h3>1) Register or Login</h3>
            <p class="muted">Candidate section</p>
            <div class="grid2">
              <div>
                <label>Name</label>
                <input class="input" id="regName" placeholder="Your name" />
              </div>
              <div>
                <label>Email</label>
                <input class="input" id="regEmail" placeholder="you@example.com" />
              </div>
            </div>
            <div class="grid2">
              <div>
                <label>Password</label>
                <input class="input" id="regPass" type="password" placeholder="minimum 8 chars" />
              </div>
              <div>
                <label>Login password</label>
                <input class="input" id="loginPass" type="password" placeholder="for login" />
              </div>
            </div>
            <div class="row">
              <button class="btn" onclick="registerUser()">Register</button>
              <button class="btn secondary" onclick="loginUser()">Login</button>
            </div>
            <p class="muted">Token status: <span id="tokenStatus">Not logged in</span></p>

            <div class="role-split">
              <p class="muted">Recruiter section</p>
              <div class="grid2">
                <div>
                  <label>Recruiter Name</label>
                  <input class="input" id="recruiterName" placeholder="Recruiter name" />
                </div>
                <div>
                  <label>Recruiter Email</label>
                  <input class="input" id="recruiterEmail" placeholder="recruiter@company.com" />
                </div>
              </div>
              <div class="grid2">
                <div>
                  <label>Company</label>
                  <input class="input" id="recruiterCompany" placeholder="Company name" />
                </div>
                <div>
                  <label>Domain Focus</label>
                  <input class="input" id="recruiterDomain" placeholder="Python / Java / JS" />
                </div>
              </div>
              <div class="grid2">
                <div>
                  <label>Recruiter Password</label>
                  <input class="input" id="recruiterPass" type="password" placeholder="minimum 8 chars" />
                </div>
                <div>
                  <label>Recruiter Login Password</label>
                  <input class="input" id="recruiterLoginPass" type="password" placeholder="for recruiter login" />
                </div>
              </div>
              <div class="row">
                <button class="btn" onclick="registerRecruiter()">Recruiter Register</button>
                <button class="btn secondary" onclick="loginRecruiter()">Recruiter Login</button>
              </div>
            </div>
          </div>
          </div>

          <div id="homePage" class="page hidden">
            <div class="card">
              <div class="home-hero">
                <h2>Welcome to AI Interview Hub</h2>
                <p>Your interview workspace is ready. Complete guidelines, pass device checks, and start your timed interview.</p>
                <div class="feature-grid">
                  <div class="feature-card">
                    <h4>Dynamic Questions</h4>
                    <p>Randomized 15-question set based on your selected domain.</p>
                  </div>
                  <div class="feature-card">
                    <h4>Voice + Typing</h4>
                    <p>Answer with speech-to-text or type directly with timer support.</p>
                  </div>
                  <div class="feature-card">
                    <h4>Integrity Guard</h4>
                    <p>Tab switch/minimize is monitored for fair interview conduct.</p>
                  </div>
                </div>
              </div>
              <p class="muted" id="homeAlert" style="margin-top:10px;"></p>
              <div class="row">
                <button class="btn" onclick="goToGuidelinesPage()">Go to Guidelines</button>
                <button class="btn secondary" onclick="setLog('Flow: Guidelines → Device Check → Start Interview → Answer 15 Questions → Final Evaluation')">How It Works</button>
              </div>
            </div>
          </div>

          <div id="guidelinePage" class="page hidden">
          <div class="card" id="guidelineCard">
            <h3>2) Guidelines & Integrity</h3>
            <p>- Keep camera and mic ON for entire interview.</p>
            <p>- No external help, tab switching, or copied responses.</p>
            <p>- You will get 15 random questions with 60s timer per question.</p>
            <label><input type="checkbox" id="integrityCheck" /> I agree to interview integrity rules.</label>
            <div class="row"><button class="btn success" onclick="goToDeviceCheck()">Proceed</button></div>
          </div>
          </div>

          <div id="interviewPage" class="page hidden">
          <div class="card hidden" id="deviceCard">
            <h3>3) Camera & Mic Integrity Check</h3>
            <video id="camPreview" autoplay muted playsinline></video>
            <p class="muted" id="deviceStatus">Device not checked.</p>
            <div class="row">
              <button class="btn" onclick="checkDevices()">Check Camera + Mic</button>
            </div>
          </div>

          <div class="card hidden" id="startCard">
            <h3>4) Start Interview</h3>
            <label>Role</label>
            <select id="role" class="input">
              <option value="Python">Python</option>
              <option value="Java">Java</option>
              <option value="JavaScript">JavaScript</option>
              <option value="HTML">HTML</option>
              <option value="CSS">CSS</option>
            </select>
            <div class="row">
              <button class="btn" onclick="startInterview()">Start Interview</button>
            </div>
            <p class="muted">Interview ID: <span id="interviewId">-</span></p>
          </div>

          <div class="card hidden" id="interviewCard">
            <h3>5) Interview In Progress</h3>
            <p>Question <b><span id="qNo">0</span>/15</b> | Time Left: <span id="timer">60s</span></p>
            <label>Question</label>
            <textarea id="questionText" rows="3" class="input" readonly></textarea>
            <label>Your Answer</label>
            <textarea id="candidateAnswer" rows="5" class="input" placeholder="Type your answer..."></textarea>
            <div class="row">
              <button class="btn secondary" onclick="startListening()">Start Mic</button>
              <button class="btn secondary" onclick="stopListening()">Stop Mic</button>
              <button class="btn" onclick="submitCurrentAnswer()">Submit & Next</button>
              <button class="btn success" onclick="completeInterview()">Submit Interview</button>
            </div>
            <p class="muted" id="speechStatus">Speech status: idle.</p>
          </div>

          <div class="card">
            <h3>6) Live Status</h3>
            <div id="log">Ready.</div>
          </div>
          </div>

          <div id="resultPage" class="page hidden">
            <div class="card">
              <h3>Interview Result</h3>
              <p class="muted" id="resultSummary">Submit interview to view final evaluation.</p>
            </div>
            <div class="card">
              <h3>Question-wise Performance</h3>
              <div id="resultDetails"></div>
            </div>
          </div>

          <div id="recruiterPage" class="page hidden">
            <div class="card">
              <h3>Recruiter Dashboard</h3>
              <p class="muted" id="recruiterSummary">Login as recruiter to view candidate performance.</p>
              <div class="row">
                <button class="btn secondary" onclick="loadRecruiterCandidates()">Refresh Candidates</button>
                <button class="btn" onclick="showPage('authPage')">Logout</button>
              </div>
            </div>
            <div class="card">
              <h3>Candidate Details</h3>
              <div id="recruiterCandidates">No candidate data loaded.</div>
            </div>
          </div>
        </div>

        <script>
          const API = "";
          let token = null;
          let interviewId = null;
          let questionBank = [];
          let currentIndex = 0;
          let timerValue = 60;
          let timerHandle = null;
          let recognition = null;
          let stream = null;
          let actorRole = "candidate";
          let deviceReady = false;
          let interviewActive = false;
          let integrityWarnings = 0;
          let banned = false;

          function showPage(pageId) {
            ["authPage", "homePage", "guidelinePage", "interviewPage", "resultPage", "recruiterPage"].forEach((id) => {
              const el = document.getElementById(id);
              if (!el) return;
              el.classList.toggle("hidden", id !== pageId);
            });
          }

          function setLog(msg) {
            document.getElementById("log").textContent = msg;
          }

          function authHeaders() {
            return token ? { "Authorization": "Bearer " + token } : {};
          }

          async function registerUser() {
            const payload = {
              full_name: document.getElementById("regName").value,
              email: document.getElementById("regEmail").value,
              password: document.getElementById("regPass").value
            };
            const res = await fetch(API + "/auth/register", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) return setLog("Register failed: " + JSON.stringify(data));
            token = data.access_token;
            actorRole = "candidate";
            document.getElementById("tokenStatus").textContent = "Logged in";
            showPage("homePage");
            setLog("Registered and logged in successfully.");
          }

          async function loginUser() {
            const form = new URLSearchParams();
            form.append("username", document.getElementById("regEmail").value);
            form.append("password", document.getElementById("loginPass").value);
            const res = await fetch(API + "/auth/login", {
              method: "POST",
              headers: { "Content-Type": "application/x-www-form-urlencoded" },
              body: form
            });
            const data = await res.json();
            if (!res.ok) return setLog("Login failed: " + JSON.stringify(data));
            token = data.access_token;
            actorRole = "candidate";
            document.getElementById("tokenStatus").textContent = "Logged in";
            showPage("homePage");
            setLog("Logged in successfully.");
          }

          async function registerRecruiter() {
            const payload = {
              full_name: document.getElementById("recruiterName").value,
              email: document.getElementById("recruiterEmail").value,
              company: document.getElementById("recruiterCompany").value,
              domain_focus: document.getElementById("recruiterDomain").value,
              password: document.getElementById("recruiterPass").value
            };
            const res = await fetch(API + "/recruiter/register", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) return setLog("Recruiter register failed: " + JSON.stringify(data));
            token = data.access_token;
            actorRole = "recruiter";
            document.getElementById("tokenStatus").textContent = "Recruiter logged in";
            showPage("recruiterPage");
            setLog("Recruiter account created and logged in.");
            await loadRecruiterCandidates();
          }

          async function loginRecruiter() {
            const form = new URLSearchParams();
            form.append("username", document.getElementById("recruiterEmail").value);
            form.append("password", document.getElementById("recruiterLoginPass").value);
            const res = await fetch(API + "/recruiter/login", {
              method: "POST",
              headers: { "Content-Type": "application/x-www-form-urlencoded" },
              body: form
            });
            const data = await res.json();
            if (!res.ok) return setLog("Recruiter login failed: " + JSON.stringify(data));
            token = data.access_token;
            actorRole = "recruiter";
            document.getElementById("tokenStatus").textContent = "Recruiter logged in";
            showPage("recruiterPage");
            setLog("Recruiter logged in successfully.");
            await loadRecruiterCandidates();
          }

          async function loadRecruiterCandidates() {
            if (actorRole !== "recruiter") return setLog("Please login as recruiter first.");
            const res = await fetch(API + "/recruiter/candidates/details", {
              headers: { ...authHeaders() }
            });
            const data = await res.json();
            if (!res.ok) return setLog("Failed to load recruiter candidates: " + JSON.stringify(data));
            document.getElementById("recruiterSummary").textContent = "Total candidates: " + data.length;
            const html = data.map((c, idx) => {
              const qa = (c.question_answers || []).map((q, i) =>
                "<div class='feature-card' style='margin-top:6px;'>" +
                "<b>Q" + (i + 1) + ":</b> " + q.question + "<br/>" +
                "<b>Answer:</b> " + q.answer + "<br/>" +
                "<b>Score:</b> " + q.score_pct + "% | " + q.feedback +
                "</div>"
              ).join("");
              return (
                "<div class='feature-card' style='margin-bottom:10px;'>" +
                "<b>Candidate " + (idx + 1) + "</b><br/>" +
                "Email: " + c.candidate_email + "<br/>" +
                "Domain: " + c.domain + "<br/>" +
                "Overall: " + c.score_pct + "% | Decision: " + c.decision + "<br/>" +
                qa +
                "</div>"
              );
            }).join("");
            document.getElementById("recruiterCandidates").innerHTML = html || "<p class='muted'>No candidate interviews found.</p>";
          }

          function goToGuidelinesPage() {
            if (!token) return setLog("Please login first.");
            showPage("guidelinePage");
          }

          function goToDeviceCheck() {
            if (!token) return setLog("Please login first.");
            if (!document.getElementById("integrityCheck").checked) return setLog("Please accept integrity rules.");
            showPage("interviewPage");
            document.getElementById("deviceCard").classList.remove("hidden");
            setLog("Now check your camera and microphone.");
          }

          async function checkDevices() {
            try {
              stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
              document.getElementById("camPreview").srcObject = stream;
              deviceReady = true;
              document.getElementById("deviceStatus").textContent = "Camera and mic look good.";
              document.getElementById("startCard").classList.remove("hidden");
              setLog("Device checks passed. You can now start interview.");
            } catch (e) {
              deviceReady = false;
              setLog("Camera/mic permission failed. Please allow access.");
              document.getElementById("deviceStatus").textContent = "Device check failed.";
            }
          }

          async function startInterview() {
            if (!token) return setLog("Please register/login first.");
            if (banned) return setLog("You are blocked for this session due to integrity violations.");
            if (!deviceReady) return setLog("Please complete camera/mic check first.");
            const role = document.getElementById("role").value;
            const res = await fetch(API + "/interviews", {
              method: "POST",
              headers: { "Content-Type": "application/json", ...authHeaders() },
              body: JSON.stringify({ role })
            });
            const data = await res.json();
            if (!res.ok) return setLog("Start interview failed: " + JSON.stringify(data));
            interviewId = data.interview_id;
            document.getElementById("interviewId").textContent = interviewId;
            const qres = await fetch(API + "/interviews/" + interviewId + "/start", {
              method: "POST",
              headers: { ...authHeaders() }
            });
            const qdata = await qres.json();
            if (!qres.ok) return setLog("Question generation failed: " + JSON.stringify(qdata));
            questionBank = qdata.questions || [];
            currentIndex = 0;
            integrityWarnings = 0;
            interviewActive = true;
            document.getElementById("interviewCard").classList.remove("hidden");
            showQuestion();
            setLog("Interview started with random 15 questions.");
          }

          function showQuestion() {
            if (currentIndex >= questionBank.length) {
              setLog("All questions attempted. Click Submit Interview.");
              return;
            }
            const q = questionBank[currentIndex];
            document.getElementById("qNo").textContent = String(currentIndex + 1);
            document.getElementById("questionText").value = q.question;
            document.getElementById("candidateAnswer").value = "";
            startTimer();
          }

          function startTimer() {
            clearInterval(timerHandle);
            timerValue = 60;
            document.getElementById("timer").textContent = "60s";
            timerHandle = setInterval(() => {
              timerValue -= 1;
              document.getElementById("timer").textContent = timerValue + "s";
              if (timerValue <= 0) {
                clearInterval(timerHandle);
                submitCurrentAnswer(true);
              }
            }, 1000);
          }

          function startListening() {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) {
              document.getElementById("speechStatus").textContent = "Speech recognition not supported in this browser.";
              return;
            }
            recognition = new SR();
            recognition.lang = "en-US";
            recognition.interimResults = true;
            recognition.onstart = () => document.getElementById("speechStatus").textContent = "Speech status: listening...";
            recognition.onresult = (event) => {
              let text = "";
              for (let i = event.resultIndex; i < event.results.length; i++) text += event.results[i][0].transcript + " ";
              document.getElementById("candidateAnswer").value = text.trim();
            };
            recognition.onend = () => document.getElementById("speechStatus").textContent = "Speech status: stopped.";
            recognition.start();
          }

          function stopListening() {
            if (recognition) recognition.stop();
          }

          async function submitCurrentAnswer(auto = false) {
            if (!token || !interviewId) return setLog("Start interview first.");
            if (currentIndex >= questionBank.length) return;
            clearInterval(timerHandle);
            const qa = questionBank[currentIndex];
            const candidate = document.getElementById("candidateAnswer").value || (auto ? "No response in allotted time." : "");
            if (!qa || !candidate.trim()) return setLog("Answer is required.");
            const payload = {
              question: qa.question,
              expected_answer: qa.expected_answer,
              candidate_answer: candidate
            };
            const res = await fetch(API + "/interviews/" + interviewId + "/answers", {
              method: "POST",
              headers: { "Content-Type": "application/json", ...authHeaders() },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (!res.ok) return setLog("Submit failed: " + JSON.stringify(data));
            currentIndex += 1;
            if (currentIndex < questionBank.length) {
              setLog("Answer submitted. Moving to question " + (currentIndex + 1) + ".");
              showQuestion();
            } else {
              document.getElementById("timer").textContent = "Done";
              setLog("All 15 answers submitted. Click Submit Interview.");
            }
          }

          async function completeInterview() {
            if (!token || !interviewId) return setLog("Start interview first.");
            if (currentIndex < questionBank.length) {
              const remaining = questionBank.length - currentIndex;
              const ok = window.confirm(
                "You still have " + remaining + " unanswered question(s).\\n" +
                "If you submit now, remaining questions will be marked 0%. Continue?"
              );
              if (!ok) return setLog("Continue interview to answer remaining questions.");
              setLog("Early submission accepted. Remaining unanswered questions will be marked 0%.");
            }
            interviewActive = false;
            const res = await fetch(API + "/interviews/" + interviewId + "/complete", {
              method: "POST",
              headers: { ...authHeaders() }
            });
            const data = await res.json();
            if (!res.ok) return setLog("Complete failed: " + JSON.stringify(data));
            renderResultPage(data);
          }

          function renderResultPage(data) {
            const summary =
              "Role: " + data.role +
              " | Overall Score: " + data.overall_score + "%" +
              " | Decision: " + data.selection_decision +
              " | Answers: " + data.total_answers + "/15";
            document.getElementById("resultSummary").textContent = summary;

            const details = (data.detailed_feedback || []).map((x, i) =>
              "<div class='feature-card' style='margin-bottom:8px;'>" +
              "<b>Q" + (i + 1) + "</b> - Score: " + x.final_score_pct + "%<br/>" +
              "<span class='muted'>" + x.question + "</span><br/>" +
              x.feedback +
              "</div>"
            ).join("");
            document.getElementById("resultDetails").innerHTML = details || "<p class='muted'>No performance details available.</p>";
            showPage("resultPage");
            setLog("Interview submitted successfully.");
          }

          function terminateForIntegrityViolation() {
            interviewActive = false;
            banned = true;
            clearInterval(timerHandle);
            showPage("homePage");
            document.getElementById("homeAlert").textContent =
              "Interview terminated: repeated tab/minimize violations detected. Session blocked.";
            setLog("Interview terminated due to integrity violation.");
          }

          document.addEventListener("visibilitychange", () => {
            if (!interviewActive) return;
            if (document.hidden) {
              integrityWarnings += 1;
              if (integrityWarnings >= 2) {
                terminateForIntegrityViolation();
              } else {
                setLog("Warning: Tab switch/minimize detected. Another violation will terminate interview.");
              }
            }
          });
        </script>
      </body>
    </html>
    """


@app.get("/assets/interview-bg")
def interview_background():
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        return FileResponse(BACKGROUND_IMAGE_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Background image not found")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": APP_NAME, "timestamp": datetime.utcnow().isoformat()}


# -----------------------------------------------------------------------------
# Auth APIs
# -----------------------------------------------------------------------------
@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        full_name=payload.full_name,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    token = create_access_token(subject=user.email, actor_type="candidate")
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form_data.username))
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(subject=user.email, actor_type="candidate")
    return TokenResponse(access_token=token)


@app.post("/recruiter/register", response_model=TokenResponse, status_code=201)
def recruiter_register(payload: RecruiterRegisterRequest, db: Session = Depends(get_db)):
    existing = db.scalar(select(Recruiter).where(Recruiter.email == payload.email))
    if existing:
        raise HTTPException(status_code=409, detail="Recruiter email already registered")
    recruiter = Recruiter(
        full_name=payload.full_name,
        email=payload.email,
        company=payload.company,
        domain_focus=payload.domain_focus,
        hashed_password=hash_password(payload.password),
    )
    db.add(recruiter)
    db.commit()
    token = create_access_token(subject=recruiter.email, actor_type="recruiter")
    return TokenResponse(access_token=token)


@app.post("/recruiter/login", response_model=TokenResponse)
def recruiter_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    recruiter = db.scalar(select(Recruiter).where(Recruiter.email == form_data.username))
    if not recruiter or not verify_password(form_data.password, recruiter.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid recruiter credentials")
    token = create_access_token(subject=recruiter.email, actor_type="recruiter")
    return TokenResponse(access_token=token)


# -----------------------------------------------------------------------------
# Interview APIs
# -----------------------------------------------------------------------------
def build_data_analyst_bank() -> list[QuestionItem]:
    return build_role_bank("PY", [
        ("python data structures", "differences between list tuple set dict and their use cases"),
        ("object oriented programming", "class inheritance encapsulation and polymorphism"),
        ("error handling", "try except finally and meaningful exception usage"),
        ("iterators and generators", "memory-efficient iteration and yield behavior"),
        ("decorators", "function wrapping and reusable cross-cutting logic"),
        ("virtual environments", "dependency isolation and reproducible setup"),
        ("file handling", "safe read write with context managers"),
        ("testing", "unit tests and edge-case coverage"),
    ])


def build_hr_bank() -> list[QuestionItem]:
    return build_role_bank("JAVA", [
        ("oop in java", "class hierarchy interfaces and abstraction"),
        ("jvm and memory", "heap stack garbage collection basics"),
        ("collections framework", "list set map and performance considerations"),
        ("exception handling", "checked vs unchecked and custom exceptions"),
        ("multithreading", "thread lifecycle synchronization and race conditions"),
        ("stream api", "functional operations map filter reduce"),
        ("spring boot basics", "dependency injection and REST endpoint design"),
        ("testing in java", "junit and mocking fundamentals"),
    ])


def build_javascript_bank() -> list[QuestionItem]:
    return build_role_bank("JS", [
        ("event loop", "call stack callback queue and asynchronous execution"),
        ("closures", "lexical scoping and persistent references"),
        ("promises and async await", "non-blocking flow and error handling"),
        ("dom manipulation", "selectors events and dynamic rendering"),
        ("es6 features", "let const destructuring spread and modules"),
        ("hoisting and scope", "var vs let const and temporal dead zone"),
        ("api integration", "fetch lifecycle and response handling"),
        ("performance optimization", "debounce throttle lazy loading"),
    ])


def build_html_bank() -> list[QuestionItem]:
    return build_role_bank("HTML", [
        ("semantic html", "proper tags improve accessibility and SEO"),
        ("forms and validation", "input types labels and client-side constraints"),
        ("accessibility basics", "aria roles keyboard navigation and contrast"),
        ("document structure", "head body meta and content hierarchy"),
        ("media embedding", "audio video and fallback content"),
        ("seo fundamentals", "meta tags heading usage and crawlability"),
        ("table structures", "thead tbody scope and readable tabular data"),
        ("responsive markup", "viewport and fluid content structure"),
    ])


def build_css_bank() -> list[QuestionItem]:
    return build_role_bank("CSS", [
        ("box model", "content padding border margin behavior"),
        ("flexbox", "alignment distribution and responsive layouts"),
        ("css grid", "two-dimensional layout and area planning"),
        ("positioning", "relative absolute fixed sticky usage"),
        ("specificity and cascade", "selector priority and style override control"),
        ("responsive design", "media queries and mobile-first strategy"),
        ("animations and transitions", "smooth interaction feedback"),
        ("css architecture", "maintainable naming and scalable styling"),
    ])


def build_role_bank(role_tag: str, topics: list[tuple[str, str]]) -> list[QuestionItem]:
    prompt_templates = [
        "Explain your approach to {topic} in a real project.",
        "How would you solve a business problem using {topic}?",
        "What are common mistakes in {topic}, and how do you avoid them?",
        "Describe a production scenario where {topic} was critical.",
        "How do you measure success for work related to {topic}?",
    ]
    bank: list[QuestionItem] = []
    for i in range(1, 36):
        for topic, expected in topics:
            template = prompt_templates[(i + len(topic)) % len(prompt_templates)]
            question = template.format(topic=topic)
            expected_answer = (
                f"A strong answer should cover {expected}, mention trade-offs, "
                "and include practical implementation details."
            )
            bank.append(QuestionItem(question=f"[{role_tag}-{i}] {question}", expected_answer=expected_answer))
    return bank[:260]


QUESTION_BANK = {
    "python": build_data_analyst_bank(),
    "java": build_hr_bank(),
    "javascript": build_javascript_bank(),
    "html": build_html_bank(),
    "css": build_css_bank(),
}
REQUIRED_QUESTIONS = 15
INTERVIEW_QUESTION_SETS: dict[int, list[QuestionItem]] = {}


@app.get("/interview-questions")
def get_interview_questions(role: str):
    role_key = role.strip().lower()
    questions = QUESTION_BANK.get(role_key) or QUESTION_BANK["python"]
    return {"role": role, "questions": [q.model_dump() for q in questions]}


@app.post("/interviews/{interview_id}/start", response_model=InterviewStartResponse)
def start_interview_questions(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = db.scalar(select(Interview).where(Interview.id == interview_id, Interview.user_id == current_user.id))
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    role_key = interview.role.strip().lower()
    pool = QUESTION_BANK.get(role_key) or QUESTION_BANK["python"]
    selected = random.sample(pool, k=min(REQUIRED_QUESTIONS, len(pool)))
    INTERVIEW_QUESTION_SETS[interview_id] = selected
    return InterviewStartResponse(interview_id=interview_id, total_questions=len(selected), questions=selected)


@app.post("/interviews", response_model=InterviewCreateResponse)
def create_interview(
    payload: InterviewCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = Interview(user_id=current_user.id, role=payload.role, status="in_progress")
    db.add(interview)
    db.commit()
    db.refresh(interview)
    return InterviewCreateResponse(interview_id=interview.id, role=interview.role, status=interview.status)


@app.post("/interviews/{interview_id}/answers", response_model=AnswerSubmitResponse)
def submit_answer(
    interview_id: int,
    payload: AnswerSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = db.scalar(
        select(Interview).where(Interview.id == interview_id, Interview.user_id == current_user.id)
    )
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    if interview.status == "completed":
        raise HTTPException(status_code=400, detail="Interview already completed")
    if len(interview.answers) >= REQUIRED_QUESTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {REQUIRED_QUESTIONS} answers allowed for this interview.",
        )

    tfidf_score, sem_score, final_score, feedback = compute_final_score(
        expected=payload.expected_answer,
        candidate=payload.candidate_answer,
    )

    answer = Answer(
        interview_id=interview.id,
        question=payload.question,
        expected_answer=payload.expected_answer,
        candidate_answer=payload.candidate_answer,
        tfidf_score=tfidf_score,
        semantic_score=sem_score,
        final_score=final_score,
        feedback=feedback,
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)

    return AnswerSubmitResponse(
        answer_id=answer.id,
        tfidf_score=round(answer.tfidf_score, 4),
        semantic_score=round(answer.semantic_score, 4),
        final_score=round(answer.final_score, 4),
        feedback=answer.feedback,
    )


@app.post("/interviews/{interview_id}/complete", response_model=InterviewResultResponse)
def complete_interview(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = db.scalar(
        select(Interview).where(Interview.id == interview_id, Interview.user_id == current_user.id)
    )
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")

    answers = list(interview.answers)
    if not answers:
        raise HTTPException(status_code=400, detail="No answers submitted for this interview")

    # If candidate submits early, auto-mark remaining questions as 0%.
    if len(answers) < REQUIRED_QUESTIONS:
        selected_questions = INTERVIEW_QUESTION_SETS.get(interview_id, [])
        answered_questions = {a.question for a in answers}

        missing_items: list[QuestionItem] = []
        for q in selected_questions:
            if q.question not in answered_questions:
                missing_items.append(q)

        # Fallback when in-memory set is unavailable (e.g., app restart).
        if not missing_items:
            missing_count = REQUIRED_QUESTIONS - len(answers)
            for idx in range(missing_count):
                missing_items.append(
                    QuestionItem(
                        question=f"[AUTO-{idx+1}] Unanswered question",
                        expected_answer="No reference available",
                    )
                )

        for q in missing_items:
            db.add(
                Answer(
                    interview_id=interview.id,
                    question=q.question,
                    expected_answer=q.expected_answer,
                    candidate_answer="Not answered by candidate.",
                    tfidf_score=0.0,
                    semantic_score=0.0,
                    final_score=0.0,
                    feedback="Not attempted. Marked as 0% due to early submission.",
                )
            )
        db.commit()
        db.refresh(interview)
        answers = list(interview.answers)

    overall = sum(a.final_score for a in answers) / len(answers)
    interview.status = "completed"
    interview.overall_score = overall
    interview.selection_decision = "Selected" if (overall * 100) > 65 else "Not Selected"
    interview.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(interview)

    chart_payload = {
        "labels": [f"Q{i+1}" for i in range(len(answers))],
        "datasets": [
            {"label": "TF-IDF", "data": [round(a.tfidf_score * 100, 2) for a in answers]},
            {"label": "Semantic", "data": [round(a.semantic_score * 100, 2) for a in answers]},
            {"label": "Final", "data": [round(a.final_score * 100, 2) for a in answers]},
        ],
    }

    detailed_feedback = [
        {
            "question": a.question,
            "final_score_pct": round(a.final_score * 100, 2),
            "feedback": a.feedback,
        }
        for a in answers
    ]

    return InterviewResultResponse(
        interview_id=interview.id,
        role=interview.role,
        status=interview.status,
        overall_score=round(interview.overall_score * 100, 2),
        total_answers=len(answers),
        selection_decision=interview.selection_decision or "Not Selected",
        chart_payload=chart_payload,
        detailed_feedback=detailed_feedback,
    )


@app.get("/dashboard/me")
def my_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    interviews = list(db.scalars(select(Interview).where(Interview.user_id == current_user.id)))
    completed = [i for i in interviews if i.status == "completed" and i.overall_score is not None]
    avg_score = round((sum(i.overall_score for i in completed) / len(completed)) * 100, 2) if completed else 0.0

    return {
        "user": {"id": current_user.id, "name": current_user.full_name, "email": current_user.email},
        "stats": {
            "total_interviews": len(interviews),
            "completed_interviews": len(completed),
            "average_score_pct": avg_score,
        },
        "interviews": [
            {
                "interview_id": i.id,
                "role": i.role,
                "status": i.status,
                "overall_score_pct": round((i.overall_score or 0) * 100, 2),
                "created_at": i.created_at.isoformat(),
            }
            for i in sorted(interviews, key=lambda x: x.created_at, reverse=True)
        ],
    }


@app.get("/recruiter/me")
def recruiter_profile(current_recruiter: Recruiter = Depends(get_current_recruiter)):
    return {
        "full_name": current_recruiter.full_name,
        "email": current_recruiter.email,
        "company": current_recruiter.company,
        "domain_focus": current_recruiter.domain_focus,
    }


@app.get("/recruiter/candidates", response_model=list[RecruiterCandidateRow])
def recruiter_candidates(
    role: Optional[str] = None,
    min_score: Optional[float] = None,
    _current_recruiter: Recruiter = Depends(get_current_recruiter),
    db: Session = Depends(get_db),
):
    query = (
        select(Interview, User)
        .join(User, Interview.user_id == User.id)
        .where(Interview.status == "completed")
        .order_by(Interview.completed_at.desc(), Interview.created_at.desc())
    )
    if role:
        query = query.where(Interview.role.ilike(f"%{role}%"))
    if min_score is not None:
        min_score_0_1 = max(0.0, min(min_score / 100.0, 1.0))
        query = query.where(Interview.overall_score >= min_score_0_1)

    rows = db.execute(query).all()
    result: list[RecruiterCandidateRow] = []
    for interview, user in rows:
        result.append(
            RecruiterCandidateRow(
                candidate_email=user.email,
                domain=interview.role,
                score_pct=round((interview.overall_score or 0.0) * 100, 2),
                decision=interview.selection_decision or ("Selected" if (interview.overall_score or 0.0) > 0.65 else "Not Selected"),
                completed_at=interview.completed_at.isoformat() if interview.completed_at else None,
            )
        )
    return result


@app.get("/recruiter/candidates/details", response_model=list[RecruiterCandidateDetailRow])
def recruiter_candidate_details(
    role: Optional[str] = None,
    min_score: Optional[float] = None,
    _current_recruiter: Recruiter = Depends(get_current_recruiter),
    db: Session = Depends(get_db),
):
    query = (
        select(Interview, User)
        .join(User, Interview.user_id == User.id)
        .where(Interview.status == "completed")
        .order_by(Interview.completed_at.desc(), Interview.created_at.desc())
    )
    if role:
        query = query.where(Interview.role.ilike(f"%{role}%"))
    if min_score is not None:
        min_score_0_1 = max(0.0, min(min_score / 100.0, 1.0))
        query = query.where(Interview.overall_score >= min_score_0_1)

    rows = db.execute(query).all()
    detailed_rows: list[RecruiterCandidateDetailRow] = []
    for interview, user in rows:
        qa_rows = [
            CandidateQuestionAnswerRow(
                question=a.question,
                answer=a.candidate_answer,
                score_pct=round(a.final_score * 100, 2),
                feedback=a.feedback,
            )
            for a in interview.answers
        ]
        detailed_rows.append(
            RecruiterCandidateDetailRow(
                candidate_email=user.email,
                domain=interview.role,
                score_pct=round((interview.overall_score or 0.0) * 100, 2),
                decision=interview.selection_decision or ("Selected" if (interview.overall_score or 0.0) > 0.65 else "Not Selected"),
                completed_at=interview.completed_at.isoformat() if interview.completed_at else None,
                question_answers=qa_rows,
            )
        )
    return detailed_rows


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tanu:app", host="127.0.0.1", port=8000, reload=True)
