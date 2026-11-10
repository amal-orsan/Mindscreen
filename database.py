"""
database.py — MindScreen SQLite layer
--------------------------------------
Tables:
  patients      — demographic & behavioural inputs
  assessments   — prediction results linked to a patient
  risk_flags    — per-assessment risk tags (normalised)
  recommendations — per-assessment clinical recs (normalised)
"""
 
import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager
 
DB_PATH = os.path.join(os.path.dirname(__file__), "mindscreen.db")
 
 
# ── Connection helper ──────────────────────────────────────────────────────────
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
 
 
# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
 
    -- Demographics
    age                   INTEGER,
    gender                TEXT,
    employment_status     TEXT,
    education_level       TEXT,
    coping_method         TEXT,
 
    -- Clinical scores
    sleep_hours           REAL,
    social_media_hours    REAL,
    depression_score      INTEGER,
    nervous_level         INTEGER,
    overeating_level      INTEGER,
    meals_per_day         INTEGER,
 
    -- Behavioural flags (0/1)
    low_selfesteem        INTEGER DEFAULT 0,
    low_energy            INTEGER DEFAULT 0,
    worsening_depression  INTEGER DEFAULT 0,
    social_while_eating   INTEGER DEFAULT 0,
    mental_health_support INTEGER DEFAULT 0,
    search_online         INTEGER DEFAULT 0,
    suicide_attempts      INTEGER DEFAULT 0,
    self_harm             INTEGER DEFAULT 0
);
 
CREATE TABLE IF NOT EXISTS assessments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    assessed_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
 
    -- Prediction output
    predicted_class   INTEGER NOT NULL,
    predicted_label   TEXT    NOT NULL,
    severity          TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    is_demo_mode      INTEGER DEFAULT 0,
 
    -- Top-3 differential (stored as JSON for simplicity)
    differential_json TEXT,
 
    -- Free-text notes (optional, for clinician use)
    clinician_notes   TEXT
);
 
CREATE TABLE IF NOT EXISTS risk_flags (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    flag_label    TEXT    NOT NULL,
    flag_class    TEXT    NOT NULL   -- tag-red / tag-orange / tag-green / tag-blue / tag-gray
);
 
CREATE TABLE IF NOT EXISTS recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    sort_order    INTEGER NOT NULL,
    rec_text      TEXT    NOT NULL
);
 
-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_assessments_patient  ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_date     ON assessments(assessed_at);
CREATE INDEX IF NOT EXISTS idx_assessments_severity ON assessments(severity);
CREATE INDEX IF NOT EXISTS idx_assessments_class    ON assessments(predicted_class);
"""
 
 
def init_db():
    """Create all tables if they don't exist yet."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
 
 
# ── Write helpers ──────────────────────────────────────────────────────────────
def save_assessment(
    # Patient inputs
    age, gender, employment, education, coping,
    sleep, social_media, depression_score, nervous_level,
    overeating, meals_per_day,
    low_selfesteem, low_energy, worsening_depression, social_while_eating,
    mental_health_support, search_online, suicide_attempts, self_harm,
    # Prediction outputs
    predicted_class, predicted_label, severity, confidence,
    differential, is_demo, flags, recs,
    clinician_notes=""
):
    """
    Insert one patient row + one assessment row + risk_flags + recommendations.
    Returns (patient_id, assessment_id).
    """
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO patients (
                age, gender, employment_status, education_level, coping_method,
                sleep_hours, social_media_hours, depression_score, nervous_level,
                overeating_level, meals_per_day,
                low_selfesteem, low_energy, worsening_depression, social_while_eating,
                mental_health_support, search_online, suicide_attempts, self_harm
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            age, gender, employment, education, coping,
            sleep, social_media, depression_score, nervous_level,
            overeating, meals_per_day,
            int(low_selfesteem), int(low_energy), int(worsening_depression),
            int(social_while_eating), int(mental_health_support),
            int(search_online), int(suicide_attempts), int(self_harm)
        ))
        patient_id = cur.lastrowid
 
        cur2 = conn.execute("""
            INSERT INTO assessments (
                patient_id, predicted_class, predicted_label,
                severity, confidence, is_demo_mode,
                differential_json, clinician_notes
            ) VALUES (?,?,?,?,?,?,?,?)
        """, (
            patient_id, predicted_class, predicted_label,
            severity, confidence, int(is_demo),
            json.dumps(differential), clinician_notes
        ))
        assessment_id = cur2.lastrowid
 
        for label, cls in flags:
            conn.execute(
                "INSERT INTO risk_flags (assessment_id, flag_label, flag_class) VALUES (?,?,?)",
                (assessment_id, label, cls)
            )
 
        for i, rec in enumerate(recs, 1):
            conn.execute(
                "INSERT INTO recommendations (assessment_id, sort_order, rec_text) VALUES (?,?,?)",
                (assessment_id, i, rec)
            )
 
    return patient_id, assessment_id
 
 
# ── Read helpers ───────────────────────────────────────────────────────────────
def get_all_assessments(limit=200):
    """Return recent assessments joined with patient demographics."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                a.id            AS assessment_id,
                a.assessed_at,
                a.predicted_label,
                a.severity,
                a.confidence,
                a.is_demo_mode,
                p.id            AS patient_id,
                p.age,
                p.gender,
                p.employment_status,
                p.education_level,
                p.coping_method,
                p.sleep_hours,
                p.social_media_hours,
                p.depression_score,
                p.nervous_level,
                p.suicide_attempts,
                p.self_harm,
                p.worsening_depression,
                p.mental_health_support,
                p.low_energy,
                p.low_selfesteem
            FROM assessments a
            JOIN patients p ON p.id = a.patient_id
            ORDER BY a.assessed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]
 
 
def get_assessment_detail(assessment_id):
    """Full detail for one assessment including flags and recs."""
    with get_conn() as conn:
        a = conn.execute("""
            SELECT a.*, p.*
            FROM assessments a JOIN patients p ON p.id = a.patient_id
            WHERE a.id = ?
        """, (assessment_id,)).fetchone()
        if not a:
            return None
 
        flags = conn.execute(
            "SELECT flag_label, flag_class FROM risk_flags WHERE assessment_id=? ORDER BY rowid",
            (assessment_id,)
        ).fetchall()
 
        recs = conn.execute(
            "SELECT rec_text FROM recommendations WHERE assessment_id=? ORDER BY sort_order",
            (assessment_id,)
        ).fetchall()
 
    result = dict(a)
    result["flags"] = [(r["flag_label"], r["flag_class"]) for r in flags]
    result["recs"]  = [r["rec_text"] for r in recs]
    result["differential"] = json.loads(result.get("differential_json") or "[]")
    return result
 
 
def get_stats():
    """Aggregate stats for the dashboard tab."""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
        by_severity = conn.execute("""
            SELECT severity, COUNT(*) as cnt
            FROM assessments GROUP BY severity ORDER BY cnt DESC
        """).fetchall()
        by_label = conn.execute("""
            SELECT predicted_label, COUNT(*) as cnt
            FROM assessments GROUP BY predicted_label ORDER BY cnt DESC
        """).fetchall()
        high_risk = conn.execute("""
            SELECT COUNT(*) FROM patients
            WHERE suicide_attempts=1 OR self_harm=1
        """).fetchone()[0]
        avg_phq = conn.execute(
            "SELECT ROUND(AVG(depression_score),1) FROM patients"
        ).fetchone()[0]
        avg_conf = conn.execute(
            "SELECT ROUND(AVG(confidence)*100,1) FROM assessments"
        ).fetchone()[0]
        recent_7d = conn.execute("""
            SELECT COUNT(*) FROM assessments
            WHERE assessed_at >= datetime('now','-7 days')
        """).fetchone()[0]
 
    return {
        "total":       total,
        "high_risk":   high_risk,
        "avg_phq":     avg_phq,
        "avg_conf":    avg_conf,
        "recent_7d":   recent_7d,
        "by_severity": [dict(r) for r in by_severity],
        "by_label":    [dict(r) for r in by_label],
    }
 
 
def delete_assessment(assessment_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM assessments WHERE id=?", (assessment_id,))
 
 
def update_notes(assessment_id, notes):
    with get_conn() as conn:
        conn.execute(
            "UPDATE assessments SET clinician_notes=? WHERE id=?",
            (notes, assessment_id)
        )