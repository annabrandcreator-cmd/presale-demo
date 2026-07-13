# -*- coding: utf-8 -*-
"""API ИИ-консультанта по уходу (B2C, Auréa Skin)."""
import json
import os
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request, abort

import cosmetic_engine as engine

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("COSMETIC_DB_PATH") or os.path.join(APP_DIR, "cosmetic_sessions.db")
DEMO_DELAY_SEC = float(os.environ.get("COSMETIC_DELAY_SEC", "0.9"))
TEST_MODE = os.environ.get("TEST_MODE", "1") == "1"

cosmetic_bp = Blueprint("cosmetic", __name__)
CATALOG = engine.load_catalog()

STAGES = [
    ("received", "Старт"),
    ("qualifying", "Диагностика"),
    ("matching", "Подбор ухода"),
    ("offer", "Предложение"),
    ("completed", "Готово"),
]


def db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                answers TEXT NOT NULL DEFAULT '{}',
                routine TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


init_db()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_session(session_id):
    with db_conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def update_session(session_id, **fields):
    fields["updated_at"] = now_iso()
    keys = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [session_id]
    with db_conn() as c:
        c.execute(f"UPDATE sessions SET {keys} WHERE id=?", vals)


def session_public(row):
    answers = json.loads(row.get("answers") or "{}")
    routine = json.loads(row["routine"]) if row.get("routine") else None
    stage_idx = next((i for i, (s, _) in enumerate(STAGES) if s == row["stage"]), 0)
    return {
        "id": row["id"],
        "status": row["status"],
        "stage": row["stage"],
        "stage_label": dict(STAGES).get(row["stage"], row["stage"]),
        "stage_index": stage_idx,
        "stages": [{"id": s, "label": l} for s, l in STAGES],
        "answers": answers,
        "routine": routine,
        "summary": engine.session_summary(answers, routine, CATALOG) if routine else None,
        "error": row.get("error"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "test_mode": TEST_MODE,
    }


def process_session_async(session_id):
    try:
        row = get_session(session_id)
        if not row:
            return
        answers = json.loads(row["answers"])

        update_session(session_id, status="processing", stage="qualifying")
        time.sleep(DEMO_DELAY_SEC)

        if engine.next_question(answers, CATALOG):
            update_session(session_id, status="awaiting_input", stage="qualifying")
            return

        update_session(session_id, stage="matching")
        time.sleep(DEMO_DELAY_SEC)

        routine = engine.build_routine(answers, CATALOG)
        update_session(session_id, stage="offer", routine=json.dumps(routine, ensure_ascii=False))
        time.sleep(DEMO_DELAY_SEC)

        update_session(session_id, status="completed", stage="completed", error=None)
    except Exception as e:
        traceback.print_exc()
        update_session(session_id, status="error", error=str(e))


@cosmetic_bp.route("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "cosmetic-consultant-demo",
        "brand": CATALOG["brand"]["name"],
        "test_mode": TEST_MODE,
    })


@cosmetic_bp.route("/api/catalog")
def api_catalog():
    return jsonify({
        "brand": CATALOG["brand"],
        "promo": CATALOG["promo"],
        "questions": CATALOG["questions"],
    })


@cosmetic_bp.route("/api/sessions", methods=["POST", "OPTIONS"])
def api_create_session():
    if request.method == "OPTIONS":
        return "", 204
    session_id = engine.new_session_id()
    with db_conn() as c:
        c.execute(
            "INSERT INTO sessions (id, status, stage, answers, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (session_id, "new", "received", "{}", now_iso(), now_iso()),
        )
    threading.Thread(target=process_session_async, args=(session_id,), daemon=True).start()
    return jsonify({"session_id": session_id}), 201


@cosmetic_bp.route("/api/sessions/<session_id>")
def api_get_session(session_id):
    row = get_session(session_id)
    if not row:
        abort(404)
    return jsonify(session_public(row))


@cosmetic_bp.route("/api/sessions/<session_id>/answer", methods=["POST", "OPTIONS"])
def api_answer(session_id):
    if request.method == "OPTIONS":
        return "", 204
    row = get_session(session_id)
    if not row:
        abort(404)
    data = request.get_json(silent=True) or {}
    qid = data.get("question_id")
    value = data.get("value")
    if not qid:
        return jsonify({"error": "question_id required"}), 400

    question = next((q for q in CATALOG["questions"] if q["id"] == qid), None)
    if not question:
        return jsonify({"error": "unknown question"}), 404

    ok, parsed = engine.validate_answer(question, value, CATALOG)
    if not ok:
        return jsonify({"error": parsed}), 400

    answers = json.loads(row["answers"])
    answers[qid] = parsed
    update_session(session_id, answers=json.dumps(answers, ensure_ascii=False), status="processing", error=None)

    threading.Thread(target=process_session_async, args=(session_id,), daemon=True).start()
    return jsonify({"ok": True, "answers": answers})
