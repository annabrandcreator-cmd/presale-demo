# -*- coding: utf-8 -*-
"""API ИИ-консультанта по уходу (B2C) — персонализация + фото-анализ."""
import base64
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
DEMO_DELAY_SEC = float(os.environ.get("COSMETIC_DELAY_SEC", "0.85"))
TEST_MODE = os.environ.get("TEST_MODE", "1") == "1"

cosmetic_bp = Blueprint("cosmetic", __name__)
CATALOG = engine.load_catalog()

STAGES = [
    ("received", "Старт"),
    ("photo", "Фото-анализ"),
    ("qualifying", "Диагностика"),
    ("profiling", "Профиль кожи"),
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
                skin_scan TEXT,
                routine TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cols = {r[1] for r in c.execute("PRAGMA table_info(sessions)").fetchall()}
        if "skin_scan" not in cols:
            c.execute("ALTER TABLE sessions ADD COLUMN skin_scan TEXT")


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
    skin_scan = json.loads(row["skin_scan"]) if row.get("skin_scan") else None
    stage_idx = next((i for i, (s, _) in enumerate(STAGES) if s == row["stage"]), 0)
    next_q = engine.next_question(answers, CATALOG) if row["status"] in ("awaiting_input", "new") else None
    profile = None
    if routine and routine.get("profile"):
        profile = routine["profile"]
    elif answers.get("primary_concern") and answers.get("skin_type"):
        profile = engine.build_profile(answers, skin_scan)
    return {
        "id": row["id"],
        "status": row["status"],
        "stage": row["stage"],
        "stage_label": dict(STAGES).get(row["stage"], row["stage"]),
        "stage_index": stage_idx,
        "stages": [{"id": s, "label": l} for s, l in STAGES],
        "answers": answers,
        "skin_scan": skin_scan,
        "routine": routine,
        "profile": profile,
        "next_question": next_q,
        "awaiting_photo": (
            answers.get("photo_choice") == "upload"
            and not skin_scan
            and row["status"] == "awaiting_input"
        ),
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
        skin_scan = json.loads(row["skin_scan"]) if row.get("skin_scan") else None

        update_session(session_id, status="processing", stage="qualifying")
        time.sleep(DEMO_DELAY_SEC * 0.7)

        if answers.get("photo_choice") == "upload" and not skin_scan:
            update_session(session_id, status="awaiting_input", stage="photo")
            return

        if engine.next_question(answers, CATALOG):
            update_session(session_id, status="awaiting_input", stage="qualifying")
            return

        update_session(session_id, stage="profiling")
        time.sleep(DEMO_DELAY_SEC)

        update_session(session_id, stage="matching")
        time.sleep(DEMO_DELAY_SEC)

        routine = engine.build_routine(answers, CATALOG, skin_scan)
        update_session(session_id, stage="offer", routine=json.dumps(routine, ensure_ascii=False))
        time.sleep(DEMO_DELAY_SEC * 0.8)

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
        "features": ["adaptive_questions", "skin_profile", "photo_scan", "why_explanations"],
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

    update_session(
        session_id,
        answers=json.dumps(answers, ensure_ascii=False),
        status="processing",
        error=None,
    )
    threading.Thread(target=process_session_async, args=(session_id,), daemon=True).start()
    return jsonify({"ok": True, "answers": answers})


@cosmetic_bp.route("/api/sessions/<session_id>/photo", methods=["POST", "OPTIONS"])
def api_photo(session_id):
    if request.method == "OPTIONS":
        return "", 204
    row = get_session(session_id)
    if not row:
        abort(404)

    image_bytes = None
    filename = "photo.jpg"
    consent = False

    if request.content_type and "multipart/form-data" in request.content_type:
        f = request.files.get("photo") or request.files.get("file")
        if not f:
            return jsonify({"error": "photo required"}), 400
        image_bytes = f.read()
        filename = f.filename or filename
        consent_raw = (request.form.get("consent") or "").strip().lower()
        consent = consent_raw in ("1", "true", "yes", "on")
    else:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image_base64") or data.get("photo")
        if not b64:
            return jsonify({"error": "image_base64 required"}), 400
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64)
        except Exception:
            return jsonify({"error": "invalid base64"}), 400
        filename = data.get("filename") or filename
        consent = bool(data.get("consent"))

    if not consent:
        return jsonify({
            "error": "Нужно согласие: фото обрабатывается для демо-подбора и не сохраняется."
        }), 400

    try:
        scan = engine.analyze_skin_photo(image_bytes, filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"analyze failed: {e}"}), 500

    # Privacy: never persist the image — only anonymous metrics for matching.
    scan_store = {k: v for k, v in scan.items() if k != "preview"}
    scan_store.pop("preview", None)
    # Drop any accidental binary-sized fields
    image_bytes = None

    answers = json.loads(row["answers"])
    answers["photo_choice"] = "upload"
    answers["photo_consent"] = True
    answers["photo_not_stored"] = True
    if not answers.get("primary_concern"):
        answers["primary_concern"] = scan["suggested_concern"]
    if not answers.get("skin_type"):
        answers["skin_type"] = scan["suggested_skin_type"]

    update_session(
        session_id,
        answers=json.dumps(answers, ensure_ascii=False),
        skin_scan=json.dumps(scan_store, ensure_ascii=False),
        status="processing",
        stage="photo",
        error=None,
    )
    threading.Thread(target=process_session_async, args=(session_id,), daemon=True).start()
    return jsonify({"ok": True, "skin_scan": scan_store, "answers": answers})
