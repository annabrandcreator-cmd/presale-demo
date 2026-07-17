# -*- coding: utf-8 -*-
"""
Демо «ИИ-инженер по продажам» — вымышленная ниша «ПромАэро Системы».
Запуск: pip install -r requirements.txt && python app.py
"""
import json
import os
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory, send_file, abort

import engine
import generate_kp
from cosmetic_api import cosmetic_bp

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("DB_PATH") or os.path.join(APP_DIR, "deals.db")
KP_DIR = os.environ.get("KP_DIR") or os.path.join(APP_DIR, "generated")
STATIC = os.path.join(APP_DIR, "static")
os.makedirs(KP_DIR, exist_ok=True)

DEMO_DELAY_SEC = float(os.environ.get("DEMO_DELAY_SEC", "1.2"))
TEST_MODE = os.environ.get("TEST_MODE", "1") == "1"

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
app.register_blueprint(cosmetic_bp, url_prefix="/cosmetic")
CATALOG = engine.load_catalog()

ALLOWED_ORIGINS = {
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        ",".join(
            [
                "https://annakurbatova.ru",
                "http://localhost:8080",
                "http://127.0.0.1:8080",
                "http://localhost:8765",
                "http://127.0.0.1:8765",
                "http://localhost:8766",
                "http://127.0.0.1:8766",
            ]
        ),
    ).split(",")
    if o.strip()
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    allow = False
    if origin and origin in ALLOWED_ORIGINS:
        allow = True
    elif origin and (
        origin.startswith("http://localhost:")
        or origin.startswith("http://127.0.0.1:")
    ):
        # Local static previews (any port) for redesign iteration
        allow = True
    if allow and origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

STAGES = [
    ("received", "Заявка получена"),
    ("qualifying", "Техническая квалификация"),
    ("calculating", "Подбор оборудования"),
    ("generating_kp", "Формирование КП"),
    ("completed", "КП готово"),
]


def db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS deals (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                answers TEXT NOT NULL DEFAULT '{}',
                spec TEXT,
                kp_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


init_db()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_deal(deal_id):
    with db_conn() as c:
        row = c.execute("SELECT * FROM deals WHERE id=?", (deal_id,)).fetchone()
    return dict(row) if row else None


def update_deal(deal_id, **fields):
    fields["updated_at"] = now_iso()
    keys = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [deal_id]
    with db_conn() as c:
        c.execute(f"UPDATE deals SET {keys} WHERE id=?", vals)


def deal_public(row):
    answers = json.loads(row.get("answers") or "{}")
    spec = json.loads(row["spec"]) if row.get("spec") else None
    stage_idx = next((i for i, (s, _) in enumerate(STAGES) if s == row["stage"]), 0)
    return {
        "id": row["id"],
        "status": row["status"],
        "stage": row["stage"],
        "stage_label": dict(STAGES).get(row["stage"], row["stage"]),
        "stage_index": stage_idx,
        "stages": [{"id": s, "label": l} for s, l in STAGES],
        "answers": answers,
        "spec": spec,
        "summary": engine.deal_summary(answers, spec, CATALOG) if spec else None,
        "kp_url": f"/api/deals/{row['id']}/kp" if row.get("kp_path") else None,
        "error": row.get("error"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "test_mode": TEST_MODE,
    }


def process_deal_async(deal_id):
  """Фоновая обработка: квалификация → расчёт → PDF."""
  try:
    row = get_deal(deal_id)
    if not row:
      return
    answers = json.loads(row["answers"])

    update_deal(deal_id, status="processing", stage="qualifying")
    time.sleep(DEMO_DELAY_SEC)

    missing = engine.next_question(answers, CATALOG)
    if missing:
      update_deal(deal_id, status="awaiting_input", stage="qualifying")
      return

    update_deal(deal_id, stage="calculating")
    time.sleep(DEMO_DELAY_SEC)

    spec = engine.build_specification(answers, CATALOG)

    update_deal(deal_id, stage="generating_kp", spec=json.dumps(spec, ensure_ascii=False))
    time.sleep(DEMO_DELAY_SEC)

    pdf_path = generate_kp.generate_kp_pdf(deal_id, answers, spec, CATALOG, KP_DIR)
    update_deal(deal_id, status="completed", stage="completed", kp_path=pdf_path, error=None)
  except Exception as e:
    traceback.print_exc()
    update_deal(deal_id, status="error", error=str(e))


# ── Routes ──

@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(STATIC, "dashboard.html")


@app.route("/api/storage")
def api_storage():
    """Схема хранения данных — для блока «архитектура» на кейсе."""
    return jsonify({
        "demo": {
            "deals": {"engine": "SQLite", "path": DB, "what": "заявки, ответы, статусы, JSON спецификации"},
            "catalog": {"engine": "JSON-файл", "path": "catalog.json", "what": "номенклатура, правила, цены"},
            "documents": {"engine": "файловая система", "path": KP_DIR, "what": "PDF коммерческих предложений"},
        },
        "production": {
            "deals": "PostgreSQL — сделки, этапы, аудит, связь с CRM",
            "catalog": "PostgreSQL + синхронизация с 1С / ERP (прайс, остатки)",
            "documents": "S3-совместимое хранилище или диск on-premise",
            "llm": "GigaChat / YandexGPT — разбор свободного ТЗ в закрытом контуре",
            "security": "On-premise опция, разграничение доступа, логи действий",
        },
    })


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "autonomous-sales-engineer-demo",
        "brand": CATALOG["brand"]["name"],
        "test_mode": TEST_MODE,
        "deals_db": DB,
    })


@app.route("/api/catalog")
def api_catalog():
    return jsonify({
        "brand": CATALOG["brand"],
        "questions": CATALOG["questions"],
        "object_types": CATALOG["object_types"],
    })


@app.route("/api/deals", methods=["POST"])
def api_create_deal():
    data = request.get_json(silent=True) or {}
    deal_id = engine.new_deal_id()
    answers = {}
    if data.get("free_text"):
        answers.update(engine.parse_free_text(data["free_text"], CATALOG))

    with db_conn() as c:
        c.execute(
            "INSERT INTO deals (id, status, stage, answers, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (deal_id, "new", "received", json.dumps(answers, ensure_ascii=False), now_iso(), now_iso()),
        )

    threading.Thread(target=process_deal_async, args=(deal_id,), daemon=True).start()
    return jsonify({"deal_id": deal_id, "url": f"/dashboard?deal={deal_id}"}), 201


@app.route("/api/deals/<deal_id>")
def api_get_deal(deal_id):
    row = get_deal(deal_id)
    if not row:
        abort(404)
    return jsonify(deal_public(row))


@app.route("/api/deals")
def api_list_deals():
    with db_conn() as c:
        rows = c.execute("SELECT * FROM deals ORDER BY created_at DESC LIMIT 50").fetchall()
    return jsonify([deal_public(dict(r)) for r in rows])


@app.route("/api/deals/<deal_id>/answer", methods=["POST"])
def api_answer(deal_id):
    row = get_deal(deal_id)
    if not row:
        abort(404)
    data = request.get_json(silent=True) or {}
    qid = data.get("question_id")
    value = data.get("value")
    if not qid:
        return jsonify({"error": "question_id required"}), 400

    question = next((q for q in CATALOG["questions"] if q["id"] == qid), None)
    if not question:
        return jsonify({"error": "unknown question"}), 400

    ok, parsed = engine.validate_answer(question, value, CATALOG)
    if not ok:
        return jsonify({"error": parsed}), 400

    answers = json.loads(row["answers"])
    answers[qid] = parsed
    update_deal(deal_id, answers=json.dumps(answers, ensure_ascii=False), status="processing", error=None)

    threading.Thread(target=process_deal_async, args=(deal_id,), daemon=True).start()
    return jsonify({"ok": True, "answers": answers})


@app.route("/api/deals/<deal_id>/kp")
def api_download_kp(deal_id):
    row = get_deal(deal_id)
    if not row or not row.get("spec"):
        abort(404)
    answers = json.loads(row["answers"])
    spec = json.loads(row["spec"])
    # Пересобираем PDF при каждой загрузке — актуальный шаблон и шрифты
    pdf_path = generate_kp.generate_kp_pdf(deal_id, answers, spec, CATALOG, KP_DIR)
    update_deal(deal_id, kp_path=pdf_path)
    resp = send_file(pdf_path, as_attachment=True, download_name=f"KP-{deal_id}.pdf")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"Демо «ИИ-инженер по продажам» → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
