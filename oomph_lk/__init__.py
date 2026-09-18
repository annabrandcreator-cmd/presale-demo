# -*- coding: utf-8 -*-
"""
Демо личного кабинета Oomph (аналитика AI-консультанта).
Монтируется в ASE на префиксе /oomph-lk.
"""
from __future__ import annotations

import os
import sqlite3
import threading

from flask import Blueprint, abort, jsonify, request

from . import analytics, engine, seed_demo

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir("/data"):
    DB = os.environ.get("OOMPH_LK_DB_PATH") or "/data/oomph_lk_sessions.db"
else:
    DB = os.environ.get("OOMPH_LK_DB_PATH") or os.path.join(APP_DIR, "oomph_lk_sessions.db")

CATALOG = engine.load_catalog()
SEED_COUNT = int(os.environ.get("OOMPH_LK_SEED_COUNT", "900"))
SEED_DAYS = int(os.environ.get("OOMPH_LK_SEED_DAYS", "90"))

bp = Blueprint("oomph_lk", __name__)

_db_ready = False
_db_lock = threading.Lock()


def db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db():
    global _db_ready
    if _db_ready:
        return
    with _db_lock:
        if _db_ready:
            return
        seed_demo.DB = DB
        with db_conn() as c:
            seed_demo.ensure_schema(c)
            count = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if count < 50:
            seed_demo.seed(count=SEED_COUNT, days_back=SEED_DAYS, reset=(count == 0))
        _db_ready = True


def _period_args():
    return {
        "days": request.args.get("days", 30),
        "date_from": request.args.get("from"),
        "date_to": request.args.get("to"),
    }


@bp.before_app_request
def _ensure_before():
    path = request.path or ""
    if path.startswith("/oomph-lk"):
        ensure_db()


@bp.route("/")
def index():
    return jsonify({
        "service": "oomph-lk-cabinet",
        "brand": CATALOG["brand"]["name"],
        "health": "/oomph-lk/health",
        "site": "https://space.ascendbrand.ru/sites/oomphlk/",
        "api": "/oomph-lk/api/analytics/overview",
    })


@bp.route("/health")
def health():
    ensure_db()
    with db_conn() as c:
        count = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    return jsonify({
        "ok": True,
        "service": "oomph-lk-cabinet",
        "brand": CATALOG["brand"]["name"],
        "sessions_count": count,
    })


@bp.route("/api/catalog")
def api_catalog():
    return jsonify({
        "brand": CATALOG["brand"],
        "promo": CATALOG["promo"],
        "products": CATALOG["products"],
        "questions": CATALOG.get("questions") or [],
    })


@bp.route("/api/analytics/overview")
def api_analytics_overview():
    ensure_db()
    with db_conn() as c:
        return jsonify(analytics.get_overview(c, **_period_args()))


@bp.route("/api/analytics/funnel")
def api_analytics_funnel():
    ensure_db()
    with db_conn() as c:
        return jsonify(analytics.get_funnel(c, **_period_args()))


@bp.route("/api/analytics/timeline")
def api_analytics_timeline():
    ensure_db()
    with db_conn() as c:
        return jsonify(analytics.get_timeline(c, **_period_args()))


@bp.route("/api/analytics/distributions")
def api_analytics_distributions():
    ensure_db()
    with db_conn() as c:
        return jsonify(analytics.get_distributions(c, **_period_args()))


@bp.route("/api/analytics/products")
def api_analytics_products():
    ensure_db()
    with db_conn() as c:
        return jsonify(analytics.get_products(c, **_period_args(), limit=request.args.get("limit", 8)))


@bp.route("/api/analytics/basket")
def api_analytics_basket():
    ensure_db()
    with db_conn() as c:
        return jsonify(analytics.get_basket(c, **_period_args()))


PHOTO_BASE = "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/"
_PHOTO_ALIASES = (
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/",
    "https://space.ascendbrand.ru/sites/oomphlk/assets/faces/",
    "/static/assets/faces/",
    "assets/faces/",
)


def _fix_photo_url(url):
    if not url:
        return url
    s = str(url)
    for prefix in _PHOTO_ALIASES:
        if s.startswith(prefix):
            return PHOTO_BASE + s[len(prefix):].lstrip("/")
    if "/faces/" in s:
        return PHOTO_BASE + s.rsplit("/", 1)[-1]
    return s


def _rewrite_photos(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("photo_url",) and isinstance(v, str):
                out[k] = _fix_photo_url(v)
            elif k == "skin_scan" and isinstance(v, dict):
                scan = dict(v)
                if isinstance(scan.get("photo_url"), str):
                    scan["photo_url"] = _fix_photo_url(scan["photo_url"])
                out[k] = scan
            else:
                out[k] = _rewrite_photos(v)
        return out
    if isinstance(obj, list):
        return [_rewrite_photos(x) for x in obj]
    return obj


@bp.route("/api/clients")
def api_clients():
    ensure_db()
    with db_conn() as c:
        data = analytics.get_clients(c, **_period_args(), limit=request.args.get("limit", 50))
    return jsonify(_rewrite_photos(data))


@bp.route("/api/clients/<session_id>")
def api_client_detail(session_id):
    ensure_db()
    with db_conn() as c:
        detail = analytics.get_client_detail(c, session_id, CATALOG)
    if not detail:
        abort(404)
    return jsonify(_rewrite_photos(detail))
