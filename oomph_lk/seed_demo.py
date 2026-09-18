# -*- coding: utf-8 -*-
"""Генерация демо-данных для кабинета бренда Oomph."""
import argparse
import json
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from . import engine

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("DB_PATH") or os.path.join(APP_DIR, "sessions.db")
CATALOG = engine.load_catalog()

NAMES = [
    "Алина", "Мария", "Екатерина", "Ольга", "Анна", "Дарья", "Полина", "Виктория",
    "София", "Ксения", "Наталья", "Ирина", "Елена", "Юлия", "Вера", "Кристина",
    "Александра", "Милана", "Арина", "Диана", "Людмила", "Светлана", "Татьяна",
    "Валерия", "Ева", "Злата", "Маргарита", "Надежда", "Оксана", "Регина",
]

CONCERNS = list(engine.CONCERN_LABELS.keys())
SKIN_TYPES = list(engine.SKIN_LABELS.keys())
AGE_RANGES = list(engine.AGE_LABELS.keys())
ROUTINE_LEVELS = list(engine.ROUTINE_LABELS.keys())
BUDGETS = list(engine.BUDGET_LABELS.keys())
SOURCES = ["web", "web", "web", "web", "web", "web", "widget", "widget", "telegram"]

FACE_PHOTOS = [
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/skin-face-1.png",
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/skin-realistic.png",
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/persona-acne.png",
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/persona-couperose.png",
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/persona-mature.png",
    "https://www.ascendbrand.ru/space/sites/oomphlk/assets/faces/persona-pores.png",
]

SKIN_HEADLINES = {
    "dry": "Сухая кожа",
    "oily": "Жирная кожа",
    "combination": "Смешанная кожа",
    "normal": "Нормальная кожа",
    "sensitive": "Чувствительная кожа",
}

METRIC_POOL = {
    "hydration": "Увлажнённость",
    "pores": "Поры",
    "redness": "Покраснения",
    "dullness": "Тусклость",
    "fine_lines": "Мелкие морщины",
    "radiance": "Сияние",
    "barrier": "Кожный барьер",
    "pigmentation": "Пигментация",
}

CONCERN_METRICS = {
    "dryness": ["hydration", "barrier", "radiance"],
    "acne": ["pores", "redness", "barrier"],
    "aging": ["fine_lines", "pigmentation", "radiance"],
    "sensitivity": ["redness", "barrier", "hydration"],
    "pores": ["pores", "hydration", "dullness"],
    "dullness": ["dullness", "radiance", "pigmentation"],
    "barrier": ["barrier", "hydration", "redness"],
}


def db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            answers TEXT NOT NULL DEFAULT '{}',
            routine TEXT,
            error TEXT,
            source TEXT DEFAULT 'web',
            skin_scan TEXT,
            outcome TEXT,
            duration_sec INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    for col, ddl in [
        ("source", "ALTER TABLE sessions ADD COLUMN source TEXT DEFAULT 'web'"),
        ("skin_scan", "ALTER TABLE sessions ADD COLUMN skin_scan TEXT"),
        ("outcome", "ALTER TABLE sessions ADD COLUMN outcome TEXT"),
        ("duration_sec", "ALTER TABLE sessions ADD COLUMN duration_sec INTEGER"),
    ]:
        if col not in cols:
            conn.execute(ddl)


def _rand_dt(days_back):
    day_offset = int(random.triangular(0, days_back, days_back * 0.2))
    base = datetime.now(timezone.utc) - timedelta(days=day_offset)
    if base.weekday() >= 5:
        base += timedelta(hours=random.randint(10, 21), minutes=random.randint(0, 59))
    else:
        base += timedelta(hours=random.randint(8, 22), minutes=random.randint(0, 59))
    return base


def build_skin_scan(concern, skin_type):
    metrics_ids = CONCERN_METRICS.get(concern, ["hydration", "pores", "dullness"])
    metrics = []
    for mid in metrics_ids:
        if mid in ("hydration", "radiance", "barrier"):
            score = random.randint(25, 55)
        else:
            score = random.randint(55, 82)
        metrics.append({"id": mid, "label": METRIC_POOL[mid], "score": score})
    return {
        "from_photo": True,
        "headline": SKIN_HEADLINES.get(skin_type, "Смешанная кожа"),
        "priority_concern": concern,
        "photo_url": random.choice(FACE_PHOTOS),
        "metrics": metrics,
    }


def build_outcome(completed, purchased=None, routine=None):
    if not completed:
        return None
    if purchased is None:
        purchased = random.random() < 0.325
    cart = purchased or random.random() < 0.55
    promo = cart or random.random() < 0.7
    outcome = {
        "cart_opened": cart,
        "promo_copied": promo,
        "purchased": purchased,
    }
    if purchased and routine and routine.get("steps"):
        priced = [s for s in routine["steps"] if (s.get("price_rub") or 0) > 0]
        if priced:
            # Часто берут часть набора, иногда весь
            k = len(priced) if random.random() < 0.35 else random.randint(1, len(priced))
            bought = sorted(random.sample(priced, k), key=lambda s: s.get("step") or 0)
            outcome["purchased_steps"] = [
                {
                    "product_id": s.get("product_id"),
                    "name": s["name"],
                    "category": s.get("category"),
                    "price_rub": s["price_rub"],
                }
                for s in bought
            ]
            outcome["purchased_total_rub"] = sum(s["price_rub"] for s in bought)
    return outcome


def build_session(days_back=90):
    concern = random.choice(CONCERNS)
    skin = random.choice(SKIN_TYPES)
    answers = {
        "primary_concern": concern,
        "skin_type": skin,
        "age_range": random.choice(AGE_RANGES),
        "routine_level": random.choice(ROUTINE_LEVELS),
        "budget": random.choice(BUDGETS),
        "contact_name": random.choice(NAMES),
    }

    roll = random.random()
    if roll < 0.88:
        status = "completed"
        stage = "completed"
        will_buy = random.random() < 0.325
        # У покупок — более полные наборы, чтобы сумма выглядела убедительнее
        if will_buy:
            answers["budget"] = random.choices(
                ["economy", "mid", "premium"], weights=[12, 48, 40]
            )[0]
            answers["routine_level"] = random.choices(
                ["minimal", "basic", "advanced"], weights=[8, 42, 50]
            )[0]
        routine = engine.build_routine(answers, CATALOG)
        outcome = build_outcome(True, purchased=will_buy, routine=routine)
        duration = random.randint(95, 210)
        error = None
    elif roll < 0.96:
        status = "awaiting_input"
        stage = "qualifying"
        answers.pop("contact_name", None)
        routine = None
        outcome = None
        duration = random.randint(40, 90)
        error = None
    else:
        status = "error"
        stage = "matching"
        routine = None
        outcome = None
        duration = random.randint(60, 120)
        error = "Демо: временная ошибка подбора"

    # Диагностика по фото — обязательна для завершённых консультаций (и покупок).
    # Без фото оставляем только тех, кто бросил на вопросах до загрузки селфи.
    if status == "completed":
        has_photo = True
    elif status == "awaiting_input":
        has_photo = random.random() < 0.55
    else:
        has_photo = False
    skin_scan = build_skin_scan(concern, skin) if has_photo else None
    source = random.choice(SOURCES)

    created = _rand_dt(days_back)
    updated = created + timedelta(seconds=duration or 60)

    return {
        "id": engine.new_session_id(),
        "status": status,
        "stage": stage,
        "answers": json.dumps(answers, ensure_ascii=False),
        "routine": json.dumps(routine, ensure_ascii=False) if routine else None,
        "error": error,
        "source": source,
        "skin_scan": json.dumps(skin_scan, ensure_ascii=False) if skin_scan else None,
        "outcome": json.dumps(outcome, ensure_ascii=False) if outcome else None,
        "duration_sec": duration,
        "created_at": created.isoformat(),
        "updated_at": updated.isoformat(),
    }


def seed(count=7200, reset=False, days_back=90, batch_size=250):
    with db_conn() as conn:
        ensure_schema(conn)
        if reset:
            conn.execute("DELETE FROM sessions")
        existing = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if existing >= count and not reset:
            print(f"Уже {existing} сессий — пропуск (используйте --reset)")
            return existing

        to_create = count if reset else max(0, count - existing)
        created = 0
        while created < to_create:
            chunk = min(batch_size, to_create - created)
            rows = [build_session(days_back) for _ in range(chunk)]
            conn.executemany(
                """
                INSERT INTO sessions
                (id, status, stage, answers, routine, error, source, skin_scan, outcome,
                 duration_sec, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r["id"], r["status"], r["stage"], r["answers"], r["routine"], r["error"],
                        r["source"], r["skin_scan"], r["outcome"], r["duration_sec"],
                        r["created_at"], r["updated_at"],
                    )
                    for r in rows
                ],
            )
            created += chunk
            print(f"  … {created}/{to_create}")
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        print(f"Seed готов: +{to_create} сессий, всего {total}")
        return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Oomph brand cabinet demo sessions")
    parser.add_argument("--count", type=int, default=7200)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    seed(count=args.count, reset=args.reset, days_back=args.days)
