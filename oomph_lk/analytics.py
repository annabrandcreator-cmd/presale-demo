# -*- coding: utf-8 -*-
"""Агрегации для B2B-кабинета бренда."""
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from . import engine

STAGE_ORDER = ["received", "qualifying", "matching", "offer", "completed"]
STAGE_LABELS = {
    "received": "Старт",
    "qualifying": "Диагностика",
    "matching": "Подбор ухода",
    "offer": "Предложение",
    "completed": "Покупка",
}


def _parse_days(days):
    try:
        return max(1, min(int(days), 365))
    except (TypeError, ValueError):
        return 30


def parse_period(days=None, date_from=None, date_to=None):
    if date_from and date_to:
        try:
            start = datetime.strptime(str(date_from)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = datetime.strptime(str(date_to)[:10], "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            if end < start:
                start, end = end.replace(hour=0, minute=0, second=0), start.replace(
                    hour=23, minute=59, second=59
                )
            span = (end.date() - start.date()).days + 1
            span = max(1, min(span, 365))
            return {
                "from_iso": start.isoformat(),
                "to_iso": end.isoformat(),
                "days": span,
                "date_from": start.date().isoformat(),
                "date_to": end.date().isoformat(),
                "custom": True,
            }
        except ValueError:
            pass

    d = _parse_days(days)
    since = datetime.now(timezone.utc) - timedelta(days=d)
    today = datetime.now(timezone.utc).date()
    return {
        "from_iso": since.isoformat(),
        "to_iso": None,
        "days": d,
        "date_from": since.date().isoformat(),
        "date_to": today.isoformat(),
        "custom": False,
    }


def _period_meta(period):
    return {
        "period_days": period["days"],
        "date_from": period["date_from"],
        "date_to": period["date_to"],
        "custom": period.get("custom", False),
    }


def _load_sessions(conn, period):
    if isinstance(period, int):
        period = parse_period(days=period)
    if period.get("to_iso"):
        rows = conn.execute(
            "SELECT * FROM sessions WHERE created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
            (period["from_iso"], period["to_iso"]),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE created_at >= ? ORDER BY created_at DESC",
            (period["from_iso"],),
        ).fetchall()
    return [dict(r) for r in rows]


def _parse_json(raw, default=None):
    if not raw:
        return default if default is not None else {}
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default if default is not None else {}


def _pct(part, whole):
    if not whole:
        return 0.0
    return round(part / whole * 100, 1)


def _avg(values):
    nums = [v for v in values if v is not None]
    if not nums:
        return 0
    return round(sum(nums) / len(nums), 1)


def _label_map(field, labels):
    return {key: {"id": key, "label": labels.get(key, key), "count": 0} for key in labels}


def get_overview(conn, days=30, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    rows = _load_sessions(conn, period)
    completed = [r for r in rows if r["status"] == "completed"]
    with_routine = [r for r in completed if r.get("routine")]
    purchased = [
        r for r in completed
        if _parse_json(r.get("outcome")).get("purchased")
    ]
    purchase_totals = []
    for row in purchased:
        outcome = _parse_json(row.get("outcome"))
        if outcome.get("purchased_total_rub"):
            purchase_totals.append(outcome["purchased_total_rub"])
            continue
        routine = _sanitize_routine(_parse_json(row.get("routine")))
        if routine and routine.get("total_rub"):
            purchase_totals.append(routine["total_rub"])

    with_photo = [r for r in rows if r.get("skin_scan")]
    durations = [r.get("duration_sec") for r in completed if r.get("duration_sec")]
    totals = []
    positions = []
    for row in with_routine:
        routine = _sanitize_routine(_parse_json(row["routine"]))
        if not routine:
            continue
        totals.append(routine.get("total_rub"))
        positions.append(routine.get("positions"))

    return {
        **_period_meta(period),
        "total_sessions": len(rows),
        "completed_sessions": len(completed),
        "completion_rate": _pct(len(completed), len(rows)),
        "conversion_rate": _pct(len(purchased), len(completed)),
        "purchased_count": len(purchased),
        "purchase_revenue_rub": round(sum(purchase_totals)),
        "purchase_revenue_formatted": engine.format_money(sum(purchase_totals)),
        "avg_purchase_rub": round(_avg(purchase_totals)),
        "avg_purchase_formatted": engine.format_money(_avg(purchase_totals)) if purchase_totals else "—",
        "avg_basket_rub": round(_avg(totals)),
        "avg_positions": round(_avg(positions), 1),
        "photo_rate": _pct(len(with_photo), len(rows)),
        "photo_count": len(with_photo),
        "avg_duration_sec": round(_avg(durations)),
        "avg_duration_label": _duration_label(round(_avg(durations))),
    }


def _duration_label(sec):
    if not sec:
        return "—"
    if sec < 60:
        return f"{sec} сек"
    mins = sec // 60
    rest = sec % 60
    if rest:
        return f"{mins} мин {rest} сек"
    return f"{mins} мин"


def get_funnel(conn, days=30, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    rows = _load_sessions(conn, period)
    total = len(rows)
    stage_counts = Counter()
    for row in rows:
        stage = row.get("stage") or "received"
        idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
        for s in STAGE_ORDER[: idx + 1]:
            stage_counts[s] += 1

    purchased = sum(
        1 for r in rows
        if r["status"] == "completed" and _parse_json(r.get("outcome")).get("purchased")
    )
    cart_opened = sum(
        1 for r in rows
        if _parse_json(r.get("outcome")).get("cart_opened")
    )
    promo_copied = sum(
        1 for r in rows
        if _parse_json(r.get("outcome")).get("promo_copied")
    )

    steps = []
    for stage in STAGE_ORDER:
        count = stage_counts.get(stage, 0)
        steps.append({
            "id": stage,
            "label": STAGE_LABELS[stage],
            "count": count,
            "rate": _pct(count, total),
        })

    if purchased:
        steps[-1]["count"] = purchased
        steps[-1]["rate"] = _pct(purchased, total)

    return {
        **_period_meta(period),
        "total": total,
        "steps": steps,
        "cart_opened": cart_opened,
        "promo_copied": promo_copied,
        "purchased": purchased,
    }


def get_timeline(conn, days=30, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    rows = _load_sessions(conn, period)
    by_day = defaultdict(lambda: {"total": 0, "completed": 0, "purchased": 0})
    for row in rows:
        day = row["created_at"][:10]
        by_day[day]["total"] += 1
        if row["status"] == "completed":
            by_day[day]["completed"] += 1
        if _parse_json(row.get("outcome")).get("purchased"):
            by_day[day]["purchased"] += 1

    start = datetime.strptime(period["date_from"], "%Y-%m-%d").date()
    end = datetime.strptime(period["date_to"], "%Y-%m-%d").date()
    span = (end - start).days + 1
    labels = []
    totals = []
    completed = []
    purchased = []
    for i in range(span):
        day = (start + timedelta(days=i)).isoformat()
        bucket = by_day.get(day, {"total": 0, "completed": 0, "purchased": 0})
        labels.append(day[5:])
        totals.append(bucket["total"])
        completed.append(bucket["completed"])
        purchased.append(bucket["purchased"])

    return {
        **_period_meta(period),
        "labels": labels,
        "datasets": [
            {"id": "total", "label": "Консультации", "data": totals},
            {"id": "completed", "label": "Завершённые сессии", "data": completed},
            {"id": "purchased", "label": "Покупки", "data": purchased},
        ],
    }


def get_distributions(conn, days=30, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    rows = _load_sessions(conn, period)
    concerns = _label_map("primary_concern", engine.CONCERN_LABELS)
    skin_types = _label_map("skin_type", engine.SKIN_LABELS)
    budgets = _label_map("budget", engine.BUDGET_LABELS)
    routines = _label_map("routine_level", engine.ROUTINE_LABELS)
    sources = _label_map("source", engine.SOURCE_LABELS)

    for row in rows:
        answers = _parse_json(row.get("answers"))
        source = row.get("source") or "web"
        if answers.get("primary_concern") in concerns:
            concerns[answers["primary_concern"]]["count"] += 1
        if answers.get("skin_type") in skin_types:
            skin_types[answers["skin_type"]]["count"] += 1
        if answers.get("budget") in budgets:
            budgets[answers["budget"]]["count"] += 1
        if answers.get("routine_level") in routines:
            routines[answers["routine_level"]]["count"] += 1
        if source in sources:
            sources[source]["count"] += 1

    def _sorted_items(bucket):
        return sorted(bucket.values(), key=lambda x: (-x["count"], x["label"]))

    return {
        **_period_meta(period),
        "concerns": _sorted_items(concerns),
        "skin_types": _sorted_items(skin_types),
        "budgets": _sorted_items(budgets),
        "routine_levels": _sorted_items(routines),
        "sources": _sorted_items(sources),
    }


def get_products(conn, days=30, limit=8, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    limit = max(1, min(int(limit or 8), 20))
    rows = _load_sessions(conn, period)
    counter = Counter()
    for row in rows:
        if row["status"] != "completed":
            continue
        routine = _parse_json(row.get("routine"))
        for product in routine.get("products") or routine.get("steps") or []:
            pid = product.get("product_id") or product.get("id")
            name = product.get("name")
            if pid:
                counter[(pid, name)] += 1

    total_refs = sum(counter.values()) or 1
    items = []
    for (pid, name), count in counter.most_common(limit):
        items.append({
            "id": pid,
            "name": name or pid,
            "count": count,
            "share": _pct(count, total_refs),
        })

    return {**_period_meta(period), "total_recommendations": total_refs, "items": items}


def get_basket(conn, days=30, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    rows = _load_sessions(conn, period)
    completed = [r for r in rows if r["status"] == "completed" and r.get("routine")]
    totals = []
    positions = []
    weekly = defaultdict(lambda: {"totals": [], "positions": []})

    for row in completed:
        routine = _parse_json(row["routine"])
        total = routine.get("total_rub", 0)
        pos = routine.get("positions", 0)
        totals.append(total)
        positions.append(pos)
        week = row["created_at"][:10]
        week_start = datetime.fromisoformat(week).date()
        week_key = (week_start - timedelta(days=week_start.weekday())).isoformat()
        weekly[week_key]["totals"].append(total)
        weekly[week_key]["positions"].append(pos)

    end = datetime.strptime(period["date_to"], "%Y-%m-%d").date()
    week_labels = []
    week_avg_totals = []
    week_avg_positions = []
    week_count = min(5, max(1, period["days"] // 7 + 1))
    for i in range(week_count):
        offset = (week_count - 1 - i) * 7
        anchor = end - timedelta(days=offset)
        week_start = anchor - timedelta(days=anchor.weekday())
        key = week_start.isoformat()
        bucket = weekly.get(key, {"totals": [], "positions": []})
        week_labels.append(f"{week_start.day}.{week_start.month:02d}")
        week_avg_totals.append(round(_avg(bucket["totals"])))
        week_avg_positions.append(round(_avg(bucket["positions"]), 1))

    catalog_avg = engine.catalog_avg_price(engine.load_catalog())

    return {
        **_period_meta(period),
        "avg_basket_rub": round(_avg(totals)),
        "avg_positions": round(_avg(positions), 1),
        "catalog_single_avg_rub": catalog_avg,
        "upsell_multiplier": round(_avg(totals) / catalog_avg, 1) if catalog_avg else 0,
        "weekly_labels": week_labels,
        "weekly_avg_totals": week_avg_totals,
        "weekly_avg_positions": week_avg_positions,
    }


def get_clients(conn, days=30, limit=50, date_from=None, date_to=None):
    period = parse_period(days, date_from, date_to)
    limit = max(1, min(int(limit or 50), 200))
    rows = _load_sessions(conn, period)[:limit]
    items = [engine.client_list_item(r) for r in rows]
    # Сначала с именами, внутри группы — свежие сверху
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    items.sort(key=lambda x: 0 if (x.get("contact_name") and x.get("contact_name") != "Без имени") else 1)
    return {**_period_meta(period), "items": items}


def _sanitize_routine(routine):
    if not routine:
        return None
    steps = [
        s for s in (routine.get("steps") or [])
        if isinstance(s.get("price_rub"), (int, float)) and s.get("price_rub") > 0
    ]
    products = [
        p for p in (routine.get("products") or [])
        if isinstance(p.get("price_rub"), (int, float)) and p.get("price_rub") > 0
    ]
    if not steps and not products:
        return None
    for i, step in enumerate(steps, 1):
        step["step"] = i
    subtotal = sum(s["price_rub"] for s in steps) if steps else sum(p["price_rub"] for p in products)
    discount_pct = routine.get("discount_percent") or 5
    discount = round(subtotal * discount_pct / 100)
    total = subtotal - discount
    return {
        **routine,
        "steps": steps,
        "products": products or [
            {
                "id": s.get("product_id"),
                "name": s["name"],
                "category": s.get("category"),
                "price_rub": s["price_rub"],
                "benefit": s.get("benefit") or "",
                "image": s.get("image"),
                "url": s.get("url"),
            }
            for s in steps
        ],
        "subtotal_rub": subtotal,
        "discount_rub": discount,
        "total_rub": total,
        "positions": len(steps),
        "total_formatted": engine.format_money(total),
        "subtotal_formatted": engine.format_money(subtotal),
    }


def get_client_detail(conn, session_id, catalog):
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        return None
    row = dict(row)
    answers = _parse_json(row.get("answers"))
    routine = _sanitize_routine(_parse_json(row.get("routine")))
    skin_scan = _parse_json(row.get("skin_scan"))
    outcome = _parse_json(row.get("outcome")) or {
        "cart_opened": False, "promo_copied": False, "purchased": False,
    }

    stages = [
        {"id": s, "label": l, "done": False, "current": False}
        for s, l in [
            ("received", "Старт"),
            ("qualifying", "Диагностика"),
            ("matching", "Подбор ухода"),
            ("offer", "Предложение"),
            ("completed", "Готово"),
        ]
    ]
    stage_idx = next((i for i, s in enumerate(stages) if s["id"] == row["stage"]), 0)
    for i, stage in enumerate(stages):
        stage["done"] = i < stage_idx or row["status"] == "completed"
        stage["current"] = i == stage_idx and row["status"] != "completed"

    answer_labels = []
    for q in catalog.get("questions", []):
        val = answers.get(q["id"])
        if val is None or val == "":
            continue
        label = val
        if q["type"] == "choice":
            label = next((o["label"] for o in q.get("options", []) if o["value"] == val), val)
        answer_labels.append({"id": q["id"], "question": q["text"], "value": label})

    purchased_steps = [
        s for s in (outcome.get("purchased_steps") or [])
        if isinstance(s.get("price_rub"), (int, float)) and s.get("price_rub") > 0
    ]
    if outcome.get("purchased") and not purchased_steps and routine and routine.get("steps"):
        purchased_steps = [
            {
                "product_id": s.get("product_id"),
                "name": s["name"],
                "category": s.get("category"),
                "price_rub": s["price_rub"],
            }
            for s in routine["steps"]
        ]
    purchased_total = outcome.get("purchased_total_rub")
    if purchased_steps and not purchased_total:
        purchased_total = sum(s["price_rub"] for s in purchased_steps)

    return {
        **engine.client_list_item(row),
        "answers": answers,
        "answer_labels": answer_labels,
        "routine": routine,
        "skin_scan": skin_scan,
        "outcome": {
            **outcome,
            "purchased_steps": purchased_steps,
            "purchased_total_rub": purchased_total,
            "purchased_total_formatted": engine.format_money(purchased_total) if purchased_total else None,
        },
        "duration_sec": row.get("duration_sec"),
        "duration_label": _duration_label(row.get("duration_sec") or 0),
        "stages": stages,
        "error": row.get("error"),
        "total_formatted": routine.get("total_formatted") if routine else None,
    }
