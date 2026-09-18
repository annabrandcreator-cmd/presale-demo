# -*- coding: utf-8 -*-
"""Движок подбора ухода Oomph (демо, rule-based) для кабинета бренда."""
import json
import os
import uuid

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(APP_DIR, "catalog.json")

TIER_ORDER = {"economy": 0, "mid": 1, "premium": 2}
BUDGET_MAX = {"economy": 3500, "mid": 7000, "premium": 99999}
CATEGORY_ORDER = ["cleanser", "exfoliant", "toner", "serum", "oil", "cream", "eye", "mask", "lip"]

CONCERN_LABELS = {
    "dryness": "Сухость и обезвоженность",
    "acne": "Высыпания и несовершенства",
    "aging": "Возрастные изменения",
    "sensitivity": "Чувствительность и покраснения",
    "pores": "Поры и жирный блеск",
    "dullness": "Тусклость и неровный тон",
    "barrier": "Восстановление барьера",
}

SKIN_LABELS = {
    "dry": "Сухая",
    "oily": "Жирная",
    "combination": "Комбинированная",
    "normal": "Нормальная",
    "sensitive": "Чувствительная",
}

BUDGET_LABELS = {
    "economy": "До 3 500 ₽",
    "mid": "3 500–7 000 ₽",
    "premium": "От 7 000 ₽",
}

ROUTINE_LABELS = {
    "minimal": "Минимальный",
    "basic": "Базовый",
    "advanced": "Расширенный",
}

AGE_LABELS = {
    "18_25": "18–25",
    "26_35": "26–35",
    "36_45": "36–45",
    "46_plus": "46+",
}

SOURCE_LABELS = {
    "web": "Сайт",
    "widget": "Виджет",
    "telegram": "Telegram",
}

STATUS_LABELS = {
    "completed": "Завершена",
    "awaiting_input": "Ждёт ответов",
    "processing": "В процессе",
    "new": "Новая",
    "error": "Ошибка",
}

# Маппинг демо-запросов → фильтры Oomph (effects / tasks / concerns)
CONCERN_MATCH = {
    "dryness": ["uvlazhnenie", "vosstanovlenie", "pitanie", "sukhaya", "obezvozh", "barrier"],
    "acne": ["anti-akne", "antibakter", "vospalen", "nesovershen", "ochischenie", "matir"],
    "aging": ["antivozrast", "morschin", "lifting", "uplotnen", "molodost", "peptide"],
    "sensitivity": ["uspokoen", "pokrasnen", "chuvstvit", "myagkoe", "uspokoenie"],
    "pores": ["pory", "matir", "zhirnaya", "sebum", "ochischenie"],
    "dullness": ["siyanie", "ton", "blesk", "radiance", "osvetl"],
    "barrier": ["vosstanovlenie", "barrier", "ceramides", "pitanie", "uspokoen"],
}

SKIN_MATCH = {
    "dry": ["sukhaya", "obezvozh", "vse-tipy"],
    "oily": ["zhirnaya", "kombinirovannaya", "vse-tipy"],
    "combination": ["kombinirovannaya", "zhirnaya", "vse-tipy", "normalnaya"],
    "normal": ["normalnaya", "vse-tipy"],
    "sensitive": ["chuvstvitelnaya", "pokrasnen", "vse-tipy"],
}


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def new_session_id():
    return uuid.uuid4().hex[:12]


def _product_tokens(product):
    parts = []
    for key in ("effects", "tasks", "concerns", "tags", "skin_types"):
        parts.extend(product.get(key) or [])
    parts.append(product.get("benefit") or "")
    parts.append(product.get("name") or "")
    return " ".join(str(x).lower() for x in parts)


def _score_product(product, answers):
    concern = answers.get("primary_concern")
    skin = answers.get("skin_type")
    score = 0
    blob = _product_tokens(product)

    for needle in CONCERN_MATCH.get(concern, []):
        if needle in blob:
            score += 8

    for needle in SKIN_MATCH.get(skin, []):
        if needle in blob:
            score += 5

    if answers.get("routine_level") == "minimal" and product["category"] in ("cleanser", "cream"):
        score += 2
    if answers.get("routine_level") == "advanced" and product["category"] in ("serum", "oil", "mask", "eye"):
        score += 2
    return score


def _has_price(product):
    price = product.get("price_rub")
    return isinstance(price, (int, float)) and price > 0


def _pick_best(candidates, answers, budget_tier):
    if not candidates:
        return None

    def sort_key(p):
        tier_penalty = abs(TIER_ORDER.get(p.get("tier", "mid"), 1) - TIER_ORDER.get(budget_tier, 1))
        return (-_score_product(p, answers), tier_penalty, p.get("price_rub") or 0)

    return sorted(candidates, key=sort_key)[0]


def build_routine(answers, catalog):
    concern = answers["primary_concern"]
    skin = answers["skin_type"]
    budget = answers.get("budget", "mid")
    routine_level = answers.get("routine_level", "basic")
    concern_label = CONCERN_LABELS.get(concern, concern)

    products = [p for p in catalog["products"] if _has_price(p)]
    picked = []
    used_categories = set()

    categories_needed = ["cleanser", "cream"]
    if routine_level in ("basic", "advanced"):
        categories_needed.insert(1, "toner")
        categories_needed.insert(2, "serum")
    if routine_level == "advanced":
        categories_needed.extend(["eye", "mask"])
    if concern in ("aging", "barrier") and "oil" not in categories_needed and routine_level != "minimal":
        categories_needed.append("oil")
    if concern in ("acne", "pores") and "exfoliant" not in categories_needed and routine_level == "advanced":
        categories_needed.insert(1, "exfoliant")

    for cat in categories_needed:
        pool = [p for p in products if p["category"] == cat and cat not in used_categories]
        best = _pick_best(pool, answers, budget)
        if best:
            picked.append(best)
            used_categories.add(cat)

    subtotal = sum(p["price_rub"] for p in picked)
    budget_max = BUDGET_MAX.get(budget, 7000)

    while subtotal > budget_max and len(picked) > 2:
        removable = [p for p in picked if p["category"] not in ("cleanser", "cream")]
        if not removable:
            break
        drop = max(removable, key=lambda p: p["price_rub"])
        picked.remove(drop)
        used_categories.discard(drop["category"])
        subtotal = sum(p["price_rub"] for p in picked)

    promo = catalog["promo"]
    discount_pct = promo.get("discount_pct") or promo.get("discount_percent") or 5
    discount = round(subtotal * discount_pct / 100)
    total = subtotal - discount

    steps = []
    for i, p in enumerate(picked, 1):
        steps.append({
            "step": i,
            "time": _step_time(p["category"]),
            "product_id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price_rub": p["price_rub"],
            "benefit": p.get("benefit") or "",
            "image": p.get("image"),
            "url": p.get("url"),
        })

    return {
        "concern_label": concern_label,
        "skin_type": skin,
        "routine_level": routine_level,
        "steps": steps,
        "products": [
            {
                "id": p["id"],
                "name": p["name"],
                "category": p["category"],
                "price_rub": p["price_rub"],
                "tier": p.get("tier", "mid"),
                "benefit": p.get("benefit") or "",
                "image": p.get("image"),
                "url": p.get("url"),
            }
            for p in picked
        ],
        "subtotal_rub": subtotal,
        "discount_rub": discount,
        "discount_percent": discount_pct,
        "promo_code": promo.get("code", "OOMPH5"),
        "promo_label": promo.get("label", ""),
        "total_rub": total,
        "tips": _skin_tips(answers),
        "positions": len(picked),
    }


def _step_time(category):
    mapping = {
        "cleanser": "утро и вечер",
        "exfoliant": "2–3 раза в неделю",
        "toner": "после умывания",
        "serum": "утро или вечер",
        "oil": "вечер",
        "cream": "утро и вечер",
        "eye": "утро и вечер",
        "mask": "1–2 раза в неделю",
        "lip": "по необходимости",
    }
    return mapping.get(category, "")


def _skin_tips(answers):
    tips = []
    concern = answers.get("primary_concern")
    if concern == "dryness":
        tips.append("Не умывайтесь горячей водой — она усиливает сухость.")
    if concern == "pores":
        tips.append("Лёгкие текстуры и регулярное очищение помогают контролю блеска.")
    if concern == "sensitivity":
        tips.append("Вводите новые средства по одному, с интервалом 5–7 дней.")
    if concern == "aging":
        tips.append("Сочетайте восстанавливающий уход с защитой от солнца днём.")
    if answers.get("routine_level") == "minimal":
        tips.append("Начните с очищения и крема — потом добавьте сыворотку.")
    return tips[:3]


def format_money(n):
    return f"{int(round(n or 0)):,}".replace(",", " ") + " ₽"


def session_summary(answers, routine, catalog):
    name = answers.get("contact_name", "")
    greeting = f"{name}, " if name else ""
    return {
        "brand": catalog["brand"]["name"],
        "contact_name": name,
        "greeting": greeting,
        "concern": routine["concern_label"],
        "skin_type": answers.get("skin_type"),
        "skin_type_label": SKIN_LABELS.get(answers.get("skin_type"), answers.get("skin_type")),
        "subtotal_rub": routine["subtotal_rub"],
        "subtotal_formatted": format_money(routine["subtotal_rub"]),
        "discount_rub": routine["discount_rub"],
        "discount_formatted": format_money(routine["discount_rub"]),
        "total_rub": routine["total_rub"],
        "total_formatted": format_money(routine["total_rub"]),
        "promo_code": routine["promo_code"],
        "promo_label": routine["promo_label"],
        "positions": routine["positions"],
    }


def client_list_item(row, catalog=None):
    answers = json.loads(row.get("answers") or "{}")
    routine = json.loads(row["routine"]) if row.get("routine") else None
    outcome = json.loads(row["outcome"]) if row.get("outcome") else None
    skin_scan = json.loads(row["skin_scan"]) if row.get("skin_scan") else None
    name = answers.get("contact_name") or "Без имени"
    concern = answers.get("primary_concern")
    return {
        "id": row["id"],
        "contact_name": name,
        "title": f"{CONCERN_LABELS.get(concern, concern or '—')} · {name}",
        "concern": concern,
        "concern_label": CONCERN_LABELS.get(concern, concern),
        "skin_type": answers.get("skin_type"),
        "skin_type_label": SKIN_LABELS.get(answers.get("skin_type"), ""),
        "status": row["status"],
        "status_label": STATUS_LABELS.get(row["status"], row["status"]),
        "stage": row["stage"],
        "source": row.get("source") or "web",
        "source_label": SOURCE_LABELS.get(row.get("source") or "web", "Сайт"),
        "total_rub": routine["total_rub"] if routine else None,
        "total_formatted": format_money(routine["total_rub"]) if routine else None,
        "positions": routine["positions"] if routine else 0,
        "has_photo": bool(skin_scan),
        "photo_url": skin_scan.get("photo_url") if skin_scan else None,
        "purchased": bool(outcome and outcome.get("purchased")),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def catalog_avg_price(catalog):
    prices = [p.get("price_rub") for p in catalog.get("products", []) if p.get("price_rub")]
    if not prices:
        return 1200
    return round(sum(prices) / len(prices))
