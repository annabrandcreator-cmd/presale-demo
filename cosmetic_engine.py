# -*- coding: utf-8 -*-
"""Движок подбора ухода (демо, rule-based)."""
import json
import os
import re
import uuid

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(APP_DIR, "cosmetic_catalog.json")

TIER_ORDER = {"economy": 0, "mid": 1, "premium": 2}
BUDGET_MAX = {"economy": 4000, "mid": 8000, "premium": 99999}
CATEGORY_ORDER = ["cleanser", "toner", "serum", "cream", "eye", "spf", "mask"]

CONCERN_LABELS = {
    "lifting": "лифтинг и упругость",
    "pores": "сужение пор",
    "aging": "возрастные изменения",
    "dryness": "увлажнение",
    "dullness": "сияние и ровный тон",
    "sensitivity": "чувствительность",
    "acne": "несовершенства",
}


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def new_session_id():
    return uuid.uuid4().hex[:12]


def next_question(answers, catalog):
    for q in catalog["questions"]:
        if q["id"] not in answers or answers[q["id"]] in (None, ""):
            return q
    return None


def validate_answer(question, value, catalog):
    if value is None or str(value).strip() == "":
        return False, "Укажите значение"

    if question["type"] == "choice":
        allowed = {o["value"] for o in question.get("options", [])}
        if value not in allowed:
            return False, "Выберите вариант"
        return True, value

    return True, str(value).strip()


def _score_product(product, answers):
    concern = answers.get("primary_concern")
    skin = answers.get("skin_type")
    score = 0

    if concern in product.get("concerns", []):
        score += 10
    if skin in product.get("skin_types", []):
        score += 6
    if answers.get("age_range") in ("36_45", "46_plus") and "aging" in product.get("concerns", []):
        score += 2
    if answers.get("routine_level") == "minimal" and product["category"] in ("cleanser", "cream"):
        score += 1
    if answers.get("routine_level") == "advanced" and product["category"] in ("serum", "spf", "mask", "eye"):
        score += 2
    return score


def _pick_best(candidates, answers, budget_tier):
    if not candidates:
        return None
    budget_max = BUDGET_MAX.get(budget_tier, 8000)

    def sort_key(p):
        tier_penalty = abs(TIER_ORDER.get(p["tier"], 1) - TIER_ORDER.get(budget_tier, 1))
        return (-_score_product(p, answers), tier_penalty, p["price_rub"])

    ranked = sorted(candidates, key=sort_key)
    return ranked[0]


def build_routine(answers, catalog):
    """Подбор персонального набора ухода."""
    concern = answers["primary_concern"]
    skin = answers["skin_type"]
    budget = answers.get("budget", "mid")
    routine_level = answers.get("routine_level", "basic")
    concern_label = CONCERN_LABELS.get(concern, concern)

    products = catalog["products"]
    picked = []
    used_categories = set()

    categories_needed = ["cleanser", "serum", "cream"]
    if routine_level in ("basic", "advanced"):
        categories_needed.insert(1, "toner")
    if routine_level == "advanced":
        categories_needed.extend(["spf"])
    if concern in ("lifting", "aging") and routine_level != "minimal":
        categories_needed.append("eye")
    if concern in ("dullness", "pores") and routine_level == "advanced":
        categories_needed.append("mask")

    for cat in categories_needed:
        pool = [
            p for p in products
            if p["category"] == cat and cat not in used_categories
        ]
        best = _pick_best(pool, answers, budget)
        if best:
            picked.append(best)
            used_categories.add(cat)

    subtotal = sum(p["price_rub"] for p in picked)
    budget_max = BUDGET_MAX.get(budget, 8000)

    while subtotal > budget_max and len(picked) > 2:
        removable = [p for p in picked if p["category"] not in ("cleanser", "cream")]
        if not removable:
            break
        drop = max(removable, key=lambda p: p["price_rub"])
        picked.remove(drop)
        used_categories.discard(drop["category"])
        subtotal = sum(p["price_rub"] for p in picked)

    promo = catalog["promo"]
    discount = round(subtotal * promo["discount_percent"] / 100)
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
            "benefit": p["benefit"],
        })

    tips = _skin_tips(answers)

    return {
        "concern_label": concern_label,
        "skin_type": skin,
        "routine_level": routine_level,
        "steps": steps,
        "products": picked,
        "subtotal_rub": subtotal,
        "discount_rub": discount,
        "discount_percent": promo["discount_percent"],
        "promo_code": promo["code"],
        "promo_label": promo["label"],
        "total_rub": total,
        "tips": tips,
        "positions": len(picked),
    }


def _step_time(category):
    mapping = {
        "cleanser": "утро и вечер",
        "toner": "после умывания",
        "serum": "утро или вечер",
        "cream": "утро и вечер",
        "eye": "утро и вечер",
        "spf": "утро",
        "mask": "1–2 раза в неделю",
    }
    return mapping.get(category, "")


def _skin_tips(answers):
    tips = []
    concern = answers.get("primary_concern")
    if concern == "dryness":
        tips.append("Не умывайтесь горячей водой — она усиливает сухость.")
    if concern == "pores":
        tips.append("SPF каждый день помогает pores не «раскрываться» от солнца.")
    if concern == "sensitivity":
        tips.append("Вводите новые средства по одному, с интервалом 5–7 дней.")
    if concern == "aging":
        tips.append("Ретинол и SPF — база anti-age ухода; не смешивайте без консультации.")
    if answers.get("routine_level") == "minimal":
        tips.append("Начните с очищения и крема 2 недели — потом добавьте сыворотку.")
    return tips[:3]


def format_money(n):
    return f"{int(round(n)):,}".replace(",", " ") + " ₽"


def session_summary(answers, routine, catalog):
    name = answers.get("contact_name", "")
    greeting = f"{name}, " if name else ""
    return {
        "brand": catalog["brand"]["name"],
        "contact_name": name,
        "greeting": greeting,
        "concern": routine["concern_label"],
        "skin_type": answers.get("skin_type"),
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
