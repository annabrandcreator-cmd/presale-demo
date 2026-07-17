# -*- coding: utf-8 -*-
"""Движок персонального подбора ухода + демо-анализ фото кожи."""
import json
import os
import uuid

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(APP_DIR, "cosmetic_catalog.json")

TIER_ORDER = {"economy": 0, "mid": 1, "premium": 2}
BUDGET_MAX = {"economy": 4000, "mid": 8000, "premium": 99999}

CONCERN_LABELS = {
    "lifting": "лифтинг и упругость",
    "pores": "сужение пор",
    "aging": "возрастные изменения",
    "dryness": "увлажнение",
    "dullness": "сияние и ровный тон",
    "sensitivity": "чувствительность",
    "acne": "несовершенства",
}

SKIN_LABELS = {
    "dry": "сухая",
    "oily": "жирная",
    "combination": "комбинированная",
    "normal": "нормальная",
    "sensitive": "чувствительная",
}

SKIN_LABELS_GEN = {
    "dry": "сухой",
    "oily": "жирной",
    "combination": "комбинированной",
    "normal": "нормальной",
    "sensitive": "чувствительной",
}

AGE_LABELS = {
    "18_25": "18–25",
    "26_35": "26–35",
    "36_45": "36–45",
    "46_plus": "46+",
}

CATEGORY_LABELS = {
    "cleanser": "очищение",
    "toner": "тоник",
    "serum": "сыворотка",
    "cream": "крем",
    "eye": "уход за веками",
    "spf": "SPF",
    "mask": "маска",
}


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def new_session_id():
    return uuid.uuid4().hex[:12]


def _question_visible(q, answers):
    when = q.get("when")
    if not when:
        return True
    for key, allowed in when.items():
        val = answers.get(key)
        if isinstance(allowed, list):
            if val not in allowed:
                return False
        elif val != allowed:
            return False
    return True


def next_question(answers, catalog):
    for q in catalog["questions"]:
        if not _question_visible(q, answers):
            continue
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

    return True, str(value).strip()[:80]


def _score_product(product, answers, skin_scan=None):
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

    if answers.get("acid_tolerance") == "low" and product["id"] in ("toner-pore", "serum-acne", "serum-retinol", "mask-glow"):
        score -= 8
    if answers.get("retinoid_experience") == "none" and product["id"] == "serum-retinol":
        score -= 6
    if answers.get("pregnancy") == "yes" and product["id"] in ("serum-retinol", "serum-acne"):
        score -= 20

    if skin_scan:
        metrics = {m["id"]: m["score"] for m in skin_scan.get("metrics", [])}
        if metrics.get("pores", 0) >= 55 and "pores" in product.get("concerns", []):
            score += 3
        if metrics.get("hydration", 100) <= 45 and "dryness" in product.get("concerns", []):
            score += 3
        if metrics.get("redness", 0) >= 50 and "sensitivity" in product.get("concerns", []):
            score += 3
        if metrics.get("radiance", 100) <= 45 and "dullness" in product.get("concerns", []):
            score += 2

    return score


def _pick_best(candidates, answers, budget_tier, skin_scan=None):
    if not candidates:
        return None

    def sort_key(p):
        tier_penalty = abs(TIER_ORDER.get(p["tier"], 1) - TIER_ORDER.get(budget_tier, 1))
        return (-_score_product(p, answers, skin_scan), tier_penalty, p["price_rub"])

    return sorted(candidates, key=sort_key)[0]


def _why_for_you(product, answers, skin_scan=None):
    concern = answers.get("primary_concern")
    skin = answers.get("skin_type")
    concern_l = CONCERN_LABELS.get(concern, concern)
    skin_l = SKIN_LABELS_GEN.get(skin, SKIN_LABELS.get(skin, skin))
    cat = CATEGORY_LABELS.get(product["category"], product["category"])

    bits = [f"Для вашей {skin_l} кожи и задачи «{concern_l}» — шаг «{cat}»."]
    bits.append(product["benefit"].rstrip(".") + ".")

    if skin_scan and skin_scan.get("headline"):
        top = skin_scan.get("priority_concern")
        if top and top in product.get("concerns", []):
            bits.append("Совпадает с зоной внимания по фото-анализу.")

    if answers.get("acid_tolerance") == "low" and product["id"] not in ("toner-pore", "serum-acne", "serum-retinol"):
        bits.append("Без агрессивных кислот — с учётом вашей чувствительности.")

    if answers.get("pregnancy") == "yes":
        bits.append("Подходит в рамках демо-ограничений для периода беременности.")

    return " ".join(bits[:3])


def _forecast_outcomes(answers, skin_scan=None):
    """2/4-week demo forecasts tied to the client's primary concern."""
    concern = answers.get("primary_concern") or "dryness"
    templates = {
        "lifting": {
            "week2": "Кожа ощущается более плотной, контур — чуть собраннее",
            "week4": "Видимый эффект лифтинга: упругость контура +{pct}%",
            "base": 18,
        },
        "pores": {
            "week2": "Поры выглядят чище, меньше сального блеска в Т-зоне",
            "week4": "Визуальное сужение пор примерно на {pct}%",
            "base": 16,
        },
        "aging": {
            "week2": "Тон ровнее, кожа мягче на ощупь",
            "week4": "Выраженность мелких морщин ниже примерно на {pct}%",
            "base": 14,
        },
        "dryness": {
            "week2": "Меньше стянутости, комфорт сохраняется дольше в течение дня",
            "week4": "Уровень увлажнённости +{pct}%",
            "base": 22,
        },
        "dullness": {
            "week2": "Кожа выглядит свежее, тон визуально ровнее",
            "week4": "Сияние и ровность тона +{pct}%",
            "base": 17,
        },
        "sensitivity": {
            "week2": "Меньше реакции на внешние раздражители",
            "week4": "Снижение видимых покраснений примерно на {pct}%",
            "base": 15,
        },
        "acne": {
            "week2": "Меньше новых воспалений при регулярном уходе",
            "week4": "Видимое снижение несовершенств примерно на {pct}%",
            "base": 19,
        },
    }
    item = templates.get(concern, templates["dryness"])
    boost = 0
    age = answers.get("age_range")
    skin = answers.get("skin_type")
    level = answers.get("routine_level")
    if age in ("18_25", "26_35"):
        boost += 2
    if level == "advanced":
        boost += 2
    elif level != "minimal":
        boost += 1
    if skin in ("oily", "combination") and concern in ("pores", "acne", "dullness"):
        boost += 2
    if skin in ("dry", "sensitive") and concern in ("dryness", "sensitivity"):
        boost += 2
    if age in ("36_45", "46_plus") and concern in ("lifting", "aging"):
        boost += 2
    if skin_scan and skin_scan.get("priority_concern") == concern:
        boost += 2
    pct = max(12, min(item["base"] + boost, 28))
    return {
        "concern": concern,
        "weeks": [
            {"at": "Через 2 недели", "text": item["week2"]},
            {"at": "Через 4 недели", "text": item["week4"].format(pct=pct), "percent": pct},
        ],
        "disclaimer": "Ориентир для демо при регулярном уходе. Не является медицинским прогнозом.",
    }


def build_profile(answers, skin_scan=None):
    concern = answers.get("primary_concern")
    skin = answers.get("skin_type")
    age = answers.get("age_range")
    routine = answers.get("routine_level", "basic")

    strengths = []
    focus = []

    if skin == "normal":
        strengths.append("Баланс себума в норме")
    if skin == "combination":
        focus.append("Т-зона требует отдельного контроля")
    if concern == "dryness":
        focus.append("Приоритет — восстановление барьера и увлажнение")
    if concern == "pores":
        focus.append("Нужны очищение пор и лёгкие текстуры")
    if concern == "sensitivity":
        focus.append("Минимум раздражителей, спокойный актив")
    if concern == "aging" or concern == "lifting":
        focus.append("Работа с плотностью и профилактикой")
    if concern == "dullness":
        focus.append("Сияние и ровный тон")
    if concern == "acne":
        focus.append("Контроль несовершенств без пересушивания")

    if skin_scan:
        for m in skin_scan.get("metrics", []):
            if m["score"] >= 60 and m["id"] in ("pores", "redness", "dullness", "fine_lines"):
                focus.append(f"{m['label']}: повышенное внимание")
            if m["score"] <= 35 and m["id"] in ("hydration", "radiance", "barrier"):
                focus.append(f"{m['label']}: ниже комфортного уровня")
            if m["id"] == "hydration" and m["score"] >= 65:
                strengths.append("Увлажнённость в хорошем диапазоне")

    if not strengths:
        strengths.append("Есть понятный запрос — можно собрать точный протокол")
    if not focus:
        focus.append("Соберём последовательный уход без лишних шагов")

    summary = (
        f"{SKIN_LABELS.get(skin, skin).capitalize()} кожа"
        f"{', ' + AGE_LABELS.get(age, '') if age else ''}"
        f" · приоритет: {CONCERN_LABELS.get(concern, concern)}"
    )

    return {
        "title": "Ваш профиль кожи",
        "summary": summary,
        "skin_type_label": SKIN_LABELS.get(skin, skin),
        "concern_label": CONCERN_LABELS.get(concern, concern),
        "age_label": AGE_LABELS.get(age, ""),
        "routine_level": routine,
        "strengths": strengths[:3],
        "focus": focus[:4],
        "from_photo": bool(skin_scan),
    }


def build_routine(answers, catalog, skin_scan=None):
    concern = answers["primary_concern"]
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
        categories_needed.append("spf")
    if concern in ("lifting", "aging") and routine_level != "minimal":
        categories_needed.append("eye")
    if concern in ("dullness", "pores") and routine_level == "advanced":
        categories_needed.append("mask")

    for cat in categories_needed:
        pool = [p for p in products if p["category"] == cat and cat not in used_categories]
        best = _pick_best(pool, answers, budget, skin_scan)
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
            "category_label": CATEGORY_LABELS.get(p["category"], p["category"]),
            "price_rub": p["price_rub"],
            "benefit": p["benefit"],
            "why": _why_for_you(p, answers, skin_scan),
        })

    profile = build_profile(answers, skin_scan)
    tips = _skin_tips(answers, skin_scan)
    narrative = _matching_narrative(answers, profile, steps)
    outcomes = _forecast_outcomes(answers, skin_scan)

    return {
        "concern_label": concern_label,
        "skin_type": answers.get("skin_type"),
        "routine_level": routine_level,
        "profile": profile,
        "steps": steps,
        "products": picked,
        "subtotal_rub": subtotal,
        "discount_rub": discount,
        "discount_percent": promo["discount_percent"],
        "promo_code": promo["code"],
        "promo_label": promo["label"],
        "total_rub": total,
        "tips": tips,
        "narrative": narrative,
        "outcomes": outcomes,
        "positions": len(picked),
        "skin_scan": skin_scan,
    }


def _matching_narrative(answers, profile, steps):
    name = answers.get("contact_name") or ""
    greet = f"{name}, " if name else ""
    lines = [
        f"{greet}собрала профиль: {profile['summary']}.",
        "Сверяю совместимость активов и текстур с вашими ответами…",
        f"Готовый протокол: {len(steps)} шага(ов) в логичной последовательности ухода.",
    ]
    return lines


def _step_time(category):
    return {
        "cleanser": "утро и вечер",
        "toner": "после умывания",
        "serum": "утро или вечер",
        "cream": "утро и вечер",
        "eye": "утро и вечер",
        "spf": "утро",
        "mask": "1–2 раза в неделю",
    }.get(category, "")


def _skin_tips(answers, skin_scan=None):
    tips = []
    concern = answers.get("primary_concern")
    if concern == "dryness":
        tips.append("Не умывайтесь горячей водой — она усиливает сухость.")
    if concern == "pores":
        tips.append("Ежедневный SPF помогает порам меньше «раскрываться» от солнца.")
    if concern == "sensitivity":
        tips.append("Вводите новые средства по одному, с интервалом 5–7 дней.")
    if concern == "aging":
        tips.append("Ретинол и SPF — база anti-age; не смешивайте всё сразу.")
    if answers.get("routine_level") == "minimal":
        tips.append("2 недели держите очищение + крем, потом добавьте сыворотку.")
    if skin_scan and skin_scan.get("tip"):
        tips.insert(0, skin_scan["tip"])
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
        "from_photo": bool(routine.get("skin_scan")),
    }


# ── Photo / skin scan (demo Vision) ─────────────────────────────────────────


# Какой запрос ухода закрывает каждый видимый признак.
_FEATURE_CONCERN = {
    "inflammation": "acne",
    "redness": "sensitivity",
    "rosacea_like": "sensitivity",
    "pigmentation": "dullness",
    "dark_circles": "dullness",
    "pores": "pores",
    "shine": "pores",
    "wrinkles": "aging",
    "nasolabial": "lifting",
    "dryness": "dryness",
}

_CONCERN_TIPS = {
    "pores": "Сфокусируемся на мягком очищении и ниацинамиде без пересушивания.",
    "sensitivity": "Начнём с восстановления барьера — активные кислоты позже.",
    "dryness": "Сначала церамиды и увлажнение, затем точечные активы.",
    "dullness": "Добавим мягкое обновление и антиоксиданты для тона.",
    "aging": "Плотность + SPF — база; ретинол только если кожа готова.",
    "acne": "Мягкое очищение и точечный уход без пересушивания кожи.",
    "lifting": "Работа с упругостью: пептиды, массаж и обязательный SPF.",
}


def _zone_dict(f, i):
    return {
        "id": f"{f['type']}_{i}",
        "metric_id": f["type"],
        "label": f["label"],
        "area": f["region_label"],
        "region": f["region"],
        "x": f["geom"]["x"],
        "y": f["geom"]["y"],
        "w": f["geom"]["w"],
        "h": f["geom"]["h"],
        "score": f["score"],
        "status": "Обнаружено",
        "attention": "high",
        "severity": f["severity"],
        "severity_label": f["severity_label"],
        "confidence": f["confidence"],
        "evidence": f["evidence"],
    }


def analyze_skin_photo(image_bytes, filename="photo.jpg"):
    """
    Анализ фото: качество → сегментация лица → поиск видимых признаков →
    уверенность и выраженность → маркер в центре найденной области.
    Не медицинская диагностика; фото не сохраняется.
    """
    from cosmetic_vision import analyze as vision_analyze, PhotoQualityError

    if not image_bytes or len(image_bytes) < 100:
        raise ValueError("Пустое изображение")
    if len(image_bytes) > 8_000_000:
        raise ValueError("Файл слишком большой (макс. 8 МБ)")

    try:
        vision = vision_analyze(image_bytes)
    except PhotoQualityError as e:
        # Честный отказ вместо угадывания: фронт попросит другое фото.
        raise ValueError(str(e))

    findings = vision["findings"]

    # Теги, маркеры и строки блока «Видимые особенности кожи» —
    # строго из одного списка findings, чтобы они не расходились.
    # Один тип признака = один тег и одна строка (по самому выраженному месту);
    # маркеров на фото для типа может быть до двух (например, обе щеки).
    per_type = {}
    for f in findings:
        per_type.setdefault(f["type"], []).append(f)

    features = []
    zones = []
    for ftype, items in per_type.items():
        # до 2 маркеров одного типа — обычно левая и правая сторона
        items = sorted(
            items,
            key=lambda f: (f["confidence"] + f.get("strength", 0), f["confidence"]),
            reverse=True,
        )
        picked = []
        seen_side = set()
        for f in items:
            rid = f.get("region") or ""
            side = (
                "L" if "left" in rid else
                "R" if "right" in rid else
                rid
            )
            if side in seen_side:
                continue
            picked.append(f)
            seen_side.add(side)
            if len(picked) >= 2:
                break
        if not picked:
            continue
        top = picked[0]
        area_label = top["region_label"]
        if len(picked) == 2:
            area_label = f"{picked[0]['region_label']} и {picked[1]['region_label']}"
        features.append({
            "id": ftype,
            "label": top["label"],
            "area": area_label,
            "score": top["score"],
            "severity": top["severity"],
            "severity_label": top["severity_label"],
            "confidence": top["confidence"],
            "evidence": top["evidence"],
        })
        for i, f in enumerate(picked):
            zones.append(_zone_dict(f, i))

    # До 6 типов признаков — самые уверенные и выраженные первыми.
    features.sort(key=lambda f: f["confidence"] + f["score"] / 100.0, reverse=True)
    features = features[:6]
    kept_types = {f["id"] for f in features}
    zones = [z for z in zones if z["metric_id"] in kept_types]

    # Приоритетный запрос — от самого выраженного подтверждённого признака.
    if findings:
        top = findings[0]
        priority = _FEATURE_CONCERN.get(top["type"], "dryness")
        headline = f"Вижу акцент на «{CONCERN_LABELS.get(priority, priority)}»"
    else:
        m = vision["metrics"]
        priority = "dryness" if m["hydration"] < 55 else "dullness"
        headline = "Выраженных проблемных зон не нашла — кожа выглядит ровной"

    skin_guess = vision["skin_type"]
    m = vision["metrics"]
    metrics = [
        {"id": "hydration", "label": "Увлажнённость", "score": m["hydration"], "hint": "комфорт"},
        {"id": "pores", "label": "Поры", "score": m["pores"], "hint": "видимость"},
        {"id": "redness", "label": "Покраснения", "score": m["redness"], "hint": "реактивность"},
        {"id": "radiance", "label": "Сияние", "score": m["radiance"], "hint": "ровность тона"},
        {"id": "fine_lines", "label": "Морщинки", "score": m["fine_lines"], "hint": "мелкие линии"},
        {"id": "barrier", "label": "Барьер кожи", "score": m["barrier"], "hint": "защита"},
    ]

    tip = _CONCERN_TIPS.get(priority, "Соберём уход вокруг вашей главной зоны внимания.")

    return {
        "ok": True,
        "mode": "vision",
        "disclaimer": "Демо-анализ по фото. Не заменяет консультацию косметолога или врача. Фото не сохраняется.",
        "headline": headline,
        "priority_concern": priority,
        "suggested_skin_type": skin_guess,
        "suggested_concern": priority,
        "quality": vision["quality"],
        "metrics": metrics,
        "features": features,
        "zones": zones,
        "tip": tip,
        "narrative": [
            "Проверяю свет и зону лица…",
            "Считываю текстуру, тон и зоны внимания…",
            f"Готово: приоритет — {CONCERN_LABELS.get(priority, priority)}.",
        ],
    }
