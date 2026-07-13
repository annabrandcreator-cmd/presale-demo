# -*- coding: utf-8 -*-
"""Движок квалификации и подбора оборудования (демо, rule-based)."""
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(APP_DIR, "catalog.json")


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def new_deal_id():
    return uuid.uuid4().hex[:12]


def parse_free_text(text, catalog):
    """Пытается извлечь параметры из свободного описания."""
    t = (text or "").lower()
    out = {}

    for key, meta in catalog["object_types"].items():
        label = meta["label"].lower()
        if any(w in t for w in label.split()[:2]):
            out["object_type"] = key
            break
    if "object_type" not in out:
        if any(w in t for w in ("склад", "логист")):
            out["object_type"] = "warehouse"
        elif any(w in t for w in ("чист", "фарм", "iso")):
            out["object_type"] = "cleanroom"
        elif any(w in t for w in ("пищ", "хлеб", "молоч")):
            out["object_type"] = "food"
        elif any(w in t for w in ("офис", "админ")):
            out["object_type"] = "office"
        elif any(w in t for w in ("цех", "производ", "завод")):
            out["object_type"] = "production"

    m = re.search(r"(\d[\d\s]{1,6})\s*м²", t) or re.search(r"площад[ьяь].*?(\d{2,5})", t)
    if m:
        out["area_m2"] = float(m.group(1).replace(" ", ""))

    m = re.search(r"высот[аы].*?(\d(?:[.,]\d)?)\s*м", t) or re.search(r"(\d(?:[.,]\d)?)\s*м\s*высот", t)
    if m:
        out["height_m"] = float(m.group(1).replace(",", "."))

    m = re.search(r"(-?\d{1,2})\s*°", t) or re.search(r"мороз.*?(-?\d{1,2})", t)
    if m:
        out["temp_outdoor"] = float(m.group(1))

    for fc in ("H13", "F7", "G4", "G3"):
        if fc.lower() in t:
            out["filter_class"] = fc
            break

    return out


def next_question(answers, catalog):
    """Возвращает следующий вопрос или None, если всё собрано."""
    for q in catalog["questions"]:
        if q["id"] not in answers or answers[q["id"]] in (None, ""):
            return q
    return None


def validate_answer(question, value, catalog):
    qid = question["id"]
    if value is None or str(value).strip() == "":
        return False, "Укажите значение"

    if question["type"] == "number":
        try:
            num = float(str(value).replace(",", "."))
        except ValueError:
            return False, "Введите число"
        if "min" in question and num < question["min"]:
            return False, f"Минимум {question['min']}"
        if "max" in question and num > question["max"]:
            return False, f"Максимум {question['max']}"
        return True, num

    if question["type"] == "email":
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(value).strip()):
            return False, "Некорректный email"
        return True, str(value).strip().lower()

    if question["type"] == "choice":
        if question.get("options_from") == "object_types":
            if value not in catalog["object_types"]:
                return False, "Выберите тип объекта"
            return True, value
        allowed = {o["value"] for o in question.get("options", [])}
        if value not in allowed:
            return False, "Выберите вариант"
        return True, value

    return True, str(value).strip()


def question_options(question, catalog):
    if question.get("options_from") == "object_types":
        return [
            {"value": k, "label": v["label"]}
            for k, v in catalog["object_types"].items()
        ]
    return question.get("options", [])


def calculate_airflow(answers, catalog):
    obj = catalog["object_types"][answers["object_type"]]
    volume = float(answers["area_m2"]) * float(answers["height_m"])
    air_changes = obj["air_changes"]
    return round(volume * air_changes)


def pick_pvu(required_flow, temp_outdoor, filter_class, catalog, log=None):
    candidates = []
    rejected = []
    for p in catalog["products"]:
        if p["category"] != "ПВУ":
            continue
        reasons = []
        if required_flow > p["flow_m3h_max"] * 1.05 or required_flow < p["flow_m3h_min"] * 0.7:
            reasons.append(f"расход {required_flow:,} м³/ч вне диапазона {p['flow_m3h_min']:,}–{p['flow_m3h_max']:,}".replace(",", " "))
        if temp_outdoor < p["temp_min"]:
            reasons.append(f"мороз {temp_outdoor}°C ниже допуска {p['temp_min']}°C")
        if filter_class not in p.get("filter_classes", []):
            reasons.append(f"фильтр {filter_class} не поддерживается установкой")
        if reasons:
            rejected.append((p, reasons))
            continue
        if not (p["flow_m3h_min"] <= required_flow <= p["flow_m3h_max"] * 1.05):
            if required_flow > p["flow_m3h_max"] * 1.05 or required_flow < p["flow_m3h_min"] * 0.7:
                continue
        score = abs((p["flow_m3h_min"] + p["flow_m3h_max"]) / 2 - required_flow)
        candidates.append((score, p))

    if log is not None:
        pvu_count = sum(1 for p in catalog["products"] if p["category"] == "ПВУ")
        log.append({
            "step": "catalog_scan",
            "title": "Проверка каталога",
            "detail": f"Сопоставлено {pvu_count} моделей ПВУ с расходом, температурой и классом фильтрации.",
        })
        for p, reasons in rejected[:4]:
            log.append({
                "step": "rejected",
                "title": "Отклонено",
                "detail": f"{p['name']}: {'; '.join(reasons)}",
            })
        if len(rejected) > 4:
            log.append({
                "step": "rejected",
                "title": "Отклонено",
                "detail": f"…и ещё {len(rejected) - 4} позиций по правилам совместимости",
            })

    if not candidates:
        pvus = [p for p in catalog["products"] if p["category"] == "ПВУ"]
        pvus.sort(key=lambda p: abs((p["flow_m3h_min"] + p["flow_m3h_max"]) / 2 - required_flow))
        chosen = pvus[0] if pvus else None
        if log is not None and chosen:
            log.append({
                "step": "fallback",
                "title": "Резервный подбор",
                "detail": f"Строгих совпадений нет — выбрана ближайшая по расходу: {chosen['name']}. Требует проверки инженером.",
            })
        return chosen
    candidates.sort(key=lambda x: x[0])
    chosen = candidates[0][1]
    if log is not None:
        log.append({
            "step": "selected",
            "title": "Подобрана ПВУ",
            "detail": f"{chosen['name']} — оптимальное совпадение по расходу и ограничениям объекта.",
        })
    return chosen


def pick_filter(filter_class, catalog):
    mapping = {"G3": "filter-g4", "G4": "filter-g4", "F7": "filter-f7", "H13": "filter-h13"}
    fid = mapping.get(filter_class, "filter-g4")
    return next((p for p in catalog["products"] if p["id"] == fid), None)


def pick_silencer(required_flow, catalog):
    for p in catalog["products"]:
        if p["category"] == "Шумоглушитель" and p["flow_m3h_min"] <= required_flow <= p["flow_m3h_max"]:
            return p
    return None


def price_duct(required_flow, catalog):
    p = next(x for x in catalog["products"] if x["id"] == "duct-kit")
    return round(required_flow * p["price_per_m3h"])


def build_specification(answers, catalog):
    """Подбор комплекта и расчёт сметы."""
    obj_meta = catalog["object_types"][answers["object_type"]]
    volume = float(answers["area_m2"]) * float(answers["height_m"])
    air_changes = obj_meta["air_changes"]
    required_flow = round(volume * air_changes)
    temp_outdoor = float(answers.get("temp_outdoor", -25))
    filter_class = answers.get("filter_class") or obj_meta["default_filter"]

    reasoning_log = [
        {
            "step": "qualify",
            "title": "Нормализация ТЗ",
            "detail": (
                f"Тип «{obj_meta['label']}»: объём {volume:,.0f} м³, кратность {air_changes} 1/ч "
                f"→ расчётный расход {required_flow:,} м³/ч. Это не «3 поля в калькуляторе», "
                f"а инженерная модель под тип объекта.".replace(",", " ")
            ),
        },
        {
            "step": "constraints",
            "title": "Ограничения объекта",
            "detail": f"Учтены: tнаруж = {temp_outdoor}°C, фильтрация {filter_class}, совместимость компонентов комплекта.",
        },
    ]

    lines = []
    notes = []

    pvu = pick_pvu(required_flow, temp_outdoor, filter_class, catalog, log=reasoning_log)
    if not pvu:
        raise ValueError("Не удалось подобрать ПВУ под заданные параметры")

    lines.append({
        "id": pvu["id"],
        "name": pvu["name"],
        "qty": 1,
        "unit": "шт.",
        "price_rub": pvu["price_rub"],
        "sum_rub": pvu["price_rub"],
        "lead_days": pvu["lead_days"],
        "specs": pvu["specs"],
    })

    filt = pick_filter(filter_class, catalog)
    if filt:
        lines.append({
            "id": filt["id"],
            "name": filt["name"],
            "qty": 2,
            "unit": "компл.",
            "price_rub": filt["price_rub"],
            "sum_rub": filt["price_rub"] * 2,
            "lead_days": filt["lead_days"],
            "specs": filt["specs"],
        })

    sil = pick_silencer(required_flow, catalog)
    if sil:
        lines.append({
            "id": sil["id"],
            "name": sil["name"],
            "qty": 2,
            "unit": "шт.",
            "price_rub": sil["price_rub"],
            "sum_rub": sil["price_rub"] * 2,
            "lead_days": sil["lead_days"],
            "specs": sil["specs"],
        })

    duct_sum = price_duct(required_flow, catalog)
    lines.append({
        "id": "duct-kit",
        "name": "Комплект воздуховодов (расчётный)",
        "qty": required_flow,
        "unit": "м³/ч × тариф",
        "price_rub": round(duct_sum / max(required_flow, 1), 2),
        "sum_rub": duct_sum,
        "lead_days": 12,
        "specs": f"Ориентир под расход {required_flow:,} м³/ч".replace(",", " "),
    })

    auto = next(p for p in catalog["products"] if p["id"] == "automation")
    lines.append({
        "id": auto["id"],
        "name": auto["name"],
        "qty": 1,
        "unit": "шт.",
        "price_rub": auto["price_rub"],
        "sum_rub": auto["price_rub"],
        "lead_days": auto["lead_days"],
        "specs": auto["specs"],
    })

    equipment_total = sum(l["sum_rub"] for l in lines)
    install = next(p for p in catalog["products"] if p["id"] == "install")
    install_sum = round(equipment_total * install["price_percent_of_equipment"])
    lines.append({
        "id": install["id"],
        "name": install["name"],
        "qty": 1,
        "unit": "усл.",
        "price_rub": install_sum,
        "sum_rub": install_sum,
        "lead_days": install["lead_days"],
        "specs": install["specs"],
    })

    total = sum(l["sum_rub"] for l in lines)
    lead_days = max(l["lead_days"] for l in lines)

    if temp_outdoor < -30 and pvu["temp_min"] > -35:
        notes.append("Для экстремальных морозов рекомендован преднагрев приточного воздуха — уточните на этапе проектирования.")

    if answers["object_type"] == "cleanroom" and filter_class != "H13":
        notes.append("Для чистых помещений обычно требуется H13 — проверьте нормативы объекта.")
        reasoning_log.append({
            "step": "warning",
            "title": "Предупреждение нормативов",
            "detail": "Чистое помещение без H13 — в боевой системе ИИ эскалирует на инженера.",
        })

    reasoning_log.append({
        "step": "bundle",
        "title": "Сборка комплекта",
        "detail": f"Добавлены сопутствующие позиции: фильтры, воздуховоды, автоматика, монтаж — {len(lines)} строк сметы.",
    })
    reasoning_log.append({
        "step": "production_note",
        "title": "В продакшене",
        "detail": (
            "Свободное ТЗ разбирает GigaChat/YandexGPT, прайс синхронизируется с 1С, "
            "история сделок — в PostgreSQL, документы — в S3 или on-premise."
        ),
    })

    return {
        "required_flow_m3h": required_flow,
        "filter_class": filter_class,
        "object_label": obj_meta["label"],
        "lines": lines,
        "subtotal_rub": equipment_total,
        "total_rub": total,
        "lead_days": lead_days,
        "notes": notes,
        "reasoning_log": reasoning_log,
        "checks_count": len(reasoning_log),
    }


def format_money(n):
    return f"{int(round(n)):,}".replace(",", " ") + " ₽"


def deal_summary(answers, spec, catalog):
    brand = catalog["brand"]
    return {
        "brand": brand["name"],
        "contact_name": answers.get("contact_name", ""),
        "object": spec["object_label"],
        "area_m2": answers.get("area_m2"),
        "height_m": answers.get("height_m"),
        "flow_m3h": spec["required_flow_m3h"],
        "total_rub": spec["total_rub"],
        "total_formatted": format_money(spec["total_rub"]),
        "lead_days": spec["lead_days"],
        "positions": len(spec["lines"]),
    }
