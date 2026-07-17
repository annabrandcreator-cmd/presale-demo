# -*- coding: utf-8 -*-
"""Проверки логики анализа фото (запуск: python3 test_vision_checks.py)."""
import io
import random

from PIL import Image, ImageDraw, ImageFilter

import cosmetic_engine
from cosmetic_vision import PhotoQualityError, analyze

random.seed(7)

W, H = 640, 800
SKIN = (208, 164, 142)


def base_face(brightness=1.0):
    img = Image.new("RGB", (W, H), (120, 122, 126))
    d = ImageDraw.Draw(img)
    tone = tuple(int(c * brightness) for c in SKIN)
    # лицо-эллипс + лоб
    d.ellipse([140, 120, 500, 700], fill=tone)
    # глаза (тёмные, не кожа)
    d.ellipse([215, 320, 295, 360], fill=(70, 55, 50))
    d.ellipse([345, 320, 425, 360], fill=(70, 55, 50))
    # брови
    d.rectangle([205, 290, 300, 305], fill=(80, 60, 50))
    d.rectangle([340, 290, 435, 305], fill=(80, 60, 50))
    # губы
    d.ellipse([265, 560, 375, 605], fill=(170, 95, 100))
    # лёгкая кожная текстура, чтобы фото не считалось «замыленным»
    px = img.load()
    for y in range(H):
        for x in range(W):
            r, g, b = px[x, y]
            if r > g > b and r > 150:
                n = random.randint(-7, 7)
                px[x, y] = (max(0, min(255, r + n)),
                            max(0, min(255, g + n)),
                            max(0, min(255, b + n)))
    return img


def to_bytes(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=92)
    return buf.getvalue()


def run(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        return False


results = []

# 1. Чистая кожа — не выдумываем много проблем
def t_clean():
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(base_face()))
    n = len(scan["features"])
    assert n <= 2, f"на чистой коже найдено {n} признаков: {[f['id'] for f in scan['features']]}"
    assert len(scan["zones"]) == n, "zones и features разошлись"
results.append(run("чистая кожа — без выдуманных проблем", t_clean))

# 2. Красное воспаление на щеке — маркер точно на нём
def t_red_spot():
    img = base_face()
    d = ImageDraw.Draw(img)
    # воспаление на левой щеке (px 230, 500)
    d.ellipse([215, 485, 245, 515], fill=(225, 110, 105))
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    reds = [z for z in scan["zones"] if z["metric_id"] in ("inflammation", "redness")]
    assert reds, f"краснота не найдена, найдено: {[z['metric_id'] for z in scan['zones']]}"
    z = reds[0]
    ex, ey = 100 * 230 / W, 100 * 500 / H
    assert abs(z["x"] - ex) < 6 and abs(z["y"] - ey) < 6, \
        f"маркер ({z['x']}, {z['y']}) далеко от воспаления ({ex:.1f}, {ey:.1f})"
    ids = {f["id"] for f in scan["features"]}
    assert z["metric_id"] in ids, "тег без строки в блоке особенностей"
results.append(run("воспаление — маркер попадает на него", t_red_spot))

# 3. Зеркальное отражение — x зеркалится
def t_mirror():
    img = base_face()
    d = ImageDraw.Draw(img)
    d.ellipse([215, 485, 245, 515], fill=(225, 110, 105))
    s1 = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    s2 = cosmetic_engine.analyze_skin_photo(
        to_bytes(img.transpose(Image.FLIP_LEFT_RIGHT)))
    z1 = [z for z in s1["zones"] if z["metric_id"] in ("inflammation", "redness")][0]
    z2 = [z for z in s2["zones"] if z["metric_id"] in ("inflammation", "redness")][0]
    assert abs((100 - z1["x"]) - z2["x"]) < 6, \
        f"зеркало: ожидали x≈{100 - z1['x']:.1f}, получили {z2['x']}"
    assert abs(z1["y"] - z2["y"]) < 5, "y сместился при зеркалировании"
results.append(run("зеркальное фото — координаты отражаются", t_mirror))

# 4. Изменение масштаба — результат сохраняется
def t_scale():
    img = base_face()
    d = ImageDraw.Draw(img)
    d.ellipse([215, 485, 245, 515], fill=(225, 110, 105))
    s1 = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    big = img.resize((W * 2, H * 2), Image.LANCZOS)
    s2 = cosmetic_engine.analyze_skin_photo(to_bytes(big))
    z1 = [z for z in s1["zones"] if z["metric_id"] in ("inflammation", "redness")][0]
    z2 = [z for z in s2["zones"] if z["metric_id"] in ("inflammation", "redness")][0]
    assert abs(z1["x"] - z2["x"]) < 6 and abs(z1["y"] - z2["y"]) < 6, \
        f"масштаб: ({z1['x']},{z1['y']}) vs ({z2['x']},{z2['y']})"
results.append(run("масштабирование — координаты стабильны", t_scale))

# 5. Тёмное фото → просим другое
def t_dark():
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(base_face(brightness=0.25)))
        raise AssertionError("тёмное фото не отклонено")
    except ValueError as e:
        assert "тёмн" in str(e) or "лицо" in str(e).lower() or "освещ" in str(e), str(e)
results.append(run("тёмное фото — запрашиваем новое", t_dark))

# 6. Размытое фото → просим другое
def t_blur():
    img = base_face().filter(ImageFilter.GaussianBlur(14))
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(img))
        raise AssertionError("размытое фото не отклонено")
    except ValueError as e:
        assert "размыт" in str(e) or "фильтр" in str(e), str(e)
results.append(run("размытое фото — запрашиваем новое", t_blur))

# 7. Нет лица (пейзаж) → просим другое
def t_no_face():
    img = Image.new("RGB", (W, H), (90, 140, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 500, W, H], fill=(60, 120, 70))
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(img))
        raise AssertionError("фото без лица не отклонено")
    except ValueError:
        pass
results.append(run("фото без лица — запрашиваем новое", t_no_face))

# 8. Тени под глазами не становятся пигментацией на щеках
def t_under_eye():
    img = base_face()
    d = ImageDraw.Draw(img)
    d.ellipse([210, 370, 300, 400], fill=(150, 115, 100))
    d.ellipse([340, 370, 430, 400], fill=(150, 115, 100))
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    for z in scan["zones"]:
        if z["metric_id"] == "pigmentation":
            assert "cheek" not in z.get("region", ""), \
                f"тень под глазом определена как пигментация на щеке: {z}"
results.append(run("тени под глазами ≠ пигментация щёк", t_under_eye))

# 9. Строгая синхронизация тегов и блока особенностей
def t_sync():
    img = base_face()
    d = ImageDraw.Draw(img)
    d.ellipse([215, 485, 245, 515], fill=(225, 110, 105))
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    zone_ids = {z["metric_id"] for z in scan["zones"]}
    feature_ids = {f["id"] for f in scan["features"]}
    assert zone_ids == feature_ids, f"теги {zone_ids} != строки {feature_ids}"
    from cosmetic_vision import CONF_FLOOR
    for f in scan["features"]:
        assert f["confidence"] >= CONF_FLOOR, f"показан признак с низкой уверенностью: {f}"
        assert f["severity_label"] in ("слабая", "умеренная", "выраженная")
results.append(run("теги и «Видимые особенности» из одного результата", t_sync))

print()
print(f"{sum(results)}/{len(results)} проверок пройдено")
raise SystemExit(0 if all(results) else 1)
