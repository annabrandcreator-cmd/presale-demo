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
    zone_types = {z["metric_id"] for z in scan["zones"]}
    feat_types = {f["id"] for f in scan["features"]}
    assert zone_types == feat_types, "zones и features разошлись"
    assert len(scan["zones"]) <= max(1, 2 * n), "слишком много маркеров на тип"
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
        assert "тёмн" in str(e) or "лиц" in str(e).lower() or "освещ" in str(e), str(e)
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
    except ValueError as e:
        assert "лиц" in str(e).lower(), str(e)
results.append(run("фото без лица — запрашиваем новое", t_no_face))

# 7b. Потолок / штукатурка телесного тона — тоже не лицо
def t_ceiling():
    img = Image.new("RGB", (W, H), (214, 188, 168))
    px = img.load()
    rnd = random.Random(3)
    for y in range(H):
        for x in range(W):
            n = rnd.randint(-8, 8)
            r, g, b = px[x, y]
            px[x, y] = (
                max(0, min(255, r + n)),
                max(0, min(255, g + n)),
                max(0, min(255, b + n)),
            )
    # лёгкая «лампа» — круг ярче, как на потолке
    d = ImageDraw.Draw(img)
    d.ellipse([220, 260, 420, 460], fill=(232, 210, 190))
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(img))
        raise AssertionError("фото потолка не отклонено")
    except ValueError as e:
        assert "лиц" in str(e).lower(), str(e)
results.append(run("потолок вместо лица — запрашиваем новое", t_ceiling))

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

# 10. Маркер на щеке — внутри лица, не на крайнем силуэте
def t_cheek_inward():
    img = base_face()
    d = ImageDraw.Draw(img)
    # тёмное пигментное пятно глубоко на левой щеке (не у края эллипса)
    d.ellipse([250, 470, 290, 510], fill=(155, 118, 98))
    # текстурный шум пор в центре той же щеки
    px = img.load()
    for y in range(460, 520):
        for x in range(245, 295):
            r, g, b = px[x, y]
            n = ((x * 17 + y * 13) % 11) - 5
            px[x, y] = (max(0, min(255, r + n * 3)),
                        max(0, min(255, g + n)),
                        max(0, min(255, b + n)))
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    cheek = [
        z for z in scan["zones"]
        if "cheek" in (z.get("region") or "")
        or "Щека" in (z.get("area") or "")
    ]
    # если нашли щечные зоны — они должны быть inward (не край кадра лица)
    for z in cheek:
        # лицо-эллипс ≈ x 140..500 → в % кадра ~22..78; край силуэта ~22/78
        assert 26.0 <= z["x"] <= 74.0, \
            f"маркер на краю силуэта: x={z['x']}% ({z['metric_id']}, {z.get('region')})"
        assert 38.0 <= z["y"] <= 72.0, \
            f"маркер слишком низко/высоко: y={z['y']}% ({z['metric_id']})"
    # отдельный кейс: красное пятно у самого края щеки не должно дать маркер на контуре
    img2 = base_face()
    d2 = ImageDraw.Draw(img2)
    d2.ellipse([148, 500, 175, 540], fill=(200, 95, 90))  # почти на краю эллипса
    d2.ellipse([250, 470, 285, 505], fill=(220, 105, 100))  # настоящее внутри
    scan2 = cosmetic_engine.analyze_skin_photo(to_bytes(img2))
    reds = [z for z in scan2["zones"] if z["metric_id"] in ("inflammation", "redness", "rosacea_like")]
    if reds:
        for z in reds:
            assert z["x"] >= 24.0, f"краснота на левом краю лица: x={z['x']}"
            assert z["x"] <= 76.0, f"краснота на правом краю лица: x={z['x']}"
results.append(run("щечные маркеры — внутри лица, не на силуэте", t_cheek_inward))

print()
print(f"{sum(results)}/{len(results)} проверок пройдено")
raise SystemExit(0 if all(results) else 1)
