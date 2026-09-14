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
        assert "размыт" in str(e) or "фильтр" in str(e) or "фокус" in str(e), str(e)
results.append(run("размытое фото — запрашиваем новое", t_blur))

# 6b. Умеренное размытие (раньше проходило) → тоже отклоняем
def t_blur_moderate():
    img = base_face().filter(ImageFilter.GaussianBlur(4))
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(img))
        raise AssertionError("умеренно размытое фото не отклонено")
    except ValueError as e:
        assert "размыт" in str(e) or "фильтр" in str(e) or "фокус" in str(e), str(e)
results.append(run("умеренно размытое фото — запрашиваем новое", t_blur_moderate))

# 6b2. Очки на лице → просим снять и переснять
def t_glasses():
    img = base_face()
    d = ImageDraw.Draw(img)
    # оправы + блики на линзах (как на реальном селфи в очках)
    for box in ((195, 295, 315, 385), (325, 295, 445, 385)):
        d.ellipse(box, outline=(30, 28, 26), width=12)
    d.rectangle([300, 330, 340, 352], fill=(25, 22, 20))
    d.ellipse([240, 320, 275, 355], fill=(255, 255, 255))
    d.ellipse([370, 320, 405, 355], fill=(255, 255, 255))
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(img))
        raise AssertionError("фото в очках не отклонено")
    except ValueError as e:
        msg = str(e).lower()
        assert "очк" in msg, str(e)
results.append(run("фото в очках — просим снять и переснять", t_glasses))

# 6c. Лицо обрезано краем кадра → просим другое
def t_face_cropped():
    img = base_face()
    canvas = Image.new("RGB", (W, H), (120, 122, 126))
    # сдвигаем лицо влево — правая половина уходит за край
    face = img.crop((140, 100, 500, 720))
    canvas.paste(face, (-120, 100))
    try:
        cosmetic_engine.analyze_skin_photo(to_bytes(canvas))
        raise AssertionError("обрезанное лицо не отклонено")
    except ValueError as e:
        msg = str(e).lower()
        assert ("лиц" in msg or "кадр" in msg or "обрез" in msg or "смещен" in msg), str(e)
results.append(run("обрезанное лицо — запрашиваем новое", t_face_cropped))

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

# 8b. Ровный лоб с верхним затемнением ≠ пигментация на чистом лбу;
#     реальное пятно на щеке должно попасть в щёку, не уехать на лоб.
def t_forehead_not_false_pigment():
    img = base_face()
    d = ImageDraw.Draw(img)
    # мягкий градиент «верхний свет» — лоб чуть темнее, без пятен
    px = img.load()
    for y in range(120, 280):
        for x in range(200, 440):
            r, g, b = px[x, y]
            shade = int((280 - y) * 0.18)
            px[x, y] = (max(0, r - shade), max(0, g - shade), max(0, b - shade))
    # явное пигментное пятно на щеке
    d.ellipse([255, 475, 295, 515], fill=(148, 108, 88))
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    pig = [z for z in scan["zones"] if z["metric_id"] == "pigmentation"]
    for z in pig:
        assert z.get("region") != "forehead", \
            f"ровный лоб помечен как пигментация: {z}"
    if pig:
        assert any("cheek" in (z.get("region") or "") for z in pig), \
            f"пятно на щеке не найдено как пигментация щёк: {pig}"
results.append(run("ровный лоб ≠ пигментация; пятно остаётся на щеке", t_forehead_not_false_pigment))

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

# 11. Чёлка / волосы на лбу ≠ расширенные поры
def t_bangs_not_pores():
    img = base_face()
    d = ImageDraw.Draw(img)
    px = img.load()
    # тёмные «пряди» на верхнем лбу / линии роста — имитация чёлки
    for i, x0 in enumerate(range(220, 420, 14)):
        for t in range(0, 55):
            x = x0 + (t % 3) - 1
            y = 130 + t + (i % 5)
            if 0 <= x < img.width and 0 <= y < img.height:
                r, g, b = px[x, y]
                px[x, y] = (max(0, r - 70), max(0, g - 75), max(0, b - 65))
        d.line([(x0, 125), (x0 + 2, 185)], fill=(60, 45, 35), width=2)
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    pores = [z for z in scan["zones"] if z["metric_id"] == "pores"]
    for z in pores:
        assert z.get("region") != "forehead", f"чёлка на лбу как поры: {z}"
        assert "Лоб" not in (z.get("area") or ""), f"поры в зоне лба: {z}"
        # маркер не должен сидеть у линии роста (верх лица)
        assert z["y"] >= 28.0, f"маркер пор слишком высоко (похоже на волосы): y={z['y']}% {z}"
results.append(run("чёлка/волосы на лбу ≠ расширенные поры", t_bangs_not_pores))

# 12. Тёмные круги / морщины / усталость — маркеры ПОД глазами, не на зрачках
def t_eye_markers_under_not_on_pupils():
    from cosmetic_vision import (
        _detect_face_haar, _face_frac, _EXCLUDE, _in_rect,
        _decode, _Grid, _find_eye_centers,
    )
    img = base_face()
    d = ImageDraw.Draw(img)
    d.ellipse([210, 365, 300, 415], fill=(135, 100, 85))
    d.ellipse([340, 365, 430, 415], fill=(135, 100, 85))
    for y in (372, 380, 388):
        d.line([(220, y), (290, y)], fill=(125, 95, 80), width=1)
        d.line([(350, y), (420, y)], fill=(125, 95, 80), width=1)
    data = to_bytes(img)
    scan = cosmetic_engine.analyze_skin_photo(data)
    face = _detect_face_haar(img)
    assert face, "лицо не найдено"
    bbox = (
        int(face[0] * W), int(face[1] * H),
        int(face[2] * W), int(face[3] * H),
    )
    # рабочие координаты сетки анализа
    _img2, px, gw, gh = _decode(data)
    grid = _Grid(px, gw, gh)
    gb = (
        max(0, int(face[0] * gw)), max(0, int(face[1] * gh)),
        min(gw - 1, int(face[2] * gw)), min(gh - 1, int(face[3] * gh)),
    )
    eyes = _find_eye_centers(grid, gb, source_img=_img2)
    eye_types = ("dark_circles", "tired_eyes", "wrinkles", "puffiness")
    eye_zones = [
        z for z in scan["zones"]
        if z["metric_id"] in eye_types
        or "under_eye" in (z.get("region") or "")
    ]
    assert eye_zones, f"нет глазных зон: {[z['metric_id'] for z in scan['zones']]}"
    for z in eye_zones:
        cx = z["x"] / 100.0 * W
        cy = z["y"] / 100.0 * H
        fx, fy = _face_frac(cx, cy, bbox)
        rid = z.get("region") or ""
        if "crow_feet" in rid:
            # гусиные лапки у внешнего угла — рядом с глазом ок, на зрачке нет
            assert fx <= 0.34 or fx >= 0.66, f"гусиные лапки не у внешнего угла: {z}"
            assert fy >= 0.36, f"гусиные лапки слишком высоко: {z}"
        elif z["metric_id"] in ("dark_circles", "tired_eyes", "puffiness") or "under_eye" in rid:
            # подглазье соседствует с хардкод-зоной глаза — проверяем по зрачкам ниже
            pass
        else:
            assert not any(_in_rect(fx, fy, r) for r in _EXCLUDE), \
                f"маркер в глазу/рту: {z['metric_id']} face=({fx:.3f},{fy:.3f}) img%=({z['x']:.1f},{z['y']:.1f})"
        if z["metric_id"] in ("dark_circles", "tired_eyes", "puffiness") or "under_eye" in rid:
            # только относительно найденных зрачков — абсолютный y_px ломается
            # при сдвиге Haar-бокса / размера глаз на synthetic
            side = "left" if z["x"] < 50 else "right"
            assert side in eyes, f"нет глаза {side} для маркера: {z}"
            ey = eyes[side][1]
            eye_y_pct = 100.0 * ey / gh
            assert z["y"] > eye_y_pct + 2.5, \
                f"маркер не ниже зрачка: zone_y={z['y']} eye_y={eye_y_pct:.1f} y_px={cy:.0f} {z}"
            from cosmetic_vision import _geom_hits_eye
            geom = {"x": z["x"], "y": z["y"], "w": 6, "h": 6}
            assert not _geom_hits_eye(geom, gb, eyes, grid), \
                f"маркер попал в глаз: {z['metric_id']} {z}"
            if z["metric_id"] == "tired_eyes":
                assert "under_eye" in rid, f"усталость должна быть под глазом: {z}"
results.append(run("глазные маркеры — под глазами, не на зрачках", t_eye_markers_under_not_on_pupils))

# 13. Тёмные круги и усталость взгляда — маркеры на ОБОИХ глазах
def t_both_eyes_markers():
    random.seed(7)
    img = base_face()
    d = ImageDraw.Draw(img)
    d.ellipse([210, 365, 300, 415], fill=(135, 100, 85))
    d.ellipse([340, 365, 430, 415], fill=(135, 100, 85))
    for y in (372, 380, 388):
        d.line([(220, y), (290, y)], fill=(125, 95, 80), width=1)
        d.line([(350, y), (420, y)], fill=(125, 95, 80), width=1)
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    dark = [z for z in scan["zones"] if z["metric_id"] == "dark_circles"]
    assert len(dark) >= 2, f"dark_circles: ожидались маркеры на оба глаза, получили {dark}"
    xs = sorted(z["x"] for z in dark)
    assert xs[0] < 50 < xs[-1], f"dark_circles: оба маркера слева и справа: {xs}"
    tired = [z for z in scan["zones"] if z["metric_id"] == "tired_eyes"]
    assert len(tired) >= 2, f"усталость: нужен маркер на каждый глаз, получили {tired}"
    xs = sorted(z["x"] for z in tired)
    assert xs[0] < 50 < xs[-1], f"усталость: оба глаза: {xs}"
results.append(run("тёмные круги/усталость — по маркеру на каждый глаз", t_both_eyes_markers))

# 14. Густые брови ≠ глаза (загруженные фото: маркер уезжал на зрачок)
def t_brows_not_mistaken_for_eyes():
    from cosmetic_vision import _decode, _Grid, _detect_face_haar, _find_eye_centers
    img = base_face()
    d = ImageDraw.Draw(img)
    # брови заметно темнее и массивнее глаз — как на реальных селфи
    d.rectangle([200, 282, 305, 308], fill=(38, 26, 20))
    d.rectangle([335, 282, 440, 308], fill=(38, 26, 20))
    d.ellipse([210, 365, 300, 415], fill=(135, 100, 85))
    d.ellipse([340, 365, 430, 415], fill=(135, 100, 85))
    data = to_bytes(img)
    _img2, px, gw, gh = _decode(data)
    face = _detect_face_haar(_img2)
    assert face, "лицо не найдено"
    gb = (
        max(0, int(face[0] * gw)), max(0, int(face[1] * gh)),
        min(gw - 1, int(face[2] * gw)), min(gh - 1, int(face[3] * gh)),
    )
    eyes = _find_eye_centers(_Grid(px, gw, gh), gb, source_img=_img2)
    for side, eye in eyes.items():
        eye_y_px = eye[1] / gh * H
        assert eye_y_px > 315, \
            f"{side}: центр глаза уехал на бровь (y={eye_y_px:.0f}px, бровь ~297px)"
    scan = cosmetic_engine.analyze_skin_photo(data)
    for z in scan["zones"]:
        if "under_eye" not in (z.get("region") or ""):
            continue
        y_px = z["y"] / 100.0 * H
        assert y_px > 370, f"маркер на глазу: {z['metric_id']} y={y_px:.0f}px"
results.append(run("густые брови ≠ глаза; маркеры остаются под глазами", t_brows_not_mistaken_for_eyes))

# 15. Все глазные признаки — строго по два маркера, слева и справа
def t_eye_features_always_symmetric():
    img = base_face()
    d = ImageDraw.Draw(img)
    # тени только под одним глазом — вторую сторону достраиваем зеркально
    d.ellipse([210, 365, 300, 415], fill=(120, 88, 76))
    for y in (372, 380, 388):
        d.line([(220, y), (290, y)], fill=(112, 82, 70), width=1)
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    per_type = {}
    for z in scan["zones"]:
        if z["metric_id"] in ("dark_circles", "tired_eyes", "puffiness") \
                or "under_eye" in (z.get("region") or ""):
            per_type.setdefault(z["metric_id"], []).append(z)
    assert per_type, "глазные признаки не найдены"
    for ftype, zs in per_type.items():
        sides = {"left" if "left" in (z.get("region") or "") else "right" for z in zs}
        assert sides == {"left", "right"}, f"{ftype}: маркеры только с одной стороны ({zs})"
        # разные признаки не должны лежать точка в точку
        ys = sorted(z["y"] for z in zs)
        assert abs(ys[0] - ys[-1]) < 3.0, f"{ftype}: стороны на разной высоте {ys}"
results.append(run("глазные признаки — всегда симметричная пара", t_eye_features_always_symmetric))

# 16. Мешок под глазами ≠ мелкие морщины
def t_bags_not_wrinkles():
    """Мягкая объёмная полка под глазами должна стать puffiness, не wrinkles."""
    img = base_face()
    px = img.load()
    # объёмный мешок: плавный градиент (светлый холмик → тень полки), без тонких линий
    for side_x0, side_x1 in ((205, 300), (340, 435)):
        for y in range(360, 425):
            for x in range(side_x0, side_x1):
                r, g, b = px[x, y]
                t = (y - 360) / 65.0
                n = int(18 * (1 - t) - 35 * t)
                px[x, y] = (
                    max(0, min(255, r + n)),
                    max(0, min(255, g + n - 2)),
                    max(0, min(255, b + n - 4)),
                )
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    ids = {f["id"] for f in scan["features"]}
    wr_under = [
        z for z in scan["zones"]
        if z["metric_id"] == "wrinkles" and "under_eye" in (z.get("region") or "")
    ]
    puff = [z for z in scan["zones"] if z["metric_id"] == "puffiness"]
    assert "puffiness" in ids or len(puff) >= 1, f"мешок не найден: features={ids} zones={scan['zones']}"
    assert not wr_under, f"мешок ошибочно помечен как морщины: {wr_under}"
    assert len(puff) >= 2, f"мешки должны быть на оба глаза: {puff}"
results.append(run("мешки под глазами ≠ мелкие морщины", t_bags_not_wrinkles))

# 17. Не выдумывать морщины у угла глаз на ровной коже
def t_no_invented_eye_wrinkles():
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(base_face()))
    wr = [z for z in scan["zones"] if z["metric_id"] == "wrinkles"]
    crow = [z for z in wr if "crow" in (z.get("region") or "")]
    under = [z for z in wr if "under_eye" in (z.get("region") or "")]
    assert not crow, f"ложные гусиные лапки на чистом лице: {crow}"
    assert not under, f"ложные морщины под глазами на чистом лице: {under}"
results.append(run("не выдумывать морщины вокруг глаз на ровной коже", t_no_invented_eye_wrinkles))

# 18. Реальные тонкие линии вокруг глаз обязаны находиться
def t_real_eye_wrinkles_found():
    """Тонкие складки под глазами и у внешних углов — это морщины, их нельзя пропускать."""
    img = base_face()
    d = ImageDraw.Draw(img)
    # мелкие линии под обоими глазами
    for x0, x1 in ((215, 295), (345, 425)):
        for i, y in enumerate(range(372, 400, 6)):
            inset = 4 * i
            d.line([x0 + inset, y, x1 - inset, y + 2], fill=(172, 132, 114), width=2)
    # гусиные лапки: лучи от внешних углов
    for corner_x, sign in ((213, -1), (427, 1)):
        for k, dy in enumerate((-10, 0, 10)):
            d.line(
                [corner_x, 340 + dy, corner_x + sign * 34, 340 + dy + (k - 1) * 8],
                fill=(172, 132, 114),
                width=2,
            )
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    scan = cosmetic_engine.analyze_skin_photo(to_bytes(img))
    wr = [z for z in scan["zones"] if z["metric_id"] == "wrinkles"]
    assert wr, f"морщины не найдены, хотя линии есть: {[z['metric_id'] for z in scan['zones']]}"
    sides = {"left" if "left" in (z.get("region") or "") else "right" for z in wr}
    assert len(sides) == 2, f"морщины найдены только с одной стороны: {wr}"
results.append(run("реальные морщины вокруг глаз — находятся", t_real_eye_wrinkles_found))



print()
print(f"{sum(results)}/{len(results)} проверок пройдено")
raise SystemExit(0 if all(results) else 1)
