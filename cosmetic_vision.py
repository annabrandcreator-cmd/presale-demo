# -*- coding: utf-8 -*-
"""
Пиксельный анализ фото кожи для демо-консультанта.

Последовательность: проверка качества фото → сегментация кожи лица →
поиск конкретных визуальных признаков в анатомических зонах →
оценка выраженности и уверенности → маркер в центре найденной области.

Определяются только видимые признаки. Не медицинская диагностика.
"""
import io
from collections import deque

GRID_W = 168          # ширина рабочей сетки анализа
MIN_SOURCE_SIDE = 200  # минимальный размер исходного фото, px
CONF_FLOOR = 0.70      # признаки с меньшей уверенностью не показываем


class PhotoQualityError(ValueError):
    """Фото не подходит для анализа — нужен новый снимок."""


# ── Анатомические зоны (доли от рамки лица: x0, y0, x1, y1) ────────────────
# Рамка = Haar-бокс лица, расширенный вверх на лоб. Глаза ≈ y 0.46,
# кончик носа ≈ 0.67, рот ≈ 0.81, подбородок ≈ 0.98.
_REGIONS = [
    ("forehead", "Лоб", 0.24, 0.13, 0.76, 0.32),
    ("glabella", "Межбровье", 0.41, 0.32, 0.59, 0.41),
    ("left_under_eye", "Под глазом слева", 0.18, 0.54, 0.42, 0.63),
    ("right_under_eye", "Под глазом справа", 0.58, 0.54, 0.82, 0.63),
    ("nose", "Нос", 0.41, 0.42, 0.59, 0.66),
    ("left_cheek", "Щека слева", 0.10, 0.58, 0.39, 0.82),
    ("right_cheek", "Щека справа", 0.61, 0.58, 0.90, 0.82),
    ("upper_lip", "Над верхней губой", 0.41, 0.66, 0.59, 0.71),
    ("chin", "Подбородок", 0.36, 0.90, 0.64, 1.0),
]

# Глаза с бровями и рот исключаются из анализа кожи полностью.
_EXCLUDE = [
    (0.10, 0.36, 0.45, 0.54),
    (0.55, 0.36, 0.90, 0.54),
    (0.33, 0.72, 0.67, 0.90),
]

FEATURE_LABELS = {
    "inflammation": "Воспаления",
    "redness": "Покраснения",
    "pigmentation": "Пигментация",
    "dark_circles": "Тёмные круги",
    "pores": "Расширенные поры",
    "shine": "Жирный блеск",
    "wrinkles": "Морщинки",
    "dryness": "Сухость",
}

SEVERITY_LABELS = {"mild": "слабая", "moderate": "умеренная", "high": "выраженная"}


def _severity(strength):
    if strength >= 0.75:
        return "high"
    if strength >= 0.42:
        return "moderate"
    return "mild"


# ── Декодирование ────────────────────────────────────────────────────────────


def _decode(image_bytes):
    try:
        from PIL import Image, ImageOps
    except ImportError as e:  # pragma: no cover
        raise PhotoQualityError(
            "Сервис анализа фото временно недоступен — продолжите без фото."
        ) from e
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception:
        raise PhotoQualityError(
            "Не удалось прочитать файл. Загрузите фото в формате JPG или PNG."
        )
    orig_w, orig_h = img.size
    if min(orig_w, orig_h) < MIN_SOURCE_SIDE:
        raise PhotoQualityError(
            "Фото слишком маленькое для анализа. Сделайте снимок ближе или в большем разрешении."
        )
    grid_h = max(32, round(orig_h * GRID_W / orig_w))
    small = img.resize((GRID_W, grid_h))
    return img, list(small.getdata()), GRID_W, grid_h


def _detect_face_haar(img):
    """Позиция лица каскадом OpenCV; None, если cv2 недоступен или лица нет."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    w, h = img.size
    scale = min(1.0, 760.0 / max(w, h))
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1.0 else img
    gray = np.asarray(small.convert("L"))
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        return None
    min_side = max(48, round(min(gray.shape) * 0.18))
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    sw, sh = small.size
    # расширяем бокс вверх на лоб (каскад начинает от бровей/середины лба)
    top = max(0.0, (y - 0.18 * fh) / sh)
    return (x / sw, top, (x + fw) / sw, min(1.0, (y + fh * 1.02) / sh))


# ── Сегментация кожи ─────────────────────────────────────────────────────────


def _is_skin(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    # Верхние границы отсекают яркую одежду (красная ткань: r-g слишком велико).
    rgb_rule = (
        r > 60 and g > 30 and b > 18
        and r > b and 6 <= (r - g) <= 62 and (r - b) <= 105
        and 12 < (mx - mn) <= 110
    )
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    ycc_rule = 80 <= cb <= 125 and 135 <= cr <= 168
    return rgb_rule or (ycc_rule and r > 60 and (r - g) <= 62)


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


class _Grid:
    def __init__(self, px, w, h):
        self.w = w
        self.h = h
        self.px = px
        self.luma = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in px]
        self.rg = [p[0] - p[1] for p in px]
        self.skin = [_is_skin(*p) for p in px]

    def lap(self, x, y):
        """|лапласиан| яркости — локальный микроконтраст."""
        w, h, L = self.w, self.h, self.luma
        if x <= 0 or y <= 0 or x >= w - 1 or y >= h - 1:
            return 0.0
        i = y * w + x
        return abs(4 * L[i] - L[i - 1] - L[i + 1] - L[i - w] - L[i + w])

    def grad(self, x, y):
        w, L = self.w, self.luma
        if x <= 0 or y <= 0 or x >= self.w - 1 or y >= self.h - 1:
            return 0.0, 0.0
        i = y * w + x
        return abs(L[i + 1] - L[i - 1]) / 2.0, abs(L[i + w] - L[i - w]) / 2.0


def _largest_skin_component(grid):
    w, h = grid.w, grid.h
    seen = [False] * (w * h)
    best = []
    for start in range(w * h):
        if not grid.skin[start] or seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        comp = []
        while queue:
            i = queue.popleft()
            comp.append(i)
            x, y = i % w, i // w
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    j = ny * w + nx
                    if grid.skin[j] and not seen[j]:
                        seen[j] = True
                        queue.append(j)
        if len(comp) > len(best):
            best = comp
    return best


def _face_bbox(grid):
    """Резерв без OpenCV: верх крупнейшей связной области кожи (до линии шеи)."""
    comp = _largest_skin_component(grid)
    if len(comp) < grid.w * grid.h * 0.05:
        raise PhotoQualityError(
            "Не получилось уверенно найти лицо: оно перекрыто, далеко или слабо освещено. "
            "Сделайте селфи анфас при дневном свете."
        )
    rows = {}
    for i in comp:
        rows.setdefault(i // grid.w, []).append(i % grid.w)
    ys = sorted(rows)
    span = {y: (max(xs) - min(xs) + 1) for y, xs in rows.items()}
    head_top = ys[0]
    comp_h = ys[-1] - head_top + 1
    # ширина головы — максимум в верхних 45% компоненты
    top_band = [y for y in ys if y <= head_top + 0.45 * comp_h]
    head_w = max(span[y] for y in top_band)

    # идём вниз: лицо заканчивается там, где кожа резко сужается (шея)
    # или на разумной пропорции головы (~1.35 ширины)
    y_end = ys[-1]
    started = False
    for y in ys:
        if span[y] >= 0.55 * head_w:
            started = True
        if started and y > head_top + 0.7 * head_w and span[y] < 0.58 * head_w:
            y_end = y
            break
        if started and y - head_top > 1.45 * head_w:
            y_end = y
            break

    xs_face = sorted(x for y in ys if y <= y_end for x in rows[y])
    x0 = _percentile(xs_face, 0.04)
    x1 = _percentile(xs_face, 0.96)
    y0, y1 = head_top, y_end
    if (x1 - x0) < grid.w * 0.18 or (y1 - y0) < grid.h * 0.16:
        raise PhotoQualityError(
            "Лицо занимает слишком маленькую часть кадра. Сделайте снимок ближе."
        )
    return x0, y0, x1, y1


# ── Качество фото ────────────────────────────────────────────────────────────


def _check_quality(grid, bbox):
    x0, y0, x1, y1 = bbox
    lumas, sharp = [], []
    for y in range(y0, y1 + 1):
        row = y * grid.w
        for x in range(x0, x1 + 1):
            if grid.skin[row + x]:
                lumas.append(grid.luma[row + x])
                sharp.append(grid.lap(x, y))
    mean_luma = sum(lumas) / max(1, len(lumas))
    mean_sharp = sum(sharp) / max(1, len(sharp))
    if mean_luma < 55:
        raise PhotoQualityError(
            "Фото слишком тёмное — состояние кожи не различить. "
            "Сделайте снимок при дневном свете."
        )
    if mean_luma > 235:
        raise PhotoQualityError(
            "Фото пересвечено — детали кожи не видны. Попробуйте мягкий рассеянный свет."
        )
    if mean_sharp < 1.1:
        raise PhotoQualityError(
            "Фото размыто или сильно сглажено фильтром. "
            "Сделайте чёткий снимок без фильтров."
        )
    return {"mean_luma": round(mean_luma, 1), "sharpness": round(mean_sharp, 2)}


# ── Зоны ─────────────────────────────────────────────────────────────────────


def _in_rect(fx, fy, rect):
    return rect[0] <= fx <= rect[2] and rect[1] <= fy <= rect[3]


def _region_of(fx, fy):
    for rid, label, x0, y0, x1, y1 in _REGIONS:
        if x0 <= fx <= x1 and y0 <= fy <= y1:
            return rid, label
    return None, None


def _is_reddened_skin(r, g, b):
    """Сильно покрасневшая кожа (воспаление) не проходит обычное skin-правило."""
    return r > 95 and g > 30 and r > b and (r - g) > 30 and (g - b) > -20


def _collect_region_pixels(grid, bbox):
    """Пиксели кожи по зонам; глаза/рот/фон исключены."""
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    regions = {rid: [] for rid, *_ in _REGIONS}
    face_pixels = []
    for y in range(y0, y1 + 1):
        row = y * grid.w
        fy = (y - y0) / fh
        for x in range(x0, x1 + 1):
            i = row + x
            if not (grid.skin[i] or _is_reddened_skin(*grid.px[i])):
                continue
            fx = (x - x0) / fw
            if any(_in_rect(fx, fy, r) for r in _EXCLUDE):
                continue
            face_pixels.append((x, y))
            rid, _ = _region_of(fx, fy)
            if rid:
                regions[rid].append((x, y))
    if len(face_pixels) < 400:
        raise PhotoQualityError(
            "Кожа лица почти не видна на снимке — уберите волосы и предметы с лица "
            "и попробуйте ещё раз."
        )
    # Эрозия маски: оставляем только пиксели, чьё окружение — тоже кожа.
    # Отсекает границы волос, украшений и края лица.
    all_pts = set(face_pixels)
    def interior(p):
        x, y = p
        return all(
            (x + dx, y + dy) in all_pts
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
    regions = {rid: [p for p in pts if interior(p)] for rid, pts in regions.items()}
    face_pixels = [p for p in face_pixels if interior(p)]
    if len(face_pixels) < 300:
        raise PhotoQualityError(
            "Кожа лица почти не видна на снимке — уберите волосы и предметы с лица "
            "и попробуйте ещё раз."
        )
    return regions, face_pixels


def _baseline(grid, face_pixels):
    n = len(face_pixels)
    luma = sum(grid.luma[y * grid.w + x] for x, y in face_pixels) / n
    rg = sum(grid.rg[y * grid.w + x] for x, y in face_pixels) / n
    tex = sum(grid.lap(x, y) for x, y in face_pixels) / n
    return {"luma": luma, "rg": rg, "tex": tex}


# ── Связные компоненты аномалий ──────────────────────────────────────────────


def _components(anomaly, grid, min_area=4):
    w, h = grid.w, grid.h
    seen = set()
    comps = []
    for start in anomaly:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        pts = []
        while queue:
            x, y = queue.popleft()
            pts.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) in anomaly and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        if len(pts) >= min_area:
            comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps[:12]


def _comp_geometry(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return cx, cy, min(xs), min(ys), max(xs), max(ys)


def _to_pct(grid, cx, cy, bx0, by0, bx1, by1):
    return {
        "x": round(100.0 * cx / grid.w, 1),
        "y": round(100.0 * cy / grid.h, 1),
        "w": round(max(6.0, 100.0 * (bx1 - bx0 + 1) / grid.w), 1),
        "h": round(max(6.0, 100.0 * (by1 - by0 + 1) / grid.h), 1),
    }


def _face_frac(fx, fy, bbox):
    x0, y0, x1, y1 = bbox
    return (fx - x0) / max(1, x1 - x0), (fy - y0) / max(1, y1 - y0)


# ── Детекторы признаков ──────────────────────────────────────────────────────


def _detect_red(grid, bbox, regions, base):
    """Воспаления (компактные красные элементы) и разлитая краснота."""
    findings = []
    region_pts = {rid: set(pts) for rid, pts in regions.items()}
    allowed = set().union(*region_pts.values()) if region_pts else set()
    anomaly = {
        (x, y) for (x, y) in allowed
        if grid.rg[y * grid.w + x] - base["rg"] > 21
        and 60 < grid.luma[y * grid.w + x] < base["luma"] + 35
    }
    face_area = max(1, len(allowed))
    for pts in _components(anomaly, grid, min_area=3):
        cx, cy, bx0, by0, bx1, by1 = _comp_geometry(pts)
        fx, fy = _face_frac(cx, cy, bbox)
        rid, rlabel = _region_of(fx, fy)
        if not rid:
            continue
        # выраженность считаем по ядру области (верхняя половина пикселей),
        # чтобы сглаживание JPEG по краям не занижало оценку
        deltas = sorted((grid.rg[y * grid.w + x] - base["rg"] for x, y in pts), reverse=True)
        core = deltas[: max(2, len(deltas) // 2)]
        excess = sum(core) / len(core)
        area_frac = len(pts) / face_area
        comp_w = bx1 - bx0 + 1
        comp_h = by1 - by0 + 1
        # губы: очень красная широкая горизонтальная полоса у рта — не кожа
        if fy > 0.60 and excess > 50 and comp_w >= comp_h * 2.0:
            continue
        compact = comp_w * comp_h <= len(pts) * 3.2
        strength = min(1.0, (excess - 21) / 34.0 + area_frac * 4.0)
        conf = min(0.95, 0.5 + (excess - 21) / 55.0 + min(0.12, area_frac * 8))
        ftype = "inflammation" if (area_frac < 0.01 and compact) else "redness"
        evidence = (
            "локальный красный элемент, контрастный к окружающей коже"
            if ftype == "inflammation"
            else "участок кожи заметно краснее среднего тона лица"
        )
        findings.append({
            "type": ftype, "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": evidence,
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    return findings


def _detect_dark_circles(grid, bbox, regions, base):
    findings = []
    for rid in ("left_under_eye", "right_under_eye"):
        pts = regions.get(rid) or []
        if len(pts) < 25:
            continue
        dark = [(x, y) for x, y in pts if base["luma"] - grid.luma[y * grid.w + x] > 22]
        frac = len(dark) / len(pts)
        if frac < 0.45:
            continue
        deficit = sum(base["luma"] - grid.luma[y * grid.w + x] for x, y in dark) / len(dark)
        if deficit > 70:  # настолько тёмное — скорее волосы/оправа, не кожа
            continue
        cx = sum(p[0] for p in dark) / len(dark)
        cy = sum(p[1] for p in dark) / len(dark)
        bx0, by0, bx1, by1 = min(p[0] for p in dark), min(p[1] for p in dark), \
            max(p[0] for p in dark), max(p[1] for p in dark)
        strength = min(1.0, (deficit - 22) / 32.0 + (frac - 0.45) * 0.9)
        conf = min(0.92, 0.5 + frac * 0.3 + deficit / 150.0)
        label = dict((r[0], r[1]) for r in _REGIONS)[rid]
        findings.append({
            "type": "dark_circles", "region": rid, "region_label": label,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "область под глазом заметно темнее среднего тона кожи",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    return findings


def _detect_pigmentation(grid, bbox, regions, base):
    findings = []
    # без upper_lip (тени ноздрей) — пигментацию ищем на щеках, лбу и подбородке
    zone_ids = ("left_cheek", "right_cheek", "forehead", "chin")
    allowed = set()
    for rid in zone_ids:
        allowed.update(regions.get(rid) or [])
    skin_all = set()
    for pts in regions.values():
        skin_all.update(pts)
    anomaly = set()
    for x, y in allowed:
        i = y * grid.w + x
        p = grid.px[i]
        brownish = p[0] > p[1] >= p[2] - 6
        deficit_px = base["luma"] - grid.luma[i]
        # 24..58: темнее кожи, но не настолько, чтобы быть волосами/тенью от них
        if 24 < deficit_px < 58 and brownish and grid.rg[i] - base["rg"] < 12:
            anomaly.add((x, y))
    face_area = max(1, len(skin_all))
    for pts in _components(anomaly, grid, min_area=5):
        area_frac = len(pts) / face_area
        if area_frac > 0.02:  # большое тёмное поле — тень или волосы
            continue
        cx, cy, bx0, by0, bx1, by1 = _comp_geometry(pts)
        # пятно должно быть компактным, а не полосой вдоль края лица
        if (bx1 - bx0 + 1) * (by1 - by0 + 1) > len(pts) * 3.0:
            continue
        # окружение пятна должно быть кожей: пигментация лежит внутри кожи,
        # тени от волос примыкают к границе лица
        ring = set()
        for x, y in pts:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                           (x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)):
                if (nx, ny) not in pts:
                    ring.add((nx, ny))
        skin_ring = sum(1 for p in ring if p in skin_all)
        if ring and skin_ring / len(ring) < 0.75:
            continue
        fx, fy = _face_frac(cx, cy, bbox)
        rid, rlabel = _region_of(fx, fy)
        if rid not in zone_ids:
            continue
        # верхняя кромка лба — зона прядей у линии роста волос, пропускаем
        if rid == "forehead" and fy < 0.20:
            continue
        deficit = sum(base["luma"] - grid.luma[y * grid.w + x] for x, y in pts) / len(pts)
        strength = min(1.0, (deficit - 24) / 30.0)
        conf = min(0.9, 0.48 + (deficit - 24) / 70.0 + min(0.1, area_frac * 15))
        findings.append({
            "type": "pigmentation", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "компактный участок темнее окружающей кожи, коричневатый оттенок",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings[:2]


def _detect_texture(grid, bbox, regions, base):
    """Расширенные поры / неровная текстура по микроконтрасту зоны."""
    findings = []
    for rid in ("nose", "left_cheek", "right_cheek", "chin", "forehead"):
        pts = regions.get(rid) or []
        if len(pts) < 40:
            continue
        # глубокая эрозия зоны: микроконтраст меряем вдали от волос и границ
        pset = set(pts)
        pts = [
            (x, y) for x, y in pts
            if all((x + dx, y + dy) in pset
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        ]
        if len(pts) < 40:
            continue
        laps = [(grid.lap(x, y), x, y) for x, y in pts]
        tex = sum(l[0] for l in laps) / len(laps)
        ratio = tex / max(0.5, base["tex"])
        if tex < 7.5 or ratio < 1.3:
            continue
        top = sorted(laps, reverse=True)[: max(6, len(laps) // 10)]
        cx = sum(t[1] for t in top) / len(top)
        cy = sum(t[2] for t in top) / len(top)
        bx0, by0 = min(t[1] for t in top), min(t[2] for t in top)
        bx1, by1 = max(t[1] for t in top), max(t[2] for t in top)
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (ratio - 1.3) / 1.2 + (tex - 7.5) / 25.0)
        conf = min(0.93, 0.5 + (ratio - 1.3) * 0.4 + tex / 80.0)
        findings.append({
            "type": "pores", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "неоднородная текстура и заметные устья пор относительно остальной кожи",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings[:2]


def _detect_shine(grid, bbox, regions, base):
    findings = []
    for rid in ("forehead", "nose"):
        pts = regions.get(rid) or []
        if len(pts) < 30:
            continue
        bright = [
            (x, y) for x, y in pts
            if grid.luma[y * grid.w + x] - base["luma"] > 38
        ]
        frac = len(bright) / len(pts)
        if frac < 0.22:
            continue
        cx = sum(p[0] for p in bright) / len(bright)
        cy = sum(p[1] for p in bright) / len(bright)
        bx0, by0 = min(p[0] for p in bright), min(p[1] for p in bright)
        bx1, by1 = max(p[0] for p in bright), max(p[1] for p in bright)
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (frac - 0.22) * 2.2 + 0.25)
        conf = min(0.9, 0.55 + frac * 0.8)
        findings.append({
            "type": "shine", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "выраженные блики на коже — признак избытка себума",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    return findings


def _detect_wrinkles(grid, bbox, regions, base):
    """Горизонтальные линии лба и межбровные линии по направленным градиентам."""
    findings = []
    checks = [("forehead", "horizontal"), ("glabella", "vertical")]
    for rid, direction in checks:
        pts = regions.get(rid) or []
        if len(pts) < 40:
            continue
        gx_sum = gy_sum = 0.0
        for x, y in pts:
            gx, gy = grid.grad(x, y)
            gx_sum += gx
            gy_sum += gy
        gx_m = gx_sum / len(pts)
        gy_m = gy_sum / len(pts)
        if direction == "horizontal":
            main, cross = gy_m, gx_m
            evidence = "повторяющиеся горизонтальные линии на лбу"
        else:
            main, cross = gx_m, gy_m
            evidence = "вертикальные линии в межбровной зоне"
        if main < 4.2 or main < cross * 1.6:
            continue
        strong = sorted(
            ((grid.grad(x, y)[1 if direction == "horizontal" else 0], x, y) for x, y in pts),
            reverse=True,
        )[: max(6, len(pts) // 8)]
        cx = sum(s[1] for s in strong) / len(strong)
        cy = sum(s[2] for s in strong) / len(strong)
        bx0, by0 = min(s[1] for s in strong), min(s[2] for s in strong)
        bx1, by1 = max(s[1] for s in strong), max(s[2] for s in strong)
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (main - 4.2) / 6.0 + max(0.0, main / max(0.5, cross) - 1.35) * 0.5)
        conf = min(0.9, 0.5 + (main - 4.2) / 12.0 + (main / max(0.5, cross) - 1.35) * 0.3)
        findings.append({
            "type": "wrinkles", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": evidence,
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    return findings


# ── Сборка результата ────────────────────────────────────────────────────────


def _merge_findings(raw):
    """Один тип в одной зоне = одна область (берём самую уверенную)."""
    best = {}
    for f in raw:
        key = (f["type"], f["region"])
        cur = best.get(key)
        if not cur or (f["confidence"], f["strength"]) > (cur["confidence"], cur["strength"]):
            best[key] = f
    return list(best.values())


def analyze(image_bytes):
    """
    Возвращает dict: quality, baseline-метрики и findings
    (только с уверенностью >= CONF_FLOOR). Бросает PhotoQualityError,
    если по фото нельзя дать честный результат.
    """
    img, px, w, h = _decode(image_bytes)
    grid = _Grid(px, w, h)

    norm = _detect_face_haar(img)
    if norm:
        bbox = (
            max(0, int(norm[0] * w)), max(0, int(norm[1] * h)),
            min(w - 1, int(norm[2] * w)), min(h - 1, int(norm[3] * h)),
        )
        x0, y0, x1, y1 = bbox
        area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        skin_in_box = sum(
            1 for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1)
            if grid.skin[yy * w + xx]
        )
        if skin_in_box / area < 0.22:
            raise PhotoQualityError(
                "Кожа на лице плохо различима — мешают фильтр, плотный макияж или освещение. "
                "Сделайте снимок без фильтров при дневном свете."
            )
    else:
        bbox = _face_bbox(grid)

    quality = _check_quality(grid, bbox)
    regions, face_pixels = _collect_region_pixels(grid, bbox)
    base = _baseline(grid, face_pixels)

    raw = []
    raw += _detect_red(grid, bbox, regions, base)
    raw += _detect_dark_circles(grid, bbox, regions, base)
    raw += _detect_pigmentation(grid, bbox, regions, base)
    raw += _detect_texture(grid, bbox, regions, base)
    raw += _detect_shine(grid, bbox, regions, base)
    raw += _detect_wrinkles(grid, bbox, regions, base)

    merged = _merge_findings(raw)
    findings = [f for f in merged if f["confidence"] >= CONF_FLOOR]
    findings.sort(key=lambda f: (f["confidence"] + f["strength"]), reverse=True)
    findings = findings[:6]

    for f in findings:
        f["severity"] = _severity(f["strength"])
        f["severity_label"] = SEVERITY_LABELS[f["severity"]]
        f["label"] = FEATURE_LABELS[f["type"]]
        f["score"] = int(round(30 + f["strength"] * 65))

    # Глобальные ориентиры для подбора продуктов (та же пиксельная база).
    shine_strength = max((f["strength"] for f in findings if f["type"] == "shine"), default=0.0)
    red_strength = max(
        (f["strength"] for f in findings if f["type"] in ("redness", "inflammation")), default=0.0
    )
    pores_strength = max((f["strength"] for f in findings if f["type"] == "pores"), default=0.0)
    wrinkle_strength = max((f["strength"] for f in findings if f["type"] == "wrinkles"), default=0.0)
    dark_strength = max(
        (f["strength"] for f in findings if f["type"] in ("dark_circles", "pigmentation")),
        default=0.0,
    )

    luma_dev = min(1.0, abs(base["luma"] - 150) / 90.0)
    metrics = {
        "redness": int(round(15 + red_strength * 70)),
        "pores": int(round(20 + pores_strength * 65)),
        "fine_lines": int(round(12 + wrinkle_strength * 70)),
        "hydration": int(round(max(25, 82 - base["tex"] * 2.5 - shine_strength * 8))),
        "radiance": int(round(max(25, 85 - luma_dev * 35 - dark_strength * 25))),
        "barrier": int(round(max(28, 84 - red_strength * 40 - base["tex"] * 1.5))),
    }

    if shine_strength > 0.45 and pores_strength > 0.3:
        skin_type = "oily"
    elif shine_strength > 0.3:
        skin_type = "combination"
    elif red_strength > 0.55:
        skin_type = "sensitive"
    elif metrics["hydration"] < 45:
        skin_type = "dry"
    else:
        skin_type = "normal"

    return {
        "quality": quality,
        "metrics": metrics,
        "findings": findings,
        "skin_type": skin_type,
        "grid": {"w": w, "h": h},
    }
