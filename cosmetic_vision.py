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
CONF_FLOOR = 0.62      # признаки с меньшей уверенностью не показываем
MAX_FINDINGS = 12      # максимум областей в сыром результате (пара на признак)


class PhotoQualityError(ValueError):
    """Фото не подходит для анализа — нужен новый снимок."""


# ── Анатомические зоны (доли от рамки лица: x0, y0, x1, y1) ────────────────
# Рамка = Haar-бокс лица, расширенный вверх на лоб. Глаза ≈ y 0.46,
# кончик носа ≈ 0.67, рот ≈ 0.81, подбородок ≈ 0.98.
_REGIONS = [
    ("forehead", "Лоб", 0.24, 0.13, 0.76, 0.32),
    ("glabella", "Межбровье", 0.41, 0.32, 0.59, 0.41),
    # Под глазом: сразу под нижним веком (не зрачок, не середина щеки)
    ("left_under_eye", "Под глазом слева", 0.20, 0.48, 0.44, 0.60),
    ("right_under_eye", "Под глазом справа", 0.56, 0.48, 0.80, 0.60),
    ("nose", "Нос", 0.41, 0.42, 0.59, 0.66),
    # Носогубная складка: диагональ от крыла носа к уголку рта (не «яблоко» щеки)
    ("left_nasolabial", "Носогубная зона слева", 0.28, 0.58, 0.43, 0.76),
    ("right_nasolabial", "Носогубная зона справа", 0.57, 0.58, 0.72, 0.76),
    # Щёки: внутренний «яблоко» — не край силуэта и не челюсть.
    ("left_cheek", "Щека слева", 0.16, 0.52, 0.40, 0.74),
    ("right_cheek", "Щека справа", 0.60, 0.52, 0.84, 0.74),
    ("upper_lip", "Над верхней губой", 0.41, 0.66, 0.59, 0.71),
    ("chin", "Подбородок", 0.36, 0.90, 0.64, 1.0),
]

# Глаза (зрачок/радужка/веко) и рот — маркеры сюда не ставим.
# Нижняя граница глаз ~0.47: ниже начинается подглазье.
_EXCLUDE = [
    (0.08, 0.32, 0.48, 0.47),   # левый глаз + веко
    (0.52, 0.32, 0.92, 0.47),   # правый глаз + веко
    (0.28, 0.78, 0.72, 0.92),
    (0.34, 0.72, 0.66, 0.78),
]

FEATURE_LABELS = {
    "inflammation": "Отдельные воспаления",
    "redness": "Покраснение",
    "rosacea_like": "Покраснение",  # только визуальный признак, не диагноз
    "pigmentation": "Пигментация",
    "dark_circles": "Тёмные круги",
    "pores": "Заметные поры",
    "shine": "Блеск в Т-зоне",
    "wrinkles": "Мелкие морщины",
    "nasolabial": "Носогубные складки",
    "dryness": "Признаки сухости",
    "dullness": "Тусклость тона",
    "uneven_texture": "Неровность текстуры",
    "puffiness": "Мешки под глазами",
    "tired_eyes": "Признаки усталости взгляда",
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


# Служебные маркеры результата детекции лица.
_FACE_MISSING = object()   # OpenCV доступен, лица нет
_FACE_NO_CV = object()     # OpenCV недоступен — только тогда fallback по коже


def _detect_face_haar(img):
    """
    Позиция лица каскадом OpenCV.
    Возвращает (x0,y0,x1,y1) в долях кадра, либо _FACE_MISSING / _FACE_NO_CV.
    """
    try:
        import cv2
        import numpy as np
        if not hasattr(cv2, "CascadeClassifier"):
            return _FACE_NO_CV
    except ImportError:
        return _FACE_NO_CV
    try:
        w, h = img.size
        scale = min(1.0, 760.0 / max(w, h))
        small = img.resize((max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1.0 else img
        gray = np.asarray(small.convert("L"))
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            return _FACE_NO_CV
        min_side = max(48, round(min(gray.shape) * 0.18))
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
        if faces is None or len(faces) == 0:
            # второй проход чуть мягче — дальние селфи, но не «поллица»
            soft = max(40, round(min(gray.shape) * 0.12))
            faces = cascade.detectMultiScale(gray, 1.08, 4, minSize=(soft, soft))
            soft_pass = True
        else:
            soft_pass = False
        if faces is None or len(faces) == 0:
            return _FACE_MISSING
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        sw, sh = small.size
        # лицо должно занимать заметную долю кадра (не «случайный» квадрат на фоне)
        if (fw * fh) / max(1, sw * sh) < 0.04:
            return _FACE_MISSING
        # Мягкий проход: отсекаем явные обрезы по краю сразу.
        if soft_pass:
            if x <= 2 or y <= 2 or (x + fw) >= sw - 2 or (y + fh) >= sh - 2:
                return _FACE_MISSING
            if (x + fw / 2) / sw < 0.28 or (x + fw / 2) / sw > 0.72:
                return _FACE_MISSING
        # расширяем бокс вверх на лоб (каскад начинает от бровей/середины лба)
        top = max(0.0, (y - 0.18 * fh) / sh)
        return (x / sw, top, (x + fw) / sw, min(1.0, (y + fh * 1.02) / sh))
    except Exception:
        return _FACE_NO_CV


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
    frame = grid.w * grid.h
    if len(comp) < frame * 0.05:
        raise PhotoQualityError(
            "Не получилось уверенно найти лицо: оно перекрыто, далеко или слабо освещено. "
            "Сделайте селфи анфас при дневном свете."
        )
    # Плоская «кожа» на весь кадр (потолок, стена) — не лицо.
    if len(comp) > frame * 0.55:
        tex = sum(grid.lap(i % grid.w, i // grid.w) for i in comp[:: max(1, len(comp) // 400)])
        tex /= max(1, len(comp[:: max(1, len(comp) // 400)]))
        if tex < 3.5:
            raise PhotoQualityError(
                "На фото не видно лица. Сделайте селфи анфас при дневном свете, "
                "без сильных фильтров и перекрытий."
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

_GLASSES_MSG = (
    "На фото видны очки — они закрывают зоны кожи и дают блики. "
    "Снимите очки и сделайте новое селфи анфас при дневном свете."
)


def _zone_stats(grid, bbox, fx0, fy0, fx1, fy1):
    """Средняя яркость, доля бликов и средняя резкость в доле рамки лица."""
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    xa, xb = x0 + int(fx0 * fw), x0 + int(fx1 * fw)
    ya, yb = y0 + int(fy0 * fh), y0 + int(fy1 * fh)
    xa, xb = max(x0, xa), min(x1, xb)
    ya, yb = max(y0, ya), min(y1, yb)
    if xb <= xa or yb <= ya:
        return 0.0, 0.0, 0.0, 0
    lumas, sharp, specular, n = 0.0, 0.0, 0, 0
    for y in range(ya, yb + 1):
        row = y * grid.w
        for x in range(xa, xb + 1):
            i = row + x
            # в зоне глаз кожа часто не проходит skin-маску (зрачки/оправа) —
            # считаем все пиксели, иначе очки «выпадают» из статистики
            l = grid.luma[i]
            lumas += l
            sharp += grid.lap(x, y)
            # блик на линзе: почти белый и ярче соседей
            if l >= 236:
                specular += 1
            n += 1
    if n < 20:
        return 0.0, 0.0, 0.0, n
    return lumas / n, sharp / n, specular / n, n


def _glasses_bridge_score(grid, bbox):
    """
    Тёмная перемычка оправы между глазами (переносица).
    Сравниваем горизонтальную полосу над переносицей с соседней кожей.
    """
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    # переносица / bridge: центр лица, чуть ниже бровей
    bx0 = x0 + int(0.42 * fw)
    bx1 = x0 + int(0.58 * fw)
    by0 = y0 + int(0.40 * fh)
    by1 = y0 + int(0.50 * fh)
    bridge = []
    cheeks = []
    for y in range(by0, by1 + 1):
        row = y * grid.w
        for x in range(bx0, bx1 + 1):
            bridge.append(grid.luma[row + x])
        for x in list(range(x0 + int(0.18 * fw), x0 + int(0.28 * fw))) + list(
            range(x0 + int(0.72 * fw), x0 + int(0.82 * fw))
        ):
            if x0 <= x <= x1:
                cheeks.append(grid.luma[row + x])
    if len(bridge) < 12 or len(cheeks) < 12:
        return 0.0
    bridge.sort()
    cheeks.sort()
    # берём тёмную четверть — оправа, не блик
    b_dark = sum(bridge[: max(1, len(bridge) // 4)]) / max(1, len(bridge) // 4)
    c_med = cheeks[len(cheeks) // 2]
    return max(0.0, c_med - b_dark)


def _glasses_frame_arcs(grid, bbox):
    """
    Тёмные дуги оправы вокруг глаз: много тёмных пикселей по контуру
    глазных прямоугольников при светлой середине (линза / блик).
    """
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    eyes = (
        (0.18, 0.36, 0.44, 0.54),
        (0.56, 0.36, 0.82, 0.54),
    )
    hits = 0
    for fx0, fy0, fx1, fy1 in eyes:
        xa, xb = x0 + int(fx0 * fw), x0 + int(fx1 * fw)
        ya, yb = y0 + int(fy0 * fh), y0 + int(fy1 * fh)
        w, h = max(1, xb - xa), max(1, yb - ya)
        border_dark = border_n = core_bright = core_n = 0
        for y in range(ya, yb + 1):
            row = y * grid.w
            ty = (y - ya) / h
            for x in range(xa, xb + 1):
                tx = (x - xa) / w
                on_border = tx < 0.14 or tx > 0.86 or ty < 0.18 or ty > 0.82
                l = grid.luma[row + x]
                if on_border:
                    border_n += 1
                    if l < 95:
                        border_dark += 1
                elif 0.28 < tx < 0.72 and 0.28 < ty < 0.72:
                    core_n += 1
                    if l > 200:
                        core_bright += 1
        if border_n < 20 or core_n < 10:
            continue
        if border_dark / border_n > 0.22 and core_bright / core_n > 0.08:
            hits += 1
    return hits


def _glasses_likely(grid, bbox, source_img=None, face_frac=None):
    """
    Эвристика «на лице очки»: блики на линзах + тёмная оправа/перемычка.
    Без блика или без оправы не отклоняем — иначе обычные глаза/брови
    дают ложные срабатывания на синтетических и реальных селфи.
    """
    eye_luma, eye_sharp, eye_spec, eye_n = _zone_stats(grid, bbox, 0.14, 0.34, 0.86, 0.56)
    cheek_luma, cheek_sharp, cheek_spec, cheek_n = _zone_stats(
        grid, bbox, 0.18, 0.56, 0.82, 0.74
    )
    if eye_n < 40 or cheek_n < 40:
        return False

    glare = eye_spec >= 0.006 and eye_spec > cheek_spec * 2.5 + 0.002
    strong_glare = eye_spec >= 0.012 and eye_spec > cheek_spec * 2.0 + 0.003
    bridge = _glasses_bridge_score(grid, bbox)
    arcs = _glasses_frame_arcs(grid, bbox)
    rim = arcs >= 2 or (arcs >= 1 and bridge >= 20)
    strong_rim = arcs >= 2 and bridge >= 16

    # Главное правило: блик на линзах + признаки оправы
    if (glare or strong_glare) and (rim or strong_rim or bridge >= 22):
        return True
    # Сильная оправа с двумя дугами и перемычкой даже при слабом блике
    if strong_rim and bridge >= 24 and eye_sharp > cheek_sharp * 1.3:
        return True
    return False


def _assert_no_glasses(grid, bbox, source_img=None, face_frac=None):
    if _glasses_likely(grid, bbox, source_img=source_img, face_frac=face_frac):
        raise PhotoQualityError(_GLASSES_MSG)


def _check_quality(grid, bbox, source_img=None, face_frac=None):
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
    # Сетка грубая: порог выше прежнего 1.1, чтобы отсекать заметно
    # «мягкие»/размытые селфи, которые раньше проходили анализ.
    if mean_sharp < 2.4:
        raise PhotoQualityError(
            "Фото размыто или сильно сглажено фильтром. "
            "Сделайте чёткий снимок без фильтров."
        )
    # Доп. проверка на исходном разрешении (variance of Laplacian).
    lap_var = _face_laplacian_variance(source_img, face_frac) if source_img and face_frac else None
    if lap_var is not None and lap_var < 55.0:
        raise PhotoQualityError(
            "Фото размыто или не в фокусе. "
            "Сделайте чёткий снимок анфас при дневном свете, без движения камеры."
        )
    quality = {"mean_luma": round(mean_luma, 1), "sharpness": round(mean_sharp, 2)}
    if lap_var is not None:
        quality["lap_var"] = round(lap_var, 1)
    return quality


def _face_laplacian_variance(img, face_frac):
    """Variance of Laplacian по кропу лица — классический тест на blur."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if not face_frac or img is None:
        return None
    w, h = img.size
    x0 = max(0, int(face_frac[0] * w))
    y0 = max(0, int(face_frac[1] * h))
    x1 = min(w, int(face_frac[2] * w))
    y1 = min(h, int(face_frac[3] * h))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    crop = img.crop((x0, y0, x1, y1)).convert("L")
    # Нормализуем размер, чтобы порог был стабилен для разных камер.
    target = 180
    cw, ch = crop.size
    scale = target / max(cw, ch)
    if scale < 1.0:
        crop = crop.resize((max(24, round(cw * scale)), max(24, round(ch * scale))))
    arr = np.asarray(crop, dtype=np.uint8)
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def _validate_face_framing(face_frac, grid, bbox):
    """
    Отклоняет кадры, где лицо обрезано краем или видна только половина.
    face_frac — доли кадра (Haar), bbox — пиксели сетки.
    """
    fx0, fy0, fx1, fy1 = face_frac
    fw, fh = fx1 - fx0, fy1 - fy0
    cx = (fx0 + fx1) / 2.0

    # Лицо слишком маленькое в кадре — нельзя оценить зоны.
    if fw < 0.18 or fh < 0.22 or (fw * fh) < 0.055:
        raise PhotoQualityError(
            "Лицо слишком далеко или мелко в кадре. "
            "Сделайте селфи ближе, анфас, чтобы лицо занимало большую часть кадра."
        )

    # Обрезка по бокам / сверху / снизу.
    side_hit = fx0 <= 0.03 or fx1 >= 0.97
    top_hit = fy0 <= 0.01
    bottom_hit = fy1 >= 0.99
    if side_hit or (top_hit and fh > 0.55) or (bottom_hit and fy0 > 0.25):
        raise PhotoQualityError(
            "Лицо обрезано краем кадра. "
            "Поместите лицо целиком в кадр анфас — без обрезанного подбородка или щеки."
        )
    # Сильный сдвиг в сторону — типичный «поллица».
    if cx < 0.30 or cx > 0.70:
        raise PhotoQualityError(
            "Лицо смещено и частично вне кадра. "
            "Сделайте селфи по центру, анфас, чтобы было видно всё лицо."
        )

    # Обе половины бокса должны содержать кожу (иначе видна одна щека).
    x0, y0, x1, y1 = bbox
    mid = (x0 + x1) // 2
    left = right = 0
    for y in range(y0, y1 + 1):
        row = y * grid.w
        for x in range(x0, mid + 1):
            if grid.skin[row + x]:
                left += 1
        for x in range(mid, x1 + 1):
            if grid.skin[row + x]:
                right += 1
    total = left + right
    if total < 30:
        raise PhotoQualityError(
            "На фото плохо различима кожа лица. "
            "Сделайте селфи анфас при дневном свете без сильных фильтров."
        )
    balance = min(left, right) / max(left, right)
    if balance < 0.32:
        raise PhotoQualityError(
            "На фото видно только часть лица. "
            "Сделайте селфи анфас по центру кадра, без обрезанной половины лица."
        )


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
    # Базлайн «средней кожи» считаем только по пикселям анатомических зон:
    # остальная часть рамки может содержать волосы, уши и край фона.
    face_pixels = [p for pts in regions.values() for p in pts]
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


def _bbox_skin_set(grid, bbox):
    """Все пиксели кожи (и сильной красноты) внутри рамки лица — для ring-проверок."""
    x0, y0, x1, y1 = bbox
    out = set()
    for y in range(y0, y1 + 1):
        row = y * grid.w
        for x in range(x0, x1 + 1):
            if grid.skin[row + x] or _is_reddened_skin(*grid.px[row + x]):
                out.add((x, y))
    return out


def _skin_ring_fraction(pts, skin_all, radius=2):
    """Доля соседних пикселей вокруг компоненты, которые тоже кожа."""
    pts_set = pts if isinstance(pts, set) else set(pts)
    ring = set()
    for x, y in pts_set:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n not in pts_set:
                    ring.add(n)
    if not ring:
        return 1.0
    return sum(1 for p in ring if p in skin_all) / len(ring)


def _local_skin_frac(x, y, skin_all, radius=2):
    neigh = [
        (x + dx, y + dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    ]
    return sum(1 for p in neigh if p in skin_all) / len(neigh)


def _pick_interior_centroid(pts, bbox, skin_all, score_fn=None, min_local=0.78, y_bias="cheek"):
    """
    Центроид маркера только по внутренним пикселям аномалии.
    Отсекает край силуэта/фон; для щёк смещает точку к центру лица.
    y_bias="cheek" — тянет к «яблоку» щеки; "none" — остаётся на самом тёмном ядре
    (нужно для пигментации, иначе маркер с линии роста волос уезжает на чистый лоб).
    Возвращает (cx, cy, bx0, by0, bx1, by1) или None.
    """
    if not pts:
        return None
    scored = []
    min_fy = 0.12 if y_bias == "none" else 0.14
    for x, y in pts:
        if _local_skin_frac(x, y, skin_all) < min_local:
            continue
        fx, fy = _face_frac(x, y, bbox)
        # край bbox / волосы / фон — не ставим маркер
        if fx < 0.18 or fx > 0.82 or fy < min_fy or fy > 0.90:
            continue
        base = score_fn((x, y)) if score_fn else 1.0
        # inward: дальше от наружного контура щёк
        inward = 1.0 - max(0.0, 0.22 - fx) * 4.0 - max(0.0, fx - 0.78) * 4.0
        inward *= 1.0 - abs(fx - 0.50) * 0.55
        if y_bias == "cheek":
            # mid-cheek: не уезжать к челюсти / вверх на лоб
            mid_y = 1.0 - max(0.0, fy - 0.72) * 2.8
            mid_y *= 1.0 - max(0.0, 0.48 - fy) * 1.5
        else:
            mid_y = 1.0
        scored.append((base * max(0.12, inward) * max(0.18, mid_y), x, y))
    if len(scored) < 3:
        # запасной проход: чуть мягче по локальной коже, но силуэт всё равно режем
        scored = []
        for x, y in pts:
            if _local_skin_frac(x, y, skin_all, radius=1) < 0.65:
                continue
            fx, fy = _face_frac(x, y, bbox)
            if fx < 0.20 or fx > 0.80 or fy > 0.88 or fy < min_fy:
                continue
            base = score_fn((x, y)) if score_fn else 1.0
            scored.append((base * (1.0 - abs(fx - 0.5)), x, y))
    if not scored:
        return None
    scored.sort(reverse=True)
    top = scored[: max(6, len(scored) // 5)]
    xs = [t[1] for t in top]
    ys = [t[2] for t in top]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    fx, fy = _face_frac(cx, cy, bbox)
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    # финальный soft-clamp внутрь лица
    if fx < 0.22:
        cx = x0 + 0.26 * fw
        fx = 0.26
    elif fx > 0.78:
        cx = x0 + 0.74 * fw
        fx = 0.74
    if y_bias == "cheek" and fy > 0.76:
        cy = y0 + 0.68 * fh
    # точка должна остаться на коже (после clamp)
    ix, iy = int(round(cx)), int(round(cy))
    if (ix, iy) not in skin_all and _local_skin_frac(ix, iy, skin_all, radius=1) < 0.5:
        # ближайший кандидат из top
        cx, cy = top[0][1], top[0][2]
    bx0, by0 = min(xs), min(ys)
    bx1, by1 = max(xs), max(ys)
    return cx, cy, bx0, by0, bx1, by1


def _local_ring_deficit(pts, grid, radius=5):
    """
    Локальный контраст: средняя яркость кольца вокруг пятна минус яркость пятна.
    Настоящая пигментация — островок темнее соседей; ровный лоб с тенью от света — нет.
    """
    if not pts:
        return 0.0
    pset = set(pts)
    w = grid.w
    h = grid.h
    inside = sum(grid.luma[y * w + x] for x, y in pts) / len(pts)
    ring = []
    r2 = radius * radius
    for x, y in pts:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > r2 or (dx == 0 and dy == 0):
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in pset:
                    continue
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if not grid.skin[ny * w + nx]:
                    continue
                ring.append(grid.luma[ny * w + nx])
    if len(ring) < 10:
        return 0.0
    return (sum(ring) / len(ring)) - inside


# ── Детекторы признаков ──────────────────────────────────────────────────────


def _detect_red(grid, bbox, regions, base):
    """Воспаления (компактные красные элементы) и разлитая краснота."""
    findings = []
    region_pts = {rid: set(pts) for rid, pts in regions.items()}
    allowed = set().union(*region_pts.values()) if region_pts else set()
    skin_all = _bbox_skin_set(grid, bbox)

    # Типичный красный тон самих губ (центр рта): чтобы отличать помаду и
    # уголки губ от настоящих воспалений рядом со ртом.
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)
    mx0, my0 = int(x0 + 0.38 * fw), int(y0 + 0.80 * fh)
    mx1, my1 = int(x0 + 0.62 * fw), int(y0 + 0.90 * fh)
    lip_rgs = sorted(
        grid.rg[yy * grid.w + xx]
        for yy in range(max(0, my0), min(grid.h, my1))
        for xx in range(max(0, mx0), min(grid.w, mx1))
    )
    lip_rg = lip_rgs[len(lip_rgs) * 3 // 4] if lip_rgs else None
    anomaly = {
        (x, y) for (x, y) in allowed
        if grid.rg[y * grid.w + x] - base["rg"] > 21
        and 60 < grid.luma[y * grid.w + x] < base["luma"] + 35
    }
    face_area = max(1, len(allowed))
    for pts in _components(anomaly, grid, min_area=3):
        area_frac_pre = len(pts) / face_area
        ring = _skin_ring_fraction(pts, skin_all)
        # компактные воспаления — мягче; разлитые пятна не должны обнимать край лица
        if area_frac_pre < 0.012 and ring < 0.55:
            continue
        if area_frac_pre >= 0.012 and ring < 0.78:
            continue
        geom = _pick_interior_centroid(
            pts, bbox, skin_all,
            score_fn=lambda p: grid.rg[p[1] * grid.w + p[0]] - base["rg"],
            min_local=0.70,
        )
        if not geom:
            cx, cy, bx0, by0, bx1, by1 = _comp_geometry(pts)
            fx0, fy0 = _face_frac(cx, cy, bbox)
            if fx0 < 0.18 or fx0 > 0.82 or _local_skin_frac(int(cx), int(cy), skin_all) < 0.6:
                continue
        else:
            cx, cy, bx0, by0, bx1, by1 = geom
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
        compact = comp_w * comp_h <= len(pts) * 3.2
        # губы/помада: очень красные участки в нижней трети лица — не кожа
        # компактные «прыщики» не отбрасываем — это как раз тестовые воспаления
        is_compact_spot = area_frac < 0.01 and compact
        if (not is_compact_spot) and fy > 0.60 and (
            excess > 80 or (excess > 50 and comp_w >= comp_h * 2.0)
        ):
            continue
        # уголки губ/края помады: красный элемент около рта с тоном как у губ
        comp_rg = sum(grid.rg[y * grid.w + x] for x, y in pts) / len(pts)
        near_mouth = 0.24 < fx < 0.76 and 0.66 < fy < 0.98
        if (
            (not is_compact_spot)
            and near_mouth
            and lip_rg is not None
            and abs(comp_rg - lip_rg) < 28
        ):
            continue
        strength = min(1.0, (excess - 21) / 34.0 + area_frac * 4.0)
        conf = min(0.95, 0.5 + (excess - 21) / 55.0 + min(0.12, area_frac * 8))
        ftype = "inflammation" if is_compact_spot else "redness"
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


def _zone_color_ratios(grid, pts):
    rr = gg = bb = 0
    for x, y in pts:
        r, g, b = grid.px[y * grid.w + x]
        rr += r
        gg += g
        bb += b
    return rr / max(1, gg), gg / max(1, bb)


def _detect_diffuse_redness(grid, bbox, regions, base):
    """
    Разлитая краснота щёк/носа по индексу r/g (устойчив к теням: свет
    масштабирует каналы одинаково). Тёплый баланс белого вычитается через g/b.
    Симметричная краснота обеих щёк — «сосудистая краснота» (не диагноз).
    """
    findings = []
    skin_all = _bbox_skin_set(grid, bbox)
    idx = {}
    for rid in ("left_cheek", "right_cheek", "nose"):
        pts = regions.get(rid) or []
        if len(pts) < 40:
            continue
        rg_ratio, gb_ratio = _zone_color_ratios(grid, pts)
        # компенсация тёплого света: желтизна (g/b) выше нейтральной ~1.15
        redness_index = rg_ratio - 0.6 * max(0.0, gb_ratio - 1.15)
        idx[rid] = (redness_index, pts)

    threshold = 1.30
    reds = {rid: v for rid, v in idx.items() if v[0] > threshold}
    symmetric = "left_cheek" in reds and "right_cheek" in reds
    cheek_mean = (
        (reds["left_cheek"][0] + reds["right_cheek"][0]) / 2 if symmetric else 0.0
    )
    for rid, (index, pts) in reds.items():
        # маркер — плотнейшие красные пиксели внутри зоны (не край/челюсть)
        def px_ratio(p):
            r, g, b = grid.px[p[1] * grid.w + p[0]]
            return r / max(1, g)

        geom = _pick_interior_centroid(pts, bbox, skin_all, score_fn=px_ratio)
        if not geom:
            continue
        cx, cy, bx0, by0, bx1, by1 = geom
        fx, fy = _face_frac(cx, cy, bbox)
        # для щёк дополнительно отсекаем наружный силуэт
        if "cheek" in rid and (fx < 0.18 or fx > 0.82 or fy > 0.78):
            continue
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (index - threshold) / 0.20)
        conf = min(0.93, 0.5 + (index - threshold) * 1.8 + (0.12 if symmetric else 0.0))
        rosacea = symmetric and cheek_mean > 1.32 and rid != "nose"
        findings.append({
            "type": "rosacea_like" if rosacea else "redness",
            "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": (
                "симметричная разлитая краснота щёк — сосудистая картина, не диагноз"
                if rosacea
                else "устойчивый красный подтон зоны независимо от освещения"
            ),
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    return findings


def _detect_nasolabial(grid, bbox, regions, base):
    """
    Носогубные складки: маркер на диагонали крыло носа → уголок рта.
    Показываем только при заметном гребне; вторая сторона — парно.
    """
    findings = []
    x0, y0, x1, y1 = bbox
    fw, fh = max(1, x1 - x0), max(1, y1 - y0)

    nose_pts = regions.get("nose") or []
    if nose_pts:
        nose_xs = [p[0] for p in nose_pts]
        nose_ys = [p[1] for p in nose_pts]
        nose_left = min(nose_xs)
        nose_right = max(nose_xs)
        mid_y = sum(nose_ys) / len(nose_ys)
        low = [p for p in nose_pts if p[1] >= mid_y]
        wing_y = sum(p[1] for p in low) / len(low) if low else (y0 + 0.62 * fh)
    else:
        nose_left = x0 + int(0.41 * fw)
        nose_right = x0 + int(0.59 * fw)
        wing_y = y0 + 0.62 * fh

    mouth_y = y0 + 0.80 * fh
    mouth_left = x0 + 0.30 * fw
    mouth_right = x0 + 0.70 * fw

    sides = (
        ("left_nasolabial", nose_left, mouth_left),
        ("right_nasolabial", nose_right, mouth_right),
    )

    scored_sides = []
    for rid, wing_x, mouth_x in sides:
        best = None  # (score, x, y, t)
        for ti in range(14, 28):  # t = 0.35 … 0.675 — середина и низ складки
            t = ti / 40.0
            px = wing_x + (mouth_x - wing_x) * t
            py = wing_y + (mouth_y - wing_y) * t
            x = int(round(px))
            y = int(round(py))
            if x <= 1 or y <= 1 or x >= grid.w - 2 or y >= grid.h - 2:
                continue
            local_best = None
            for ox in (-0.015, -0.008, 0.0, 0.008, 0.015):
                xx = int(round(x + ox * fw))
                if xx <= 1 or xx >= grid.w - 2:
                    continue
                gx, gy = grid.grad(xx, y)
                across = abs(gx) * 1.45 + abs(gy) * 0.3
                lap = grid.lap(xx, y)
                mid_pref = 1.0 - abs(t - 0.50) * 1.6
                score = (across + lap * 0.3) * max(0.4, mid_pref)
                if local_best is None or score > local_best[0]:
                    local_best = (score, xx, y, t)
            if local_best and (best is None or local_best[0] > best[0]):
                best = local_best
        if best is None:
            continue
        scored_sides.append((best[0], rid, wing_x, mouth_x, best))

    if not scored_sides:
        return findings

    # нужен хотя бы один заметный гребень складки
    strong = any(s[0] >= 4.0 for s in scored_sides)
    peak_floor = 3.2 if strong else 4.5
    kept = [s for s in scored_sides if s[0] >= peak_floor]
    if not kept and strong:
        kept = [max(scored_sides, key=lambda s: s[0])]
    if not kept:
        return findings

    for peak, rid, wing_x, mouth_x, best in kept:
        t_anchor = 0.50
        ax = wing_x + (mouth_x - wing_x) * t_anchor
        ay = wing_y + (mouth_y - wing_y) * t_anchor
        cx = 0.25 * best[1] + 0.75 * ax
        cy = 0.25 * best[2] + 0.75 * ay
        strength = min(1.0, max(0.0, (peak - 3.5) / 7.0))
        conf = min(0.92, 0.48 + (peak - 3.5) / 14.0)
        if conf < CONF_FLOOR or strength < 0.28:
            continue
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        ix, iy = int(cx), int(cy)
        findings.append({
            "type": "nasolabial", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "линия-складка от крыла носа к уголку рта",
            "geom": _to_pct(grid, cx, cy, ix, iy, ix, iy),
        })
    # парность: одна уверенная сторона → зеркало (с conf чуть выше порога)
    if len(findings) == 1:
        found = findings[0]
        other = (
            "left_nasolabial" if found["region"] == "right_nasolabial" else "right_nasolabial"
        )
        wing_x, mouth_x = (
            (nose_left, mouth_left) if other == "left_nasolabial" else (nose_right, mouth_right)
        )
        t = 0.50
        cx = wing_x + (mouth_x - wing_x) * t
        cy = wing_y + (mouth_y - wing_y) * t
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[other]
        ix, iy = int(cx), int(cy)
        findings.append({
            "type": "nasolabial", "region": other, "region_label": rlabel,
            "strength": max(0.32, found["strength"] * 0.75),
            "confidence": round(max(CONF_FLOOR + 0.02, found["confidence"] * 0.9), 2),
            "evidence": "линия-складка от крыла носа к уголку рта",
            "geom": _to_pct(grid, cx, cy, ix, iy, ix, iy),
        })
    return findings


def _eye_pair_is_plausible(left, right, bbox):
    """Анатомическая проверка пары глаз: один уровень и разумное расстояние."""
    if not left or not right:
        return False
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    lfx, lfy = _face_frac(left[0], left[1], bbox)
    rfx, rfy = _face_frac(right[0], right[1], bbox)
    if not (0.12 <= lfx <= 0.46 and 0.54 <= rfx <= 0.88):
        return False
    if abs(lfy - rfy) > 0.06:
        return False
    if not (0.26 <= (rfx - lfx) <= 0.62):
        return False
    if not (0.26 <= lfy <= 0.54 and 0.26 <= rfy <= 0.54):
        return False
    return True


def _eyes_from_cascade(grid, bbox, source_img):
    """
    Зрачки каскадами OpenCV на вырезе лица. Несколько каскадов и несколько
    режимов чувствительности: берём первую пару, прошедшую анатомическую проверку.
    Возвращает {"left": (cx, cy, eye_h), "right": (...)} в координатах сетки.
    """
    if source_img is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        x0, y0, x1, y1 = bbox
        sw, sh = source_img.size
        sx0 = max(0, int(round(x0 / max(1, grid.w - 1) * (sw - 1))))
        sy0 = max(0, int(round(y0 / max(1, grid.h - 1) * (sh - 1))))
        sx1 = min(sw, int(round(x1 / max(1, grid.w - 1) * (sw - 1))) + 1)
        sy1 = min(sh, int(round(y1 / max(1, grid.h - 1) * (sh - 1))) + 1)
        crop = source_img.crop((sx0, sy0, sx1, sy1))
        cw, ch = crop.size
        if cw < 40 or ch < 40:
            return None
        up = 1.0
        if cw < 360:
            up = 360.0 / cw
        elif cw > 900:
            up = 900.0 / cw
        if up != 1.0:
            crop = crop.resize((max(1, int(cw * up)), max(1, int(ch * up))))
        gray = cv2.equalizeHist(np.asarray(crop.convert("L")))
        gh_px, gw_px = gray.shape
        min_e = max(10, int(gw_px * 0.06))

        def to_grid(ex, ey, ew, eh):
            cx_s = sx0 + (ex + ew / 2.0) / up
            cy_s = sy0 + (ey + eh / 2.0) / up
            cx = cx_s / max(1, sw - 1) * (grid.w - 1)
            cy = cy_s / max(1, sh - 1) * (grid.h - 1)
            eh_grid = (eh / up) / max(1, sh - 1) * (grid.h - 1)
            return cx, cy, eh_grid

        pool = []
        cascades = (
            "haarcascade_eye.xml",
            "haarcascade_eye_tree_eyeglasses.xml",
            "haarcascade_lefteye_2splits.xml",
            "haarcascade_righteye_2splits.xml",
        )
        for xml in cascades:
            clf = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
            if clf.empty():
                continue
            for scale, neighbors in ((1.05, 3), (1.1, 4), (1.2, 5)):
                try:
                    found = clf.detectMultiScale(
                        gray, scale, neighbors, minSize=(min_e, min_e)
                    )
                except Exception:
                    continue
                for ex, ey, ew, eh in found if found is not None else []:
                    rel_y = (ey + eh * 0.5) / max(1, gh_px)
                    rel_x = (ex + ew * 0.5) / max(1, gw_px)
                    if not (0.28 <= rel_y <= 0.58):
                        continue
                    if not (0.10 <= rel_x <= 0.90):
                        continue
                    pool.append((rel_x, *to_grid(ex, ey, ew, eh)))

        if not pool:
            return None

        def consensus(cands):
            """Медиана по кластеру совпадающих находок — устойчивее одиночной."""
            if not cands:
                return None
            ys = sorted(c[2] for c in cands)
            med_y = ys[len(ys) // 2]
            near = [c for c in cands if abs(c[2] - med_y) <= 0.05 * max(1, y1 - y0)]
            near = near or cands
            return (
                sum(c[1] for c in near) / len(near),
                sum(c[2] for c in near) / len(near),
                sum(c[3] for c in near) / len(near),
            )

        left = consensus([c for c in pool if c[0] < 0.50])
        right = consensus([c for c in pool if c[0] >= 0.50])
        if _eye_pair_is_plausible(left, right, bbox):
            return {"left": left, "right": right}
        out = {}
        if left:
            out["left"] = left
        if right:
            out["right"] = right
        return out or None
    except Exception:
        return None


def _eyes_from_dark_blobs(grid, bbox):
    """
    Резерв: тёмные пятна в полосе глазниц.
    Брови тоже тёмные, поэтому берём НИЖНИЙ тёмный кластер каждой стороны.
    """
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    eyes = {}
    for side, fx0, fx1 in (("left", 0.18, 0.46), ("right", 0.54, 0.82)):
        rows = []
        y_lo = max(y0, y0 + int(0.28 * fh))
        y_hi = min(y1, y0 + int(0.54 * fh))
        x_lo = max(x0, x0 + int(fx0 * fw))
        x_hi = min(x1, x0 + int(fx1 * fw))
        if y_hi <= y_lo or x_hi <= x_lo:
            continue
        for y in range(y_lo, y_hi + 1):
            dark = [
                (grid.luma[y * grid.w + x], x)
                for x in range(x_lo, x_hi + 1)
                if 16 <= grid.luma[y * grid.w + x] <= 128
            ]
            if len(dark) >= 2:
                rows.append((y, sum(d[0] for d in dark) / len(dark), len(dark), dark))
        if not rows:
            continue
        # группируем подряд идущие «тёмные» строки в кластеры
        clusters = []
        cur = [rows[0]]
        for r in rows[1:]:
            if r[0] - cur[-1][0] <= 1:
                cur.append(r)
            else:
                clusters.append(cur)
                cur = [r]
        clusters.append(cur)
        clusters = [c for c in clusters if len(c) >= 1]
        if not clusters:
            continue
        # бровь — верхний кластер; глаз — следующий ниже (если есть)
        eye_cluster = clusters[-1] if len(clusters) == 1 else clusters[1] if len(clusters) >= 2 else clusters[0]
        if len(clusters) >= 2:
            # выбираем самый «тёмный» из нижних кластеров
            lower = clusters[1:]
            eye_cluster = min(lower, key=lambda c: sum(r[1] for r in c) / len(c))
        pts = [(lum, x, r[0]) for r in eye_cluster for lum, x in r[3]]
        if len(pts) < 6:
            continue
        pts.sort(key=lambda t: t[0])
        core = pts[: max(5, len(pts) // 6)]
        eyes[side] = (
            sum(c[1] for c in core) / len(core),
            sum(c[2] for c in core) / len(core),
        )
    return eyes


def _find_eye_centers(grid, bbox, source_img=None):
    """
    Центры зрачков в координатах сетки — всегда обе стороны.
    Каскады → тёмные кластеры глазниц → анатомическая модель.
    Значение: (cx, cy, высота глаза в пикселях сетки).
    """
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    default_h = 0.075 * fh

    def with_h(pt):
        if pt is None:
            return None
        return (pt[0], pt[1], pt[2] if len(pt) > 2 else default_h)

    def geometric():
        return {
            "left": (x0 + 0.325 * fw, y0 + 0.455 * fh, default_h),
            "right": (x0 + 0.675 * fw, y0 + 0.455 * fh, default_h),
        }

    cascade = _eyes_from_cascade(grid, bbox, source_img) or {}
    left = with_h(cascade.get("left"))
    right = with_h(cascade.get("right"))

    if not _eye_pair_is_plausible(left, right, bbox):
        blobs = {k: with_h(v) for k, v in _eyes_from_dark_blobs(grid, bbox).items()}
        merged_left = left or blobs.get("left")
        merged_right = right or blobs.get("right")
        if _eye_pair_is_plausible(merged_left, merged_right, bbox):
            left, right = merged_left, merged_right
        elif _eye_pair_is_plausible(blobs.get("left"), blobs.get("right"), bbox):
            left, right = blobs["left"], blobs["right"]
        else:
            single = None
            for side, cand in (
                ("left", left), ("right", right),
                ("left", blobs.get("left")), ("right", blobs.get("right")),
            ):
                if not cand:
                    continue
                fx, fy = _face_frac(cand[0], cand[1], bbox)
                on_side = (side == "left" and 0.14 <= fx <= 0.46) or (
                    side == "right" and 0.54 <= fx <= 0.86
                )
                if on_side and 0.30 <= fy <= 0.56:
                    single = (side, cand)
                    break
            if single:
                side, cand = single
                fx, _fy = _face_frac(cand[0], cand[1], bbox)
                mirror = (x0 + (1.0 - fx) * fw, cand[1], cand[2])
                if side == "left":
                    left, right = cand, mirror
                else:
                    right, left = cand, mirror
            else:
                g = geometric()
                left, right = g["left"], g["right"]

    # селфи анфас: зрачки на одном уровне
    eye_y = (left[1] + right[1]) / 2.0
    eye_h = max(0.045 * fh, min(0.11 * fh, (left[2] + right[2]) / 2.0))
    eyes = {"left": (left[0], eye_y, eye_h), "right": (right[0], eye_y, eye_h)}
    if not _eye_pair_is_plausible(eyes["left"], eyes["right"], bbox):
        eyes = geometric()
    return eyes


# Небольшие смещения по типам, чтобы маркеры разных признаков
# не ложились друг на друга ровно в одной точке.
# (сдвиг по X к внутреннему углу, множитель отступа вниз от зрачка)
# (сдвиг по X к внутреннему углу / наружу, доп. доля высоты лица вниз от базы)
_EYE_MARKER_OFFSET = {
    "dark_circles": (0.010, 0.000),
    "tired_eyes": (-0.022, 0.008),
    "puffiness": (0.000, 0.012),
    "wrinkles": (-0.020, -0.008),  # ближе к нижнему веку
    "dryness": (0.024, 0.006),
    "pigmentation": (-0.030, 0.010),
}


def _anchor_under_eye(cx, cy, bbox, rid, eyes=None, ftype=None):
    """
    Якорь в подглазье: сразу под нижним веком, не на зрачке и не на щеке.
    База — от найденного зрачка; абсолютный clamp только как страховка.
    """
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    side = "left" if "left" in (rid or "") else "right"
    eye = (eyes or {}).get(side)
    dx_frac, dy_extra = _EYE_MARKER_OFFSET.get(ftype or "", (0.0, 0.004))

    if eye:
        ex, ey = eye[0], eye[1]
        eye_h = eye[2] if len(eye) > 2 else 0.075 * fh
        # сразу под веком: ~0.55–0.9 высоты глаза, не больше ~9% лица
        drop = min(0.095 * fh, max(0.068 * fh, eye_h * 0.90))
        cy = ey + drop + dy_extra * fh
        # относительно зрачка: не на веке, не на середине щеки
        cy = min(max(cy, ey + 0.060 * fh), ey + 0.105 * fh)
        # абсолютная страховка: ниже зоны зрачка (_EXCLUDE до ~0.47), выше щеки
        cy = min(max(cy, y0 + 0.485 * fh), y0 + 0.545 * fh)
        dx = dx_frac * fw if side == "left" else -dx_frac * fw
        cx = max(x0 + 0.12 * fw, min(x1 - 0.12 * fw, ex + dx))
        return cx, cy

    cy = y0 + max(0.50, min(0.55, (cy - y0) / fh)) * fh
    if side == "left":
        cx = x0 + max(0.24, min(0.40, (cx - x0) / fw)) * fw
        cx = 0.35 * cx + 0.65 * (x0 + 0.32 * fw)
    else:
        cx = x0 + max(0.60, min(0.76, (cx - x0) / fw)) * fw
        cx = 0.35 * cx + 0.65 * (x0 + 0.68 * fw)
    return cx, cy


def _geom_hits_eye(geom, bbox, eyes, grid):
    """True, если маркер слишком близко к зрачку (эффект «очков»)."""
    if not eyes:
        return False
    cx = geom["x"] / 100.0 * grid.w
    cy = geom["y"] / 100.0 * grid.h
    fh = max(1, bbox[3] - bbox[1])
    fw = max(1, bbox[2] - bbox[0])
    for eye in eyes.values():
        ex, ey = eye[0], eye[1]
        eye_h = eye[2] if len(eye) > 2 else 0.075 * fh
        min_dist = max(0.075 * fh, eye_h * 0.95)
        if (cx - ex) ** 2 + (cy - ey) ** 2 < min_dist ** 2:
            return True
        # строго выше или на уровне глаза в его колонке — это «очки»
        if abs(cx - ex) < 0.09 * fw and cy < ey + max(0.07 * fh, eye_h * 0.8):
            return True
    return False


def _geom_hits_forbidden(grid, geom, bbox, eyes=None):
    """True, если центр маркера попал в глаз/рот/линию роста."""
    cx = geom["x"] / 100.0 * grid.w
    cy = geom["y"] / 100.0 * grid.h
    fx, fy = _face_frac(cx, cy, bbox)
    if any(_in_rect(fx, fy, r) for r in _EXCLUDE):
        return True
    if fy < 0.16:
        return True
    if _geom_hits_eye(geom, bbox, eyes, grid):
        return True
    return False


_EYE_ALWAYS_PAIRED = ("dark_circles", "tired_eyes", "puffiness")


def _is_eye_finding(f):
    rid = f.get("region") or ""
    ftype = f.get("type") or ""
    return (
        ftype in _EYE_ALWAYS_PAIRED
        or "under_eye" in rid
        or "crow_feet" in rid
    )


def _pair_eye_findings(grid, bbox, findings, eyes):
    """
    Всё, что относится к глазам, показываем симметрично: два маркера,
    по одному под каждым глазом. Недостающую сторону достраиваем зеркально.
    """
    labels = {r[0]: r[1] for r in _REGIONS}
    out = list(findings)
    by_type = {}
    for f in findings:
        if _is_eye_finding(f):
            by_type.setdefault(f.get("type"), []).append(f)

    for ftype, items in by_type.items():
        sides = set()
        for f in items:
            rid = f.get("region") or ""
            if "left" in rid:
                sides.add("left")
            elif "right" in rid:
                sides.add("right")
            else:
                fx = _face_frac(f["geom"]["x"] / 100.0 * grid.w,
                                f["geom"]["y"] / 100.0 * grid.h, bbox)[0]
                sides.add("left" if fx < 0.5 else "right")
        missing = {"left", "right"} - sides
        if not missing or not items:
            continue
        src = max(items, key=lambda f: f.get("confidence", 0) + f.get("strength", 0))
        for side in missing:
            if "crow_feet" in (src.get("region") or ""):
                rid = f"{side}_crow_feet"
                cx, cy = _anchor_crow_feet(bbox, side, eyes=eyes)
                rlabel = (
                    "У внешнего угла глаза слева" if side == "left"
                    else "У внешнего угла глаза справа"
                )
            else:
                rid = f"{side}_under_eye"
                eye = (eyes or {}).get(side) or (0, 0)
                cx, cy = _anchor_under_eye(
                    eye[0], eye[1], bbox, rid, eyes=eyes, ftype=ftype
                )
                rlabel = labels.get(rid, src.get("region_label"))
            clone = {
                **src,
                "region": rid,
                "region_label": rlabel,
                "geom": _to_pct(
                    grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2
                ),
                "confidence": max(0.0, src.get("confidence", 0.0) - 0.02),
                "strength": src.get("strength", 0.0),
            }
            out.append(clone)
    return out


def _sanitize_findings_markers(grid, bbox, findings, eyes=None):
    """Починить или отбросить маркеры, попавшие на глаза/волосы."""
    out = []
    x0, y0, x1, y1 = bbox
    fh = max(1, y1 - y0)
    eyes = eyes or {}
    for f in findings:
        geom = f.get("geom")
        if not geom:
            continue
        rid = f.get("region") or ""
        ftype = f.get("type") or ""
        is_crow = "crow_feet" in rid
        eye_related = (
            "under_eye" in rid
            or ftype in ("dark_circles", "tired_eyes", "puffiness")
            or (ftype == "wrinkles" and "under_eye" in rid)
        ) and not is_crow
        cx = geom["x"] / 100.0 * grid.w
        cy = geom["y"] / 100.0 * grid.h
        fx, fy = _face_frac(cx, cy, bbox)
        forbidden = _geom_hits_forbidden(grid, geom, bbox, eyes=eyes)
        if is_crow:
            side = "left" if "left" in rid else "right"
            cx, cy = _anchor_crow_feet(bbox, side, eyes=eyes)
            f = {
                **f,
                "region": f"{side}_crow_feet",
                "region_label": (
                    "У внешнего угла глаза слева" if side == "left"
                    else "У внешнего угла глаза справа"
                ),
                "geom": _to_pct(grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2),
            }
            out.append(f)
            continue
        if eye_related or (forbidden and ftype in ("dark_circles", "tired_eyes", "puffiness")):
            side = rid if "under_eye" in rid else (
                "left_under_eye" if fx < 0.5 else "right_under_eye"
            )
            if "left" not in side and "right" not in side:
                side = "left_under_eye" if fx < 0.5 else "right_under_eye"
            cx, cy = _anchor_under_eye(cx, cy, bbox, side, eyes=eyes, ftype=ftype)
            f = {
                **f,
                "region": side if "under_eye" in side else f.get("region"),
                "region_label": dict((r[0], r[1]) for r in _REGIONS).get(
                    side, f.get("region_label")
                ) if eye_related else f.get("region_label"),
                "geom": _to_pct(grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2),
            }
            if _geom_hits_forbidden(grid, f["geom"], bbox, eyes=eyes):
                side_key = "left" if "left" in side else "right"
                if side_key in eyes:
                    cx, cy = _anchor_under_eye(
                        eyes[side_key][0], eyes[side_key][1], bbox, side,
                        eyes=eyes, ftype=ftype,
                    )
                else:
                    cy = y0 + 0.58 * fh
                    cx = x0 + (0.32 if "left" in side else 0.68) * max(1, x1 - x0)
                f = {
                    **f,
                    "geom": _to_pct(grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2),
                }
        elif forbidden:
            continue
        cy2 = f["geom"]["y"] / 100.0 * grid.h
        if f.get("type") in ("pores", "pigmentation") and _face_frac(
            f["geom"]["x"] / 100.0 * grid.w, cy2, bbox
        )[1] < 0.20:
            continue
        if eye_related and _geom_hits_eye(f["geom"], bbox, eyes, grid):
            side_key = "left" if "left" in (f.get("region") or "") else "right"
            if side_key in eyes:
                cx, cy = _anchor_under_eye(
                    eyes[side_key][0], eyes[side_key][1], bbox,
                    f.get("region") or side_key, eyes=eyes, ftype=f.get("type"),
                )
                f = {
                    **f,
                    "geom": _to_pct(grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2),
                }
        out.append(f)
    return out


def _detect_dark_circles(grid, bbox, regions, base):
    findings = []
    raw_sides = {}
    x0, y0, x1, y1 = bbox
    for rid in ("left_under_eye", "right_under_eye"):
        pts = regions.get(rid) or []
        if len(pts) < 18:
            continue
        # только слёзная борозда / середина подглазья (не веко, не щека)
        filtered = []
        for x, y in pts:
            fx, fy = _face_frac(x, y, bbox)
            if fy < 0.49 or fy > 0.59:
                continue
            if any(_in_rect(fx, fy, r) for r in _EXCLUDE):
                continue
            filtered.append((x, y))
        if len(filtered) < 12:
            # мягкий запас: нижние 70% зоны under_eye
            filtered = [
                (x, y) for x, y in pts
                if _face_frac(x, y, bbox)[1] >= 0.49
                and not any(_in_rect(*_face_frac(x, y, bbox), r) for r in _EXCLUDE)
            ] or pts
        dark = [(x, y) for x, y in filtered if base["luma"] - grid.luma[y * grid.w + x] > 16]
        frac = len(dark) / max(1, len(filtered))
        raw_sides[rid] = (frac, dark, filtered)

    strong = any(v[0] >= 0.30 for v in raw_sides.values())
    frac_floor = 0.22 if strong else 0.34

    for rid, (frac, dark, filtered) in raw_sides.items():
        if frac < frac_floor or len(dark) < 6:
            continue
        deficit = sum(base["luma"] - grid.luma[y * grid.w + x] for x, y in dark) / len(dark)
        if deficit > 70:
            continue
        dark_sorted = sorted(
            dark, key=lambda p: base["luma"] - grid.luma[p[1] * grid.w + p[0]], reverse=True
        )
        core = dark_sorted[: max(6, len(dark_sorted) // 3)]
        cx = sum(p[0] for p in core) / len(core)
        cy = sum(p[1] for p in core) / len(core)
        cx, cy = _anchor_under_eye(cx, cy, bbox, rid)
        bx0, by0 = min(p[0] for p in core), min(p[1] for p in core)
        bx1, by1 = max(p[0] for p in core), max(p[1] for p in core)
        strength = min(1.0, (deficit - 16) / 32.0 + (frac - frac_floor) * 0.9)
        conf = min(0.92, 0.50 + frac * 0.35 + deficit / 150.0)
        if strong and frac < 0.38:
            conf = min(conf, 0.80)
        label = dict((r[0], r[1]) for r in _REGIONS)[rid]
        findings.append({
            "type": "dark_circles", "region": rid, "region_label": label,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "область под глазом заметно темнее среднего тона кожи",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })

    # Всегда пара: если одна сторона есть — зеркалим вторую
    if len(findings) == 1:
        found = findings[0]
        other = "left_under_eye" if found["region"] == "right_under_eye" else "right_under_eye"
        pts = regions.get(other) or []
        pts = [
            p for p in pts
            if 0.49 <= _face_frac(p[0], p[1], bbox)[1] <= 0.59
        ] or list(pts)
        if len(pts) >= 8:
            ranked = sorted(
                pts, key=lambda p: base["luma"] - grid.luma[p[1] * grid.w + p[0]], reverse=True
            )
            core = ranked[: max(5, len(ranked) // 4)]
            cx = sum(p[0] for p in core) / len(core)
            cy = sum(p[1] for p in core) / len(core)
            bx0, by0 = min(p[0] for p in core), min(p[1] for p in core)
            bx1, by1 = max(p[0] for p in core), max(p[1] for p in core)
        else:
            g = found["geom"]
            fw = max(1, x1 - x0)
            cx0 = g["x"] / 100.0 * grid.w
            cy0 = g["y"] / 100.0 * grid.h
            fx, fy = _face_frac(cx0, cy0, bbox)
            cx = x0 + (1.0 - fx) * fw
            cy = y0 + fy * max(1, y1 - y0)
            bx0 = by0 = int(cx)
            bx1 = by1 = int(cy)
        cx, cy = _anchor_under_eye(cx, cy, bbox, other)
        label = dict((r[0], r[1]) for r in _REGIONS)[other]
        findings.append({
            "type": "dark_circles", "region": other, "region_label": label,
            "strength": max(0.35, found["strength"] * 0.7),
            "confidence": round(min(0.82, found["confidence"] * 0.88), 2),
            "evidence": "область под глазом заметно темнее среднего тона кожи",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    return findings


def _detect_pigmentation(grid, bbox, regions, base):
    """
    Коричневатые/более тёмные компактные пятна (веснушки, постакне, солнечные).
    Сравниваем с локальной базой зоны и кольцом вокруг пятна — иначе ровный лоб
    при верхнем свете ошибочно помечается как пигментация.
    """
    findings = []
    # щёки — основной поиск; лоб только при явном локальном островке
    zone_ids = ("left_cheek", "right_cheek", "forehead")
    region_luma = {}
    for rid in zone_ids:
        pts = regions.get(rid) or []
        if len(pts) >= 24:
            region_luma[rid] = sum(grid.luma[y * grid.w + x] for x, y in pts) / len(pts)
    allowed = set()
    pixel_zone = {}
    for rid in zone_ids:
        for p in regions.get(rid) or []:
            allowed.add(p)
            pixel_zone[p] = rid
    skin_all = _bbox_skin_set(grid, bbox)
    anomaly = set()
    for x, y in allowed:
        i = y * grid.w + x
        p = grid.px[i]
        fx, fy = _face_frac(x, y, bbox)
        rid0 = pixel_zone.get((x, y))
        # линия роста волос / верх лба — частые ложные «пятна» от тени
        if rid0 == "forehead" and (fy < 0.16 or fy > 0.34):
            continue
        # коричневый / тёплый подтон: R≥G, заметный отрыв от синего
        brownish = (
            p[0] >= p[1] - 6
            and p[1] >= p[2] - 10
            and (p[0] - p[2]) > 12
        )
        local_base = region_luma.get(rid0, base["luma"])
        deficit_px = local_base - grid.luma[i]
        # на лбу нужен более сильный отрыв от собственной зоны, не от щёк
        min_def = 26 if rid0 == "forehead" else 18
        max_def = 70 if rid0 == "forehead" else 72
        # тёплые пигментные пятна чуть краснее базы — допускаем небольшой rg-offset
        if min_def < deficit_px < max_def and brownish and grid.rg[i] - base["rg"] < 22:
            anomaly.add((x, y))
    face_area = max(1, len(skin_all))
    for pts in _components(anomaly, grid, min_area=4):
        area_frac = len(pts) / face_area
        # кластеры веснушек на лбу могут быть крупнее одиночного пятна
        if area_frac > 0.035 or area_frac < 0.00025:
            continue
        _, _, bx0, by0, bx1, by1 = _comp_geometry(pts)
        bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
        if bw * bh > len(pts) * 3.4:
            continue
        # почти компактное пятно, не длинная тень-полоса
        if max(bw, bh) > min(bw, bh) * 3.0:
            continue
        if _skin_ring_fraction(pts, skin_all) < 0.78:
            continue
        local_contrast = _local_ring_deficit(pts, grid, radius=5)
        # без локального островка (темнее соседей) — это освещение, не пигмент
        rid_guess, _ = _region_of(*_face_frac(
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            bbox,
        ))
        min_local = 20.0 if rid_guess == "forehead" else 12.0
        if local_contrast < min_local:
            continue
        geom = _pick_interior_centroid(
            pts, bbox, skin_all,
            score_fn=lambda p: (
                region_luma.get(pixel_zone.get(p), base["luma"])
                - grid.luma[p[1] * grid.w + p[0]]
            ),
            min_local=0.72,
            y_bias="none",
        )
        if not geom:
            continue
        cx, cy, bx0, by0, bx1, by1 = geom
        fx, fy = _face_frac(cx, cy, bbox)
        rid, rlabel = _region_of(fx, fy)
        if rid not in zone_ids:
            continue
        # лоб: только середина зоны, без линии волос
        if rid == "forehead" and (fy < 0.16 or fy > 0.34 or fx < 0.28 or fx > 0.72):
            continue
        # щеки: середина «яблока», не низ и не край
        if "cheek" in rid and (fx < 0.24 or fx > 0.76 or fy < 0.48 or fy > 0.70):
            continue
        # не путать тень носогубной складки с пигментом
        if 0.58 <= fy <= 0.74 and abs(fx - 0.5) < 0.18:
            continue
        local_base = region_luma.get(rid, base["luma"])
        deficit = sum(local_base - grid.luma[y * grid.w + x] for x, y in pts) / len(pts)
        if rid == "forehead" and (deficit < 28 or local_contrast < 22):
            continue
        if deficit < 20:
            continue
        strength = min(1.0, (deficit - 16) / 32.0 + min(0.25, area_frac * 12))
        conf = min(0.92, 0.52 + (deficit - 16) / 48.0 + min(0.12, area_frac * 18))
        # локальный контраст повышает уверенность; ровный градиент — понижает
        conf = min(0.92, conf + min(0.10, (local_contrast - 12) / 80.0))
        if rid == "forehead":
            conf *= 0.88  # лоб чаще ловит свет/волосы — требуем сильнее
        if conf < CONF_FLOOR:
            continue
        if rid == "forehead" and conf < CONF_FLOOR + 0.08:
            continue
        findings.append({
            "type": "pigmentation", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "компактный участок темнее окружающей кожи, коричневатый оттенок",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    # если есть пигмент на щеках — слабые «пятна» на лбу отбрасываем
    cheeks = [f for f in findings if "cheek" in f["region"]]
    if cheeks:
        best_cheek = max(f["confidence"] for f in cheeks)
        findings = [
            f for f in findings
            if f["region"] != "forehead" or f["confidence"] >= best_cheek + 0.06
        ]
    return findings[:4]


def _is_hair_like_pixel(grid, x, y, skin_luma_ref):
    """
    Волос/чёлка: темнее кожи + направленный микроконтраст (нить),
    а не изотропная «ямка» поры.
    """
    w = grid.w
    if x <= 1 or y <= 1 or x >= w - 2 or y >= grid.h - 2:
        return True
    i = y * w + x
    L = grid.luma[i]
    # заметно темнее типичной кожи зоны
    if L > skin_luma_ref - 16:
        return False
    gx, gy = grid.grad(x, y)
    gmax, gmin = max(gx, gy), min(gx, gy)
    aniso = gmax / (gmin + 1.6)
    lap = grid.lap(x, y)
    # тёмная направленная структура = прядь, не пора
    if L < skin_luma_ref - 28 and aniso >= 1.85 and lap >= 6.0:
        return True
    if aniso >= 2.4 and lap >= 9.0 and L < skin_luma_ref - 18:
        return True
    return False


def _filter_hair_pixels(grid, pts):
    """Убирает пиксели волос; возвращает (очищенные_pts, доля_волос)."""
    if len(pts) < 10:
        return pts, 0.0
    lumas = sorted(grid.luma[y * grid.w + x] for x, y in pts)
    ref = lumas[len(lumas) // 2]
    clean = [(x, y) for x, y in pts if not _is_hair_like_pixel(grid, x, y, ref)]
    hair_frac = 1.0 - (len(clean) / max(1, len(pts)))
    return clean, hair_frac


def _detect_texture(grid, bbox, regions, base):
    """
    Расширенные поры / неровная текстура по микроконтрасту зоны.
    Волосы и линия роста никогда не считаются порами.
    """
    findings = []
    skin_all = _bbox_skin_set(grid, bbox)
    # Лоб намеренно исключён: чёлка/линия роста дают ложный микроконтраст «пор».
    for rid in ("nose", "left_cheek", "right_cheek", "chin"):
        pts = regions.get(rid) or []
        if len(pts) < 40:
            continue
        # глубокая эрозия зоны: микроконтраст меряем вдали от границ
        pset = set(pts)
        pts = [
            (x, y) for x, y in pts
            if all((x + dx, y + dy) in pset
                   for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2)
                   if abs(dx) + abs(dy) <= 2)
        ]
        if len(pts) < 40:
            continue
        pts, hair_frac = _filter_hair_pixels(grid, pts)
        # зона сильно засорена волосами — не анализируем как поры
        if hair_frac > 0.22 or len(pts) < 40:
            continue
        if _skin_ring_fraction(pts, skin_all, radius=3) < 0.82:
            continue
        laps = [(grid.lap(x, y), x, y) for x, y in pts]
        tex = sum(l[0] for l in laps) / len(laps)
        ratio = tex / max(0.5, base["tex"])
        if tex < 7.5 or ratio < 1.3:
            continue
        geom = _pick_interior_centroid(
            pts, bbox, skin_all,
            score_fn=lambda p: grid.lap(p[0], p[1]),
        )
        if not geom:
            continue
        cx, cy, bx0, by0, bx1, by1 = geom
        fx, fy = _face_frac(cx, cy, bbox)
        # линия роста / виски / верх кадра — не поры
        if fy < 0.22 or fy > 0.78:
            continue
        if "cheek" in rid and (fx < 0.24 or fx > 0.76 or fy > 0.74):
            continue
        if rid == "chin" and fy < 0.78:
            continue
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (ratio - 1.3) / 1.2 + (tex - 7.5) / 25.0)
        conf = min(0.93, 0.5 + (ratio - 1.3) * 0.4 + tex / 80.0)
        # Поры — нос и щёки; подбородок чаще текстура, если сигнал не очень сильный
        ftype = "pores" if rid in ("nose", "left_cheek", "right_cheek") or ratio >= 1.65 else "uneven_texture"
        if ftype == "pores" and hair_frac > 0.12:
            # остаточный волос в зоне — понижаем до текстуры или отбрасываем слабый сигнал
            if conf < CONF_FLOOR + 0.12:
                continue
            ftype = "uneven_texture"
        evidence = (
            "неоднородная текстура и заметные устья пор относительно остальной кожи"
            if ftype == "pores"
            else "визуально неровная текстура кожи относительно соседних участков"
        )
        findings.append({
            "type": ftype, "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": evidence,
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings[:3]


def _drop_hairline_false_pores(findings, bbox):
    """Финальный отсев: поры на линии роста волос / лбу не показываем."""
    out = []
    for f in findings:
        if f.get("type") != "pores":
            out.append(f)
            continue
        if f.get("region") == "forehead":
            continue
        geom = f.get("geom") or {}
        # geom в % кадра; переведём грубо через face frac если есть cx/cy в абсолюте
        # маркеры хранят x,y в процентах всего кадра — используем region + evidence
        # Доп. защита: если регион лоб или метка области содержит «лоб» — drop
        label = (f.get("region_label") or "") + " " + (f.get("region") or "")
        if "лоб" in label.lower() or "forehead" in label.lower():
            continue
        out.append(f)
    return out


def _detect_shine(grid, bbox, regions, base):
    """
    Жирный блеск в Т-зоне (лоб / нос / подбородок).
    Одиночный маленький блик от света не считаем жирностью —
    нужен заметный участок или совпадение в нескольких зонах Т.
    """
    candidates = []
    for rid in ("forehead", "nose", "chin"):
        pts = regions.get(rid) or []
        if len(pts) < 30:
            continue
        bright = [
            (x, y) for x, y in pts
            if grid.luma[y * grid.w + x] - base["luma"] > 38
        ]
        frac = len(bright) / len(pts)
        if frac < 0.18:
            continue
        bx0, by0 = min(p[0] for p in bright), min(p[1] for p in bright)
        bx1, by1 = max(p[0] for p in bright), max(p[1] for p in bright)
        box_area = max(1, (bx1 - bx0 + 1) * (by1 - by0 + 1))
        # Крошечный яркий блик при малой доле — скорее блик освещения.
        if box_area < len(pts) * 0.07 and frac < 0.26:
            continue
        cx = sum(p[0] for p in bright) / len(bright)
        cy = sum(p[1] for p in bright) / len(bright)
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (frac - 0.18) * 2.2 + 0.22)
        conf = min(0.9, 0.55 + frac * 0.8)
        candidates.append({
            "type": "shine", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "распределённый блеск в зоне Т относительно среднего тона кожи",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    if len(candidates) >= 2:
        return candidates
    # Одна зона — только если сигнал сильный (не похож на точечный блик света)
    return [
        c for c in candidates
        if c["strength"] >= 0.48 and c["confidence"] >= 0.70
    ]


def _detect_dryness(grid, bbox, regions, base):
    """
    Только явный визуальный сигнал шероховатости / матовой неоднородности.
    При низкой уверенности не возвращаем — лучше уточнить вопросом.
    """
    findings = []
    skin_all = _bbox_skin_set(grid, bbox)
    shine_proxy = 0
    for rid in ("forehead", "nose", "chin"):
        pts = regions.get(rid) or []
        if len(pts) < 20:
            continue
        bright = sum(1 for x, y in pts if grid.luma[y * grid.w + x] - base["luma"] > 34)
        shine_proxy = max(shine_proxy, bright / len(pts))
    if shine_proxy > 0.22:
        return []  # блеск в Т-зоне — не трактуем как сухость

    for rid in ("left_cheek", "right_cheek", "forehead", "chin"):
        pts = regions.get(rid) or []
        if len(pts) < 45:
            continue
        if _skin_ring_fraction(pts, skin_all, radius=3) < 0.82:
            continue
        laps = [grid.lap(x, y) for x, y in pts]
        tex = sum(laps) / len(laps)
        ratio = tex / max(0.5, base["tex"])
        # сухость: повышенный микроконтраст при относительно матовой зоне
        matte = sum(1 for x, y in pts if abs(grid.luma[y * grid.w + x] - base["luma"]) < 18) / len(pts)
        if tex < 9.0 or ratio < 1.35 or matte < 0.55:
            continue
        geom = _pick_interior_centroid(
            pts, bbox, skin_all,
            score_fn=lambda p: grid.lap(p[0], p[1]),
        )
        if not geom:
            continue
        cx, cy, bx0, by0, bx1, by1 = geom
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        strength = min(1.0, (ratio - 1.35) / 1.1 + (tex - 9.0) / 28.0)
        conf = min(0.86, 0.48 + (ratio - 1.35) * 0.35 + matte * 0.15)
        if conf < CONF_FLOOR + 0.04:
            continue
        findings.append({
            "type": "dryness", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "матовая неоднородная текстура — возможный визуальный признак сухости",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings[:1]



def _under_eye_band(pts, bbox, y0=0.49, y1=0.58):
    band = [
        p for p in pts
        if y0 <= _face_frac(p[0], p[1], bbox)[1] <= y1
        and not any(_in_rect(*_face_frac(p[0], p[1], bbox), r) for r in _EXCLUDE)
    ]
    return band or pts


def _mean_luma(grid, pts):
    if not pts:
        return 0.0
    return sum(grid.luma[y * grid.w + x] for x, y in pts) / len(pts)


def _mean_lap(grid, pts):
    if not pts:
        return 0.0
    return sum(grid.lap(x, y) for x, y in pts) / len(pts)


def _mean_grad(grid, pts):
    if not pts:
        return 0.0, 0.0
    gx = gy = 0.0
    for x, y in pts:
        a, b = grid.grad(x, y)
        gx += a
        gy += b
    n = len(pts)
    return gx / n, gy / n


def _line_peak_count(grid, pts, direction="horizontal"):
    """Сколько отдельных тонких пиков градиента — морщины, а не один контур мешка."""
    if len(pts) < 12:
        return 0
    buckets = {}
    for x, y in pts:
        key = y if direction == "horizontal" else x
        g = grid.grad(x, y)
        val = g[1] if direction == "horizontal" else g[0]
        buckets.setdefault(key, []).append(val)
    rows = sorted((k, sum(v) / len(v)) for k, v in buckets.items())
    if len(rows) < 4:
        return 0
    vals = [v for _, v in rows]
    med = sorted(vals)[len(vals) // 2]
    floor = med * 1.35 + 1.2
    peaks = 0
    for i in range(1, len(vals) - 1):
        if vals[i] >= floor and vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            if vals[i] >= max(vals[i - 1], vals[i + 1]) * 1.12:
                peaks += 1
    return peaks


def _bag_shelf_score(grid, bbox, under_pts):
    """
    Мешок: мягкая «полка» — верх/низ подглазья различаются по яркости,
    внизу один широкий горизонтальный контур, не сетка тонких линий.
    """
    if len(under_pts) < 14:
        return 0.0
    upper = _under_eye_band(under_pts, bbox, 0.48, 0.53)
    lower = _under_eye_band(under_pts, bbox, 0.53, 0.59)
    if len(upper) < 6 or len(lower) < 6:
        return 0.0
    u_l = _mean_luma(grid, upper)
    l_l = _mean_luma(grid, lower)
    shelf_d = abs(u_l - l_l)
    _gx_l, gy_l = _mean_grad(grid, lower)
    _gx_u, gy_u = _mean_grad(grid, upper)
    shelf_edge = max(0.0, gy_l - gy_u)
    peaks = _line_peak_count(grid, under_pts, "horizontal")
    if peaks >= 3:
        return 0.0
    score = 0.0
    if shelf_d >= 6:
        score += min(0.45, shelf_d / 28.0)
    if shelf_edge >= 1.2:
        score += min(0.40, shelf_edge / 8.0)
    if peaks <= 1:
        score += 0.18
    elif peaks == 2:
        score += 0.05
    return min(1.0, score)


def _detect_puffiness(grid, bbox, regions, base):
    """
    Мешки/отёчность под глазами: объёмная «полка», а не тонкие морщины.
    Ловим и светлый холмик, и классическую тень под объёмом.
    """
    findings = []
    pairs = (
        ("left_under_eye", "left_cheek"),
        ("right_under_eye", "right_cheek"),
    )
    for under_id, cheek_id in pairs:
        under = _under_eye_band(regions.get(under_id) or [], bbox)
        cheek = regions.get(cheek_id) or []
        if len(under) < 14 or len(cheek) < 30:
            continue
        u_luma = _mean_luma(grid, under)
        c_luma = _mean_luma(grid, cheek)
        u_tex = _mean_lap(grid, under)
        c_tex = _mean_lap(grid, cheek)
        bag = _bag_shelf_score(grid, bbox, under)
        bright_mound = u_luma >= c_luma + 5 and u_tex <= c_tex * 1.20
        shadowed_bag = bag >= 0.42 and u_luma <= c_luma + 2
        if not (bright_mound or shadowed_bag):
            continue
        dark_frac = sum(
            1 for x, y in under if base["luma"] - grid.luma[y * grid.w + x] > 18
        ) / len(under)
        if dark_frac > 0.55 and bag < 0.50 and not bright_mound:
            continue
        if _line_peak_count(grid, under, "horizontal") >= 3 and bag < 0.55:
            continue
        cx = sum(p[0] for p in under) / len(under)
        cy = sum(p[1] for p in under) / len(under)
        cx, cy = _anchor_under_eye(cx, cy, bbox, under_id, ftype="puffiness")
        bx0, by0 = min(p[0] for p in under), min(p[1] for p in under)
        bx1, by1 = max(p[0] for p in under), max(p[1] for p in under)
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[under_id]
        strength = min(1.0, 0.35 + bag * 0.55 + max(0.0, (u_luma - c_luma) / 28.0))
        conf = min(0.88, 0.52 + bag * 0.30 + (0.08 if bright_mound else 0.0))
        if conf < CONF_FLOOR:
            continue
        findings.append({
            "type": "puffiness", "region": under_id, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": "объёмная складка/мешок под глазом (мягкая полка, не тонкая морщина)",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
            "_bag_score": round(bag, 3),
        })
    return findings[:2]


def _detect_dullness(grid, bbox, regions, base, metrics_radiance_hint=None):
    """Тусклость: низкая «живость» тона без явной локальной пигментации."""
    skin_all = _bbox_skin_set(grid, bbox)
    face_pts = [p for rid in ("forehead", "left_cheek", "right_cheek", "nose", "chin")
                for p in (regions.get(rid) or [])]
    if len(face_pts) < 80:
        return []
    # контраст тона по лицу
    lumas = [grid.luma[y * grid.w + x] for x, y in face_pts]
    mean_l = sum(lumas) / len(lumas)
    var = sum((l - mean_l) ** 2 for l in lumas) / len(lumas)
    # тусклость: относительно ровный, «плоский» тон + ниже среднего luma
    flat = var < 180
    dim = mean_l < base["luma"] * 0.97 and mean_l < 138
    if not (flat and dim):
        return []
    # якорь — центр лба или щёк
    for rid in ("forehead", "left_cheek", "right_cheek"):
        pts = regions.get(rid) or []
        if len(pts) < 40:
            continue
        if _skin_ring_fraction(pts, skin_all, radius=2) < 0.8:
            continue
        geom = _pick_interior_centroid(pts, bbox, skin_all)
        if not geom:
            continue
        cx, cy, bx0, by0, bx1, by1 = geom
        rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
        conf = 0.64 if flat and dim else 0.58
        if conf < CONF_FLOOR:
            return []
        return [{
            "type": "dullness", "region": rid, "region_label": rlabel,
            "strength": 0.45,
            "confidence": round(conf, 2),
            "evidence": "тон кожи выглядит тусклым и менее живым относительно ожидаемой яркости",
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        }]
    return []


def _detect_tired_eyes(findings):
    """Признаки усталости взгляда: тёмные круги и/или мелкие морщины под глазами.
    Всегда два маркера — под каждым глазом.
    """
    under_wrinkles = [
        f for f in findings
        if f["type"] == "wrinkles" and "under_eye" in f.get("region", "")
    ]
    dark = [f for f in findings if f["type"] == "dark_circles"]
    puff = [f for f in findings if f["type"] == "puffiness"]
    if not (dark or under_wrinkles or puff):
        return []
    if not ((dark and under_wrinkles) or (dark and puff) or (len(dark) >= 2 and dark[0]["strength"] >= 0.45)):
        if not (len(dark) >= 2 and max(f["strength"] for f in dark) >= 0.55):
            return []

    def _side(rid):
        if "left" in (rid or ""):
            return "left"
        if "right" in (rid or ""):
            return "right"
        return None

    by_side = {"left": None, "right": None}
    for f in list(dark) + list(under_wrinkles) + list(puff):
        side = _side(f.get("region"))
        if side and by_side[side] is None:
            by_side[side] = f

    # если одна сторона — зеркалим геометрию на вторую
    present = [s for s, f in by_side.items() if f]
    if len(present) == 1:
        src = by_side[present[0]]
        other = "right" if present[0] == "left" else "left"
        g = dict(src.get("geom") or {})
        if "x" in g:
            g = {**g, "x": round(100.0 - float(g["x"]), 2)}
        other_rid = f"{other}_under_eye"
        by_side[other] = {
            **src,
            "region": other_rid,
            "region_label": "Под глазом справа" if other == "right" else "Под глазом слева",
            "geom": g,
        }

    out = []
    for side in ("left", "right"):
        src = by_side.get(side)
        if not src or not src.get("geom"):
            continue
        rid = src.get("region") or f"{side}_under_eye"
        out.append({
            "type": "tired_eyes",
            "region": rid,
            "region_label": src.get("region_label") or (
                "Под глазом слева" if side == "left" else "Под глазом справа"
            ),
            "strength": min(1.0, src["strength"] * 0.9 + 0.1),
            "confidence": round(min(0.88, src["confidence"] * 0.95), 2),
            "evidence": "видимые признаки усталости в зоне глаз",
            "geom": src["geom"],
        })
    return out


# Скан морщин вокруг глаз идёт по исходному фото: на сетке 168 px
# тонкие линии просто не разрешаются.
_SCAN_REF_D = 300.0      # опорное межзрачковое расстояние, px
_SCAN_UNDER_RATIO = 1.70  # во сколько раз линий больше, чем на гладкой щеке
_SCAN_CROW_RATIO = 1.45


def _scan_patch(arr, rect, scale):
    """Участок исходного фото в едином масштабе (по межзрачковому расстоянию)."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 14 or y1 - y0 < 10:
        return None
    crop = arr[y0:y1, x0:x1]
    if scale < 1.0:
        import cv2
        crop = cv2.resize(
            crop,
            (max(14, int(crop.shape[1] * scale)), max(10, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def _scan_line_energy(patch):
    """
    Энергия тонких тёмных линий на коже участка.
    blackhat отзывается на узкие складки и не реагирует на плавную тень.
    """
    if patch is None:
        return None
    import cv2
    import numpy as np

    ycc = cv2.cvtColor(patch, cv2.COLOR_RGB2YCrCb)
    cr = ycc[:, :, 1].astype(np.int16)
    cb = ycc[:, :, 2].astype(np.int16)
    luma = ycc[:, :, 0]
    skin = (
        (cr >= 132) & (cr <= 185) & (cb >= 76) & (cb <= 132) & (luma >= 60)
    ).astype(np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    med = float(np.median(gray[skin > 0])) if skin.any() else float(np.median(gray))
    # ресницы, зрачок и пряди волос темнее кожи — они не морщины.
    # Порог заметно ниже кожи: сама складка темнее лишь чуть-чуть и
    # обязана дойти до анализа.
    dark = cv2.dilate((gray < med - 45).astype(np.uint8), np.ones((3, 3), np.uint8))
    skin = skin * (1 - dark)
    skin_frac = float(skin.mean())
    if skin_frac < 0.45:
        return None
    bh = cv2.morphologyEx(
        cv2.GaussianBlur(gray, (3, 3), 0),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    vals = bh[skin > 0].astype(np.float32)
    if vals.size < 60:
        return None
    mask = ((bh >= max(6, np.percentile(vals, 93))) * skin).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    nh, nw = mask.shape
    lines = 0
    horizontal = np.zeros_like(mask)
    for i in range(1, count):
        _, _, cw, ch, area = stats[i]
        length = max(cw, ch)
        thick = max(1, min(cw, ch))
        if area < 5 or length < max(6, 0.13 * nw) or length / thick < 2.0:
            continue
        # морщины у глаз идут поперёк лица; вертикальные тяжи — это пряди волос
        if cw < ch:
            continue
        lines += 1
        horizontal[labels == i] = 1
    return {
        "p90": float(np.percentile(vals, 90)),
        "p98": float(np.percentile(vals, 98)),
        "cover": float(horizontal.mean()),
        "lines": lines,
        "skin": skin_frac,
    }


def _scan_verdict(zone, ref, ratio_floor):
    """Морщины есть, если линий заметно больше, чем на гладкой щеке того же кадра."""
    if not zone or not ref:
        return None
    r98 = zone["p98"] / max(0.8, ref["p98"])
    hit = (
        r98 >= ratio_floor
        and zone["p90"] >= 3.5
        and zone["p98"] >= 6.0
        and zone["lines"] >= 1
    )
    over = max(0.0, r98 - ratio_floor)
    return {
        "hit": hit,
        "ratio": r98,
        "lines": zone["lines"],
        "strength": max(0.30, min(0.95, 0.34 + over * 0.26 + zone["lines"] * 0.05)),
        "confidence": round(max(0.50, min(0.88, 0.56 + over * 0.11 + zone["lines"] * 0.03)), 2),
    }


def _eye_line_scan(source_img, grid, bbox, eyes):
    """
    Морщины вокруг глаз по исходному фото, а не по сетке.
    Зона глаза сравнивается с гладкой щекой того же кадра, поэтому
    разрешение, свет и JPEG-шум не сдвигают порог.
    """
    if source_img is None or not eyes:
        return None
    try:
        import cv2  # noqa: F401
        import numpy as np
    except ImportError:
        return None
    left, right = eyes.get("left"), eyes.get("right")
    if not left or not right:
        return None
    try:
        arr = np.asarray(source_img.convert("RGB"))
    except Exception:
        return None
    src_h, src_w = arr.shape[:2]
    sx = src_w / max(1, grid.w - 1)
    sy = src_h / max(1, grid.h - 1)
    px = {s: (eyes[s][0] * sx, eyes[s][1] * sy) for s in ("left", "right")}
    dist = abs(px["right"][0] - px["left"][0])
    if dist < 40:
        return None
    scale = min(1.0, _SCAN_REF_D / dist)
    out = {}
    for side in ("left", "right"):
        ex, ey = px[side]
        sign = -1 if side == "left" else 1
        under = _scan_line_energy(
            _scan_patch(arr, (ex - 0.23 * dist, ey + 0.16 * dist,
                              ex + 0.23 * dist, ey + 0.38 * dist), scale)
        )
        cx0 = ex + sign * 0.22 * dist
        cx1 = ex + sign * 0.42 * dist
        crow = _scan_line_energy(
            _scan_patch(arr, (min(cx0, cx1), ey - 0.02 * dist,
                              max(cx0, cx1), ey + 0.24 * dist), scale)
        )
        ref = _scan_line_energy(
            _scan_patch(arr, (ex + sign * 0.02 * dist - 0.16 * dist, ey + 0.60 * dist,
                              ex + sign * 0.02 * dist + 0.16 * dist, ey + 0.88 * dist), scale)
        )
        out[side] = {
            "under": _scan_verdict(under, ref, _SCAN_UNDER_RATIO),
            "crow": _scan_verdict(crow, ref, _SCAN_CROW_RATIO),
        }
    return out


def _crow_feet_pts(grid, bbox, side, eyes=None, regions=None):
    """
    Только кожа у внешнего угла глаза (не волосы у виска, не силуэт).
    Если есть зрачок — узкое окно от него к виску.
    """
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    eye = (eyes or {}).get(side)
    pts = []
    if eye:
        ex, ey = eye[0], eye[1]
        if side == "left":
            x_lo, x_hi = ex - 0.13 * fw, ex - 0.035 * fw
        else:
            x_lo, x_hi = ex + 0.035 * fw, ex + 0.13 * fw
        y_lo, y_hi = ey - 0.015 * fh, ey + 0.085 * fh
        for y in range(max(y0, int(y_lo)), min(y1, int(y_hi)) + 1):
            for x in range(max(x0, int(x_lo)), min(x1, int(x_hi)) + 1):
                if not grid.skin[y * grid.w + x]:
                    continue
                if grid.luma[y * grid.w + x] < 55:
                    continue
                pts.append((x, y))
        return pts

    under = (regions or {}).get(f"{side}_under_eye") or []
    for x, y in under:
        if not grid.skin[y * grid.w + x]:
            continue
        if grid.luma[y * grid.w + x] < 55:
            continue
        fx, fy = _face_frac(x, y, bbox)
        if side == "left":
            if 0.14 <= fx <= 0.28 and 0.42 <= fy <= 0.54:
                pts.append((x, y))
        else:
            if 0.72 <= fx <= 0.86 and 0.42 <= fy <= 0.54:
                pts.append((x, y))
    return pts


def _anchor_crow_feet(bbox, side, eyes=None):
    """
    Маркер на внешнем углу глаза (гусиные лапки).
    Выше и наружнее подглазья: у кантуса, где лучи складок, а не на щеке.
    """
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    eye = (eyes or {}).get(side)
    if eye:
        ex, ey = eye[0], eye[1]
        eye_h = eye[2] if len(eye) > 2 else 0.075 * fh
        # наружу от зрачка к виску (~полширины глаза + запас)
        cx = ex + (-0.145 * fw if side == "left" else 0.145 * fw)
        # почти на уровне внешнего угла, чуть ниже века
        cy = ey + max(0.008 * fh, eye_h * 0.18)
        cy = min(max(cy, ey + 0.004 * fh), ey + 0.032 * fh)
        # не уводить в подглазье / щёку
        cy = min(max(cy, y0 + 0.40 * fh), y0 + 0.495 * fh)
        cx = max(x0 + 0.04 * fw, min(x1 - 0.04 * fw, cx))
        return cx, cy
    if side == "left":
        return x0 + 0.16 * fw, y0 + 0.455 * fh
    return x0 + 0.84 * fw, y0 + 0.455 * fh


def _ridge_peak_count(grid, pts, direction="horizontal"):
    """Счётчик тонких складок на коже."""
    if len(pts) < 10:
        return 0
    buckets = {}
    for x, y in pts:
        key = y if direction == "horizontal" else x
        g = grid.grad(x, y)
        val = g[1] if direction == "horizontal" else g[0]
        buckets.setdefault(key, []).append(val)
    rows = sorted((k, sum(v) / len(v)) for k, v in buckets.items())
    if len(rows) < 5:
        return 0
    vals = [v for _, v in rows]
    med = sorted(vals)[len(vals) // 2]
    floor = med * 1.22 + 0.8
    peaks = 0
    for i in range(1, len(vals) - 1):
        if vals[i] >= floor and vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
            peaks += 1
    return peaks


def _ray_crease_score(grid, bbox, side, eyes):
    """
    Лучевой тест складок от внешнего угла наружу.
    Морщины = несколько чередующихся тёмных/светлых гребней вдоль лучей.
    Плавная тень глазницы даёт 0–1 перепад без осцилляций.
    """
    eye = (eyes or {}).get(side)
    if not eye:
        return 0, 0.0
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    ex, ey = eye[0], eye[1]
    # старт чуть снаружи зрачка
    sx = ex + (-0.07 * fw if side == "left" else 0.07 * fw)
    sy = ey + 0.02 * fh
    # лучи: наружу и чуть вниз / горизонталь / чуть вверх
    dirs = (
        (-1.0, 0.15) if side == "left" else (1.0, 0.15),
        (-1.0, 0.45) if side == "left" else (1.0, 0.45),
        (-0.85, -0.1) if side == "left" else (0.85, -0.1),
        (-0.7, 0.7) if side == "left" else (0.7, 0.7),
    )
    ray_hits = 0
    total_peaks = 0
    for dx, dy in dirs:
        # нормализуем шаг
        norm = (dx * dx + dy * dy) ** 0.5
        dx, dy = dx / norm, dy / norm
        samples = []
        for step in range(1, 12):
            x = int(round(sx + dx * step * 0.028 * fw))
            y = int(round(sy + dy * step * 0.028 * fw))
            if x < x0 or x > x1 or y < y0 or y > y1:
                break
            if not grid.skin[y * grid.w + x]:
                continue
            lum = grid.luma[y * grid.w + x]
            if lum < 50:
                continue
            samples.append(lum)
        if len(samples) < 6:
            continue
        # high-pass: отклонение от локального среднего
        peaks = 0
        for i in range(2, len(samples) - 2):
            window = samples[i - 2:i + 3]
            mid = samples[i]
            avg = sum(window) / len(window)
            # тёмная складка: локальный минимум заметно ниже соседей
            if mid < avg - 3.8 and mid <= samples[i - 1] and mid <= samples[i + 1]:
                if mid < samples[i - 2] - 2 or mid < samples[i + 2] - 2:
                    peaks += 1
        total_peaks += peaks
        if peaks >= 2:
            ray_hits += 1
        elif peaks >= 1:
            ray_hits += 0  # одиночный перепад тени не считается
    return ray_hits, float(total_peaks)


def _crow_feet_signal(grid, crow, floor, cheek_pts=None, bbox=None, side=None, eyes=None):
    """
    Гусиные лапки только при реальных складках на коже.
    Тень/объём глазницы и волосы у виска — не морщины.
    """
    if len(crow) < 12:
        return False, 0.0, 0.0, 0
    gx, gy = _mean_grad(grid, crow)
    peaks = max(
        _ridge_peak_count(grid, crow, "horizontal"),
        _line_peak_count(grid, crow, "horizontal"),
    )
    ray_hits, ray_peaks = 0, 0.0
    if bbox is not None and side and eyes:
        ray_hits, ray_peaks = _ray_crease_score(grid, bbox, side, eyes)
    # складки на лучах: минимум 3 локальных минимума суммарно.
    # плавная тень глазницы даёт 0–2, прищур с линиями — 3+.
    # средний градиент сам по себе НЕ достаточен (ложные морщины на ровной коже).
    hit = ray_peaks >= 3 and (ray_hits >= 1 or peaks >= 1)
    return hit, gy, gx, int(max(peaks, ray_peaks))


def _detect_wrinkles(grid, bbox, regions, base, eyes=None, source_img=None):
    """
    Морщины = тонкие линии (лоб / межбровье / под глазами / гусиные лапки).
    Под глазом и у внешнего угла — отдельные маркеры.
    Вокруг глаз решение принимает скан исходного фото: на сетке мелкие
    линии не разрешаются, поэтому их там не видно вовсе.
    """
    findings = []
    scan = _eye_line_scan(source_img, grid, bbox, eyes)
    fh_pts = regions.get("forehead") or []
    fh_gy = (
        sum(grid.grad(x, y)[1] for x, y in fh_pts) / len(fh_pts) if len(fh_pts) > 30 else 2.0
    )
    under_floor = max(2.8, fh_gy * 1.45)
    checks = [
        ("forehead", "horizontal", 4.2),
        ("glabella", "vertical", 4.2),
        ("left_under_eye", "horizontal", under_floor),
        ("right_under_eye", "horizontal", under_floor),
    ]
    under_raw = {}
    for rid, direction, floor in checks:
        pts = regions.get(rid) or []
        if len(pts) < 28:
            continue
        if "under_eye" in rid:
            pts = _under_eye_band(pts, bbox)
        gx_m, gy_m = _mean_grad(grid, pts)
        if direction == "horizontal":
            main, cross = gy_m, gx_m
            evidence = (
                "мелкие морщины под глазом"
                if "under_eye" in rid
                else "повторяющиеся горизонтальные морщины на лбу"
            )
        else:
            main, cross = gx_m, gy_m
            evidence = "вертикальные морщины в межбровной зоне"
        ratio_floor = 1.45 if "under_eye" in rid else 1.6
        if "under_eye" in rid:
            under_raw[rid] = (main, cross, pts, floor, ratio_floor, evidence)
            continue
        if main < floor or main < cross * ratio_floor:
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
        strength = min(1.0, (main - floor) / 6.0 + max(0.0, main / max(0.5, cross) - ratio_floor) * 0.5)
        conf = min(0.9, 0.5 + (main - floor) / 12.0 + (main / max(0.5, cross) - ratio_floor) * 0.3)
        findings.append({
            "type": "wrinkles", "region": rid, "region_label": rlabel,
            "strength": strength, "confidence": round(conf, 2),
            "evidence": evidence,
            "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
        })

    crow_labels = {
        "left": "У внешнего угла глаза слева",
        "right": "У внешнего угла глаза справа",
    }
    eyes = eyes or {}

    for rid, (main, cross, pts, floor, ratio_floor, evidence) in under_raw.items():
        side = "left" if "left" in rid else "right"
        bag = _bag_shelf_score(grid, bbox, pts)
        peaks = max(
            _ridge_peak_count(grid, pts, "horizontal"),
            _line_peak_count(grid, pts, "horizontal"),
        )
        crow = _crow_feet_pts(grid, bbox, side, eyes=eyes, regions=regions)
        cheek = regions.get(f"{side}_cheek") or []
        crow_hit, crow_gy, crow_gx, crow_peaks = _crow_feet_signal(
            grid, crow, floor, cheek_pts=cheek, bbox=bbox, side=side, eyes=eyes
        )

        # под глазом: только при реальных линиях, не из‑за тени мешка
        line_hit = peaks >= 2 and main >= floor * 0.9 and main >= cross * 1.35

        # скан исходного фото важнее сетки: он видит сами линии
        sc = (scan or {}).get(side) or {}
        sc_under, sc_crow = sc.get("under"), sc.get("crow")
        under_strength = under_conf = None
        if sc_under:
            line_hit = sc_under["hit"]
            if line_hit:
                peaks = max(peaks, 2 + sc_under["lines"])
                under_strength = sc_under["strength"]
                under_conf = sc_under["confidence"]
        if sc_crow:
            crow_hit = sc_crow["hit"]
            if crow_hit:
                crow_peaks = max(crow_peaks, 2 + sc_crow["lines"])

        # 1) гусиные лапки — отдельный маркер
        if crow_hit:
            cx, cy = _anchor_crow_feet(bbox, side, eyes=eyes)
            findings.append({
                "type": "wrinkles",
                "region": f"{side}_crow_feet",
                "region_label": crow_labels[side],
                "strength": (
                    sc_crow["strength"] if sc_crow and sc_crow["hit"]
                    else min(1.0, 0.35 + crow_gy / 16.0 + crow_peaks * 0.08)
                ),
                "confidence": (
                    # скан подтвердил линии — пара не должна распадаться на пороге
                    round(max(CONF_FLOOR + 0.02, sc_crow["confidence"]), 2)
                    if sc_crow and sc_crow["hit"]
                    else round(min(0.86, 0.52 + crow_gy / 22.0 + crow_peaks * 0.05), 2)
                ),
                "evidence": "мелкие морщины у внешнего угла глаза (гусиные лапки)",
                "geom": _to_pct(grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2),
                "_line_peaks": crow_peaks,
                "_bag_score": 0.0,
                "_crow": True,
                "_scan": bool(sc_crow and sc_crow["hit"]),
            })

        # 2) под глазом — только линии, не мешок.
        # Скан уже отделил тонкие линии от контура мешка, ему верим без вето.
        bag_veto = bag >= 0.50 and peaks < 3 and not (sc_under and sc_under["hit"])
        if line_hit and not bag_veto:
            strong = sorted(
                ((grid.grad(x, y)[1], x, y) for x, y in pts), reverse=True
            )[: max(6, len(pts) // 8)]
            cx = sum(s[1] for s in strong) / len(strong)
            cy = sum(s[2] for s in strong) / len(strong)
            cx, cy = _anchor_under_eye(cx, cy, bbox, rid, eyes=eyes, ftype="wrinkles")
            bx0, by0 = min(s[1] for s in strong), min(s[2] for s in strong)
            bx1, by1 = max(s[1] for s in strong), max(s[2] for s in strong)
            rlabel = dict((r[0], r[1]) for r in _REGIONS)[rid]
            strength = min(1.0, (main - floor) / 6.0 + max(0.0, main / max(0.5, cross) - ratio_floor) * 0.45)
            conf = min(0.86, 0.48 + (main - floor) / 14.0 + peaks * 0.05)
            if under_strength is not None:
                strength, conf = under_strength, under_conf
            findings.append({
                "type": "wrinkles", "region": rid, "region_label": rlabel,
                "strength": strength, "confidence": round(conf, 2),
                "evidence": evidence,
                "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
                "_line_peaks": peaks,
                "_bag_score": round(bag, 3),
                "_crow": False,
            })

    # парность crow только если сигнал был сильный (не достраиваем из воздуха)
    crow_found = [f for f in findings if f.get("region", "").endswith("crow_feet")]
    if len(crow_found) == 1 and (
        crow_found[0].get("_line_peaks", 0) >= 3 or crow_found[0].get("_scan")
    ):
        src = crow_found[0]
        side = "left" if "left" in src["region"] else "right"
        other = "right" if side == "left" else "left"
        # проверяем вторую сторону реально
        crow_o = _crow_feet_pts(grid, bbox, other, eyes=eyes, regions=regions)
        cheek_o = regions.get(f"{other}_cheek") or []
        hit_o, _, _, peaks_o = _crow_feet_signal(
            grid, crow_o, under_floor, cheek_pts=cheek_o, bbox=bbox, side=other, eyes=eyes
        )
        # морщины у глаз почти всегда симметричны: на второй стороне
        # достаточно ослабленного, но реального следа линий
        sc_o = ((scan or {}).get(other) or {}).get("crow")
        if sc_o and sc_o["ratio"] >= _SCAN_CROW_RATIO * 0.72:
            hit_o, peaks_o = True, max(peaks_o, 2 + sc_o["lines"])
        if hit_o:
            cx, cy = _anchor_crow_feet(bbox, other, eyes=eyes)
            findings.append({
                "type": "wrinkles",
                "region": f"{other}_crow_feet",
                "region_label": crow_labels[other],
                "strength": max(0.32, src["strength"] * 0.8),
                "confidence": round(
                    min(0.80, max(CONF_FLOOR + 0.02, src["confidence"] * 0.9)), 2
                ),
                "evidence": "мелкие морщины у внешнего угла глаза (гусиные лапки)",
                "geom": _to_pct(grid, cx, cy, int(cx) - 2, int(cy) - 2, int(cx) + 2, int(cy) + 2),
                "_crow": True,
                "_line_peaks": peaks_o,
                "_bag_score": 0.0,
            })

    under_findings = [f for f in findings if f.get("region", "").endswith("under_eye")]
    if len(under_findings) == 1:
        found = under_findings[0]
        other = "left_under_eye" if found["region"] == "right_under_eye" else "right_under_eye"
        if other in under_raw:
            main, cross, pts, floor, ratio_floor, evidence = under_raw[other]
            bag = _bag_shelf_score(grid, bbox, pts)
            peaks = max(
                _ridge_peak_count(grid, pts, "horizontal"),
                _line_peak_count(grid, pts, "horizontal"),
            )
            sc_o = ((scan or {}).get("left" if "left" in other else "right") or {}).get("under")
            pair_hit = peaks >= 2 and main >= floor * 0.85 and not (bag >= 0.55 and peaks < 3)
            if sc_o:
                pair_hit = sc_o["ratio"] >= _SCAN_UNDER_RATIO * 0.75
            if pair_hit:
                strong = sorted(
                    ((grid.grad(x, y)[1], x, y) for x, y in pts), reverse=True
                )[: max(6, len(pts) // 8)]
                cx = sum(s[1] for s in strong) / len(strong)
                cy = sum(s[2] for s in strong) / len(strong)
                cx, cy = _anchor_under_eye(cx, cy, bbox, other, eyes=eyes, ftype="wrinkles")
                bx0, by0 = min(s[1] for s in strong), min(s[2] for s in strong)
                bx1, by1 = max(s[1] for s in strong), max(s[2] for s in strong)
                rlabel = dict((r[0], r[1]) for r in _REGIONS)[other]
                # парный маркер не должен отсеиваться порогом уверенности:
                # морщины вокруг глаз симметричны, один кружок выглядит ошибкой
                pair_conf = max(CONF_FLOOR + 0.02, found["confidence"] * 0.90)
                findings.append({
                    "type": "wrinkles", "region": other, "region_label": rlabel,
                    "strength": max(0.3, found["strength"] * 0.75),
                    "confidence": round(min(0.80, pair_conf), 2),
                    "evidence": evidence,
                    "geom": _to_pct(grid, cx, cy, bx0, by0, bx1, by1),
                })
    return findings


def _resolve_bags_vs_wrinkles(findings):
    """
    Мешок и морщины могут сосуществовать.
    Убираем только «ложные морщины», которые на самом деле контур мешка
    (высокий bag_score, нет линий / гусиных лапок).
    """
    out = []
    for f in findings:
        if f.get("type") == "wrinkles" and "under_eye" in f.get("region", ""):
            if (
                f.get("_bag_score", 0) >= 0.55
                and f.get("_line_peaks", 0) < 2
                and not f.get("_crow")
            ):
                # чистый контур мешка без линий → мешок
                out.append({
                    **{k: v for k, v in f.items() if not k.startswith("_")},
                    "type": "puffiness",
                    "evidence": "объёмная складка/мешок под глазом (мягкая полка, не тонкая морщина)",
                })
                continue
        out.append({k: v for k, v in f.items() if not k.startswith("_")})
    # дедуп: если уже есть puffiness на стороне, не дублируем из конвертации
    seen_puff = set()
    deduped = []
    for f in out:
        if f.get("type") == "puffiness":
            side = "left" if "left" in f.get("region", "") else "right"
            key = (f["type"], side)
            if key in seen_puff:
                continue
            seen_puff.add(key)
        deduped.append(f)
    return deduped


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


def _cap_findings_keeping_pairs(findings, limit):
    """
    Обрезаем список, но никогда не оставляем глазной признак в одиночестве.
    Сначала по одному признаку каждого типа: иначе сильная краснота с
    тёмными кругами занимают все места и морщины не доходят до отчёта.
    """
    by_type = {}
    for f in findings:
        by_type.setdefault(f.get("type"), []).append(f)
    findings = [f for ftype in by_type for f in by_type[ftype]]
    kept = findings[:limit]
    kept_ids = {id(f) for f in kept}
    # «гусиные лапки» и подглазье — разные пары одного типа
    def _pair_key(f):
        spot = "crow" if "crow_feet" in (f.get("region") or "") else "under"
        return f.get("type"), spot

    eye_keys = {_pair_key(f) for f in kept if _is_eye_finding(f)}
    for key in eye_keys:
        sides = {
            "left" if "left" in (f.get("region") or "") else "right"
            for f in kept
            if _pair_key(f) == key and _is_eye_finding(f)
        }
        if len(sides) >= 2:
            continue
        for f in findings:
            if id(f) in kept_ids or _pair_key(f) != key or not _is_eye_finding(f):
                continue
            side = "left" if "left" in (f.get("region") or "") else "right"
            if side in sides:
                continue
            kept.append(f)
            kept_ids.add(id(f))
            sides.add(side)
            break
    return kept


def analyze(image_bytes):
    """
    Возвращает dict: quality, baseline-метрики и findings
    (только с уверенностью >= CONF_FLOOR). Бросает PhotoQualityError,
    если по фото нельзя дать честный результат.
    """
    img, px, w, h = _decode(image_bytes)
    grid = _Grid(px, w, h)

    face = _detect_face_haar(img)
    if face is _FACE_MISSING:
        # OpenCV уверенно сказал «лица нет» — не угадываем по цвету штукатурки/потолка.
        raise PhotoQualityError(
            "На фото не видно лица. Сделайте селфи анфас при дневном свете, "
            "без сильных фильтров и перекрытий."
        )
    if face is _FACE_NO_CV:
        # Без OpenCV — только строгий fallback по крупнейшей области кожи.
        bbox = _face_bbox(grid)
        face_frac = (
            bbox[0] / max(1, w - 1),
            bbox[1] / max(1, h - 1),
            bbox[2] / max(1, w - 1),
            bbox[3] / max(1, h - 1),
        )
    else:
        face_frac = face
        bbox = (
            max(0, int(face[0] * w)), max(0, int(face[1] * h)),
            min(w - 1, int(face[2] * w)), min(h - 1, int(face[3] * h)),
        )
        x0, y0, x1, y1 = bbox
        area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        skin_in_box = sum(
            1 for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1)
            if grid.skin[yy * w + xx]
        )
        # Потолок/стена иногда дают ложный бокс Haar — без кожи внутри это не лицо.
        if skin_in_box / area < 0.28:
            raise PhotoQualityError(
                "На фото не видно лица или кожа плохо различима. "
                "Сделайте селфи анфас без фильтров при дневном свете."
            )

    _validate_face_framing(face_frac, grid, bbox)
    quality = _check_quality(grid, bbox, source_img=img, face_frac=face_frac)
    _assert_no_glasses(grid, bbox, source_img=img, face_frac=face_frac)
    regions, face_pixels = _collect_region_pixels(grid, bbox)
    base = _baseline(grid, face_pixels)
    eyes = _find_eye_centers(grid, bbox, source_img=img)

    raw = []
    raw += _detect_red(grid, bbox, regions, base)
    raw += _detect_diffuse_redness(grid, bbox, regions, base)
    raw += _detect_dark_circles(grid, bbox, regions, base)
    raw += _detect_pigmentation(grid, bbox, regions, base)
    raw += _detect_texture(grid, bbox, regions, base)
    raw += _detect_shine(grid, bbox, regions, base)
    raw += _detect_dryness(grid, bbox, regions, base)
    raw += _detect_puffiness(grid, bbox, regions, base)
    raw += _detect_wrinkles(grid, bbox, regions, base, eyes=eyes, source_img=img)
    raw += _detect_nasolabial(grid, bbox, regions, base)
    raw += _detect_dullness(grid, bbox, regions, base)

    merged = _merge_findings(raw)
    merged = _resolve_bags_vs_wrinkles(merged)
    merged = _drop_hairline_false_pores(merged, bbox)
    merged = _sanitize_findings_markers(grid, bbox, merged, eyes=eyes)
    # Сосудистая краснота уже описывает щёки — не дублируем её ещё и
    # обычной «краснотой» в тех же зонах.
    if any(f["type"] == "rosacea_like" for f in merged):
        merged = [
            f for f in merged
            if not (f["type"] == "redness" and "cheek" in f["region"])
        ]
    # усталость взгляда — вторичный визуальный вывод из уже найденных зон глаз
    merged += _detect_tired_eyes(merged)
    merged = _sanitize_findings_markers(grid, bbox, merged, eyes=eyes)
    merged = _pair_eye_findings(grid, bbox, merged, eyes)
    findings = [f for f in merged if f["confidence"] >= CONF_FLOOR]
    findings.sort(key=lambda f: (f["confidence"] + f["strength"]), reverse=True)
    findings = _cap_findings_keeping_pairs(findings, MAX_FINDINGS)

    for f in findings:
        f["severity"] = _severity(f["strength"])
        f["severity_label"] = SEVERITY_LABELS[f["severity"]]
        f["label"] = FEATURE_LABELS[f["type"]]
        f["score"] = int(round(30 + f["strength"] * 65))

    # Глобальные ориентиры для подбора продуктов (та же пиксельная база).
    shine_strength = max((f["strength"] for f in findings if f["type"] == "shine"), default=0.0)
    red_strength = max(
        (f["strength"] for f in findings if f["type"] in ("redness", "inflammation", "rosacea_like")),
        default=0.0,
    )
    pores_strength = max((f["strength"] for f in findings if f["type"] == "pores"), default=0.0)
    wrinkle_strength = max(
        (f["strength"] for f in findings if f["type"] in ("wrinkles", "nasolabial")), default=0.0
    )
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
