# -*- coding: utf-8 -*-
"""Генерация PDF коммерческого предложения (демо)."""
import os
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(APP_DIR, "fonts")
_FONTS_READY = False


def _fmt_money(n):
    return f"{int(round(n)):,}".replace(",", " ")


def _sanitize_pdf_text(text):
    """Замена символов, которые ломаются в TTF/PDF (подстрочные индексы и т.п.)."""
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "CO₂": "CO2",
        "co₂": "co2",
        "₂": "2",
        "₃": "3",
        "\u2082": "2",
        "\u2083": "3",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _is_valid_ttf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1")
    except OSError:
        return False


def _resolve_font_paths():
    """Для PDF — Arial/DejaVu (надёжная кириллица в просмотрщиках). Gilroy только на сайте."""
    candidates = [
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            os.path.join(FONTS_DIR, "Gilroy-Regular.ttf"),
            os.path.join(FONTS_DIR, "Gilroy-Semibold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if _is_valid_ttf(regular) and _is_valid_ttf(bold):
            return regular, bold
    raise FileNotFoundError("Не найден TTF-шрифт с кириллицей для PDF.")


def _ensure_reportlab_fonts():
    """ReportLab: регистрация Gilroy + связка normal/bold (иначе <b> → Helvetica без кириллицы)."""
    global _FONTS_READY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if _FONTS_READY:
        return "KPCyr", "KPCyr-Bold"

    regular, bold = _resolve_font_paths()
    pdfmetrics.registerFont(TTFont("KPCyr", regular))
    pdfmetrics.registerFont(TTFont("KPCyr-Bold", bold))
    pdfmetrics.registerFontFamily(
        "KPCyr",
        normal="KPCyr",
        bold="KPCyr-Bold",
        italic="KPCyr",
        boldItalic="KPCyr-Bold",
    )
    _FONTS_READY = True
    return "KPCyr", "KPCyr-Bold"


def _p(text, style):
    from reportlab.platypus import Paragraph
    return Paragraph(text, style)



def _generate_kp_reportlab(deal_id, answers, spec, catalog, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle

    font, font_bold = _ensure_reportlab_fonts()

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    story = []

    base = ParagraphStyle(
        "kp_base", fontName=font, fontSize=10, leading=14,
        textColor=colors.HexColor("#403B34"),
    )
    title = ParagraphStyle(
        "kp_title", parent=base, fontName=font_bold, fontSize=17, leading=20,
        textColor=colors.HexColor("#141210"), spaceAfter=4,
    )
    muted = ParagraphStyle("kp_muted", parent=base, fontSize=9, textColor=colors.HexColor("#857F74"))
    notice = ParagraphStyle(
        "kp_notice", parent=base, fontSize=9, leading=13,
        textColor=colors.HexColor("#5C574F"),
        backColor=colors.HexColor("#F4F1EB"),
        borderPadding=(8, 10, 8, 10),
        spaceBefore=0,
        spaceAfter=8,
    )
    cell = ParagraphStyle("kp_cell", parent=base, fontSize=9, leading=12)
    cell_bold = ParagraphStyle("kp_cell_bold", parent=cell, fontName=font_bold)
    cell_muted = ParagraphStyle("kp_cell_muted", parent=cell, fontSize=8, textColor=colors.HexColor("#857F74"))

    header = ParagraphStyle("kp_header", parent=cell, fontName=font_bold, textColor=colors.HexColor("#141210"))

    story.append(_p(escape("Демонстрационное КП · промышленная вентиляция · не является публичной офертой"), muted))
    story.append(Spacer(1, 6))
    story.append(_p(
        escape(
            "Данная версия коммерческого предложения сформирована в демонстрационном режиме. "
            "В рабочем внедрении документ оформляется в фирменном стиле компании: "
            "с логотипом, реквизитами, контактными данными менеджера и условиями поставки."
        ),
        notice,
    ))
    story.append(Spacer(1, 6))
    story.append(_p(f"Коммерческое предложение № {escape(deal_id.upper())}", title))
    story.append(_p(escape(catalog["brand"]["tagline"]), muted))
    story.append(Spacer(1, 10))
    story.append(_p(
        f'<font name="{font_bold}">Заказчик:</font> {escape(answers.get("contact_name", "—"))}',
        base,
    ))
    flow = f"{spec['required_flow_m3h']:,}".replace(",", " ")
    story.append(_p(
        f'<font name="{font_bold}">Объект:</font> {escape(_sanitize_pdf_text(spec["object_label"]))} · {answers.get("area_m2")} м² · '
        f'высота {answers.get("height_m")} м · расход {flow} м³/ч · фильтр {escape(_sanitize_pdf_text(spec["filter_class"]))}',
        base,
    ))
    story.append(Spacer(1, 10))

    data = [[
        _p("№", header),
        _p("Наименование", header),
        _p("Кол-во", header),
        _p("Сумма, руб.", header),
    ]]
    for i, line in enumerate(spec["lines"], 1):
        name = _sanitize_pdf_text(line["name"])
        specs = _sanitize_pdf_text(line["specs"])
        row_text = (
            f'<font name="{font_bold}">{escape(name)}</font><br/>'
            f'{escape(specs)}'
        )
        data.append([
            _p(str(i), cell),
            _p(row_text, cell),
            _p(escape(f"{line['qty']} {line['unit']}"), cell),
            _p(escape(_fmt_money(line["sum_rub"])), cell),
        ])
    data.append([
        _p("", cell),
        _p("", cell),
        _p("Итого:", cell_bold),
        _p(_fmt_money(spec["total_rub"]), cell_bold),
    ])

    table = Table(data, colWidths=[22, 285, 72, 78], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F1EB")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E4DED3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(_p(
        escape(
            f"Срок поставки ориентировочно: {spec['lead_days']} раб. дней. "
            "Документ сформирован автоматически (демо-кейс)."
        ),
        muted,
    ))
    story.append(Spacer(1, 8))
    story.append(_p(
        escape(
            "Это демонстрационная версия КП. При внедрении шаблон настраивается под бренд компании: "
            "логотип, шапка и подвал, контакты отдела продаж, условия оплаты и гарантии."
        ),
        notice,
    ))
    doc.build(story)
    return path


def generate_kp_pdf(deal_id, answers, spec, catalog, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kp-{deal_id}.pdf")
    # Всегда ReportLab — стабильная кириллица на macOS и в Docker
    return _generate_kp_reportlab(deal_id, answers, spec, catalog, path)
