from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arabic_reshaper
from bidi import get_display
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from discql import cover_art, musical_key
from discql.web import repository

@dataclass(frozen=True)
class LabelPreset:
    key: str
    display_name: str
    label_width: float
    label_height: float
    labels_per_row: int
    labels_per_col: int
    margin_left: float
    margin_top: float
    horizontal_gap: float


# draw_label's internal layout (padding, column offsets, font sizes...) was
# designed against this label size - other presets get that same layout
# proportionally scaled to fit their own (usually smaller) label, see
# _scale_for(). Vertical gap between rows is 0 for both current presets
# (rows sit flush against each other) - generate_sticker_sheet_pdf assumes
# this rather than taking a separate vertical-gap value.
REFERENCE_LABEL_WIDTH = 96 * mm
REFERENCE_LABEL_HEIGHT = 50.8 * mm

PRESETS: dict[str, LabelPreset] = {
    # The original sheet this project targeted - see the predecessor
    # project's src/pdf_label_generator.py on its cli-reportlab-refactor
    # branch.
    "avery_l4744rev65": LabelPreset(
        key="avery_l4744rev65",
        display_name="Avery Zweckform L4744REV-65 (96 x 50.8mm, 2x5)",
        label_width=REFERENCE_LABEL_WIDTH,
        label_height=REFERENCE_LABEL_HEIGHT,
        labels_per_row=2,
        labels_per_col=5,
        margin_left=7 * mm,
        margin_top=21.5 * mm,
        horizontal_gap=4 * mm,
    ),
    # Herma "SPECIAL" film/foil labels, product no. 4222 (silver polyester,
    # 63.5 x 29.633mm, 3x9 = 27/sheet). Dimensions taken from Herma's own
    # punch template (herma.com's downloadable "Stanzvorlage" PDF for this
    # product) and cross-checked against A4: 2*7.21 + 3*63.5 + 2*2.54 =
    # 210.0mm; 2*15.15 + 9*29.633 = 297.0mm - both exact.
    "herma_4222": LabelPreset(
        key="herma_4222",
        display_name="Herma 4222 (63.5 x 29.633mm, 3x9)",
        label_width=63.5 * mm,
        label_height=29.633 * mm,
        labels_per_row=3,
        labels_per_col=9,
        margin_left=7.21 * mm,
        margin_top=15.15 * mm,
        horizontal_gap=2.54 * mm,
    ),
}
DEFAULT_PRESET_KEY = "avery_l4744rev65"


def _scale_for(preset: LabelPreset) -> float:
    """How much to shrink (never enlarge - capped at 1.0) the reference
    layout to fit preset's own label size, preserving the reference
    design's proportions (one factor for both axes) rather than distorting
    text/shapes by scaling width and height independently.
    """
    return min(1.0, preset.label_width / REFERENCE_LABEL_WIDTH, preset.label_height / REFERENCE_LABEL_HEIGHT)

# DejaVu Sans instead of the PDF-standard Helvetica: bundled under fonts/ (a
# package asset, not a system font path - see README's "no unnecessary
# system dependencies" reasoning elsewhere) so Cyrillic/Greek/extended-Latin
# release titles render correctly instead of as blank tofu boxes, which
# Helvetica's base-14 WinAnsi encoding can't do at all. Arabic and CJK titles
# still won't render (DejaVu has no CJK glyphs, and Arabic additionally needs
# reshaping/bidi reordering reportlab's plain drawString doesn't do) - a
# follow-up if needed, not solved here.
FONT_DIR = Path(__file__).parent / "fonts"
FONT = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"
pdfmetrics.registerFont(TTFont(FONT, str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_DIR / "DejaVuSans-Bold.ttf")))

MAX_TRACK_ROWS = 8


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _shaped(text: str) -> str:
    """Reshapes Arabic script into its correct joined letterforms and
    reorders bidirectional text (Arabic reads right-to-left; embedded Latin/
    digit runs stay left-to-right) before drawing - reportlab's drawString
    only ever draws characters left-to-right in isolated form on its own.
    Applied at draw time only, after truncation/width-fitting (which stay on
    the plain logical text) - a no-op for text with no Arabic in it.
    """
    return get_display(arabic_reshaper.reshape(text))


# Ranges covering Hiragana/Katakana, common + extended CJK Unified
# Ideographs, Hangul, and CJK compatibility/fullwidth forms - not
# exhaustive of every Unicode CJK block, but covers real-world Chinese/
# Japanese/Korean release metadata.
_CJK_RANGES = [
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
]


def _has_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


def _cjk_image(text: str, font_path: Path) -> Image.Image:
    """Rasterizes text with a CJK-capable font via Pillow/FreeType, which
    (unlike reportlab's TTFont) can load any font format - real-world CJK
    fonts ship as OpenType/CFF, which reportlab's TTFont rejects outright
    ("postscript outlines are not supported"). Rendered at a fixed, high
    pixel size regardless of final point size, for print-quality sharpness
    once scaled down onto the sticker; transparent background, black text.
    """
    px_size = 200
    font = ImageFont.truetype(str(font_path), px_size)
    bbox = font.getbbox(text)
    width = max(1, bbox[2] - bbox[0]) + 8
    height = max(1, bbox[3] - bbox[1]) + 8
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0] + 4, -bbox[1] + 4), text, font=font, fill=(0, 0, 0, 255))
    return img


def _draw_text(
    c: canvas.Canvas, x: float, y: float, text: str, font: str, size: float, cjk_font_path: Path | None
) -> None:
    """Draws text with its baseline at (x, y): as vector text via reportlab
    normally, or - if it contains CJK characters and cjk_font_path is
    configured (see config.py's CJK_FONT_PATH) - as a small rasterized image
    instead (see _cjk_image), sized to roughly match the surrounding vector
    text's height. If cjk_font_path isn't configured, CJK text still goes
    through the normal vector path below and renders as whatever (likely
    blank) glyphs the sticker font has for those characters.
    """
    if cjk_font_path is not None and _has_cjk(text):
        img = _cjk_image(text, cjk_font_path)
        height_pt = size * 1.05
        width_pt = height_pt * (img.width / img.height)
        c.drawImage(
            ImageReader(img), x, y - size * 0.22, width=width_pt, height=height_pt,
            preserveAspectRatio=True, anchor="sw", mask="auto",
        )
    else:
        c.setFont(font, size)
        c.drawString(x, y, _shaped(text))


def _blacken_waveform(path: Path) -> Image.Image:
    """The stored waveform PNG is colored for the web UI's dark theme
    (accent blue on transparent, see audio_analysis.generate_waveform) -
    recolored to black-on-transparent here for print, at sticker-generation
    time rather than by touching the stored file, so the web UI's waveform
    is unaffected.
    """
    source = Image.open(path).convert("RGBA")
    alpha = source.split()[3]
    black = Image.new("RGBA", source.size, (0, 0, 0, 0))
    black.putalpha(alpha)
    return black


def draw_label(
    c: canvas.Canvas,
    x: float,
    y: float,
    release: repository.ReleaseDetail,
    cover_path: Path | None,
    waveform_dir: Path,
    key_notation: str = "standard",
    cjk_font_path: Path | None = None,
    preset: LabelPreset = PRESETS[DEFAULT_PRESET_KEY],
    dj_name: str = "",
) -> None:
    """Draws one release's sticker at bottom-left position (x, y) on the
    given canvas, sized to preset's label dimensions (see _scale_for - the
    layout below was designed for the reference/Avery size and scales down
    proportionally for smaller presets).
    """
    s = _scale_for(preset)
    padding = 2 * mm * s
    cover_size = 10 * mm * s
    track_index_width = cover_size

    artist = _truncate(", ".join(release.artist_names) or "Unknown Artist", 45)
    title = _truncate(release.title, 45)
    label_name = _truncate(", ".join(label.name for label in release.labels), 30)
    catalog_no = _truncate(", ".join(label.catalog_number for label in release.labels if label.catalog_number), 20)
    year = str(release.year) if release.year else ""
    genres = _truncate(", ".join(release.genres[:2]), 30)

    header_y = y + preset.label_height - padding

    footer_y = header_y - 2 * mm * s
    footer_size = 6.5 * s
    footer_parts = [part for part in (label_name, catalog_no, year, genres) if part]
    footer_parts.append(f"ID: {release.id}")
    if dj_name:
        footer_parts.append(dj_name)
    if release.date_added:
        footer_parts.append(f"Added: {release.date_added[:10]}")
    footer_text = _truncate(", ".join(footer_parts), 120)
    _draw_text(c, x + padding, footer_y, footer_text, FONT, footer_size, cjk_font_path)

    cover_y = footer_y - 3 * mm * s - cover_size
    if cover_path is not None and cover_path.is_file():
        c.drawImage(
            str(cover_path),
            x + padding,
            cover_y,
            width=cover_size,
            height=cover_size,
            preserveAspectRatio=True,
            anchor="sw",
        )

    text_x = x + padding + track_index_width + 2 * mm * s
    text_y = cover_y + cover_size - 3 * mm * s - 1.4 * mm * s
    available_width = preset.label_width - track_index_width - 4 * mm * s - 6 * mm * s

    artist_size = 10 * s
    c.setFont(FONT, artist_size)
    while c.stringWidth(artist, FONT, artist_size) > available_width and len(artist) > 1:
        artist = artist[:-1]
    _draw_text(c, text_x, text_y, artist, FONT, artist_size, cjk_font_path)

    text_y -= 5.5 * mm * s
    title_size = 14 * s
    c.setFont(FONT_BOLD, title_size)
    while c.stringWidth(title, FONT_BOLD, title_size) > available_width and len(title) > 1:
        title = title[:-1]
    _draw_text(c, text_x, text_y, title, FONT_BOLD, title_size, cjk_font_path)

    table_y = text_y - 7 * mm * s
    col_pos = x + padding
    track_title_x = col_pos + track_index_width + 2 * mm * s
    row_height = 3.5 * mm * s
    track_font_size = 7.5 * s

    for i in range(min(MAX_TRACK_ROWS, len(release.tracks))):
        track = release.tracks[i]
        row_y = table_y - (i * row_height)

        position = _truncate(track.position or "", 3)
        track_title = _truncate(track.title, 28)
        bpm = str(int(track.bpm)) if track.bpm else ""
        key = musical_key.format_key(track.musical_key, track.musical_key_scale, key_notation, compact=True) or ""

        c.setFont(FONT, track_font_size)
        position_width = c.stringWidth(position, FONT, track_font_size)
        _draw_text(c, col_pos + track_index_width - position_width, row_y, position, FONT, track_font_size, cjk_font_path)
        _draw_text(c, track_title_x, row_y, track_title, FONT, track_font_size, cjk_font_path)
        if bpm:
            c.drawString(col_pos + 48 * mm * s, row_y, bpm)
        if key:
            c.drawString(col_pos + 58 * mm * s, row_y, key)

        if track.waveform_path:
            waveform_file = waveform_dir / str(release.id) / track.waveform_path
            if waveform_file.is_file():
                c.drawImage(
                    ImageReader(_blacken_waveform(waveform_file)),
                    col_pos + 70 * mm * s,
                    row_y - 0.5 * mm * s,
                    width=18 * mm * s,
                    height=3 * mm * s,
                    preserveAspectRatio=True,
                    anchor="sw",
                    mask="auto",
                )


def generate_release_sticker_pdf(
    release: repository.ReleaseDetail,
    cover_dir: Path,
    waveform_dir: Path,
    sticker_dir: Path,
    key_notation: str = "standard",
    cjk_font_path: Path | None = None,
    preset: LabelPreset = PRESETS[DEFAULT_PRESET_KEY],
    dj_name: str = "",
) -> Path:
    """Renders release's sticker onto a single-page A4 PDF at sticker_dir /
    "<release.id>.pdf", overwriting any previous version - regeneration is
    cheap (a single canvas draw from already-cached cover/waveform/analysis
    data) so there's no reason to keep a stale sticker around.
    """
    sticker_dir.mkdir(parents=True, exist_ok=True)
    out_path = sticker_dir / f"{release.id}.pdf"

    cover_path = cover_art.find_cached_cover(cover_dir, release.id)

    c = canvas.Canvas(str(out_path), pagesize=A4)
    x = preset.margin_left
    y = A4[1] - preset.margin_top - preset.label_height
    draw_label(c, x, y, release, cover_path, waveform_dir, key_notation, cjk_font_path, preset, dj_name)
    c.save()

    return out_path


def generate_sticker_sheet_pdf(
    releases: list[repository.ReleaseDetail],
    cover_dir: Path,
    waveform_dir: Path,
    sticker_dir: Path,
    key_notation: str = "standard",
    cjk_font_path: Path | None = None,
    preset: LabelPreset = PRESETS[DEFAULT_PRESET_KEY],
    dj_name: str = "",
) -> Path:
    """Renders one combined multi-page A4 PDF for a whole selection of
    releases, filling preset's grid in order, page-breaking once a page is
    full. Overwrites sticker_dir/"selection.pdf" on every call, same "cheap
    to regenerate" reasoning as generate_release_sticker_pdf. Does not track
    which slots were already printed across separate runs - always starts
    fresh at the first slot.
    """
    sticker_dir.mkdir(parents=True, exist_ok=True)
    out_path = sticker_dir / "selection.pdf"

    c = canvas.Canvas(str(out_path), pagesize=A4)
    per_page = preset.labels_per_row * preset.labels_per_col
    for i, release in enumerate(releases):
        if i and i % per_page == 0:
            c.showPage()
        slot = i % per_page
        row, col = divmod(slot, preset.labels_per_row)
        x = preset.margin_left + col * (preset.label_width + preset.horizontal_gap)
        y = A4[1] - preset.margin_top - (row + 1) * preset.label_height
        cover_path = cover_art.find_cached_cover(cover_dir, release.id)
        draw_label(c, x, y, release, cover_path, waveform_dir, key_notation, cjk_font_path, preset, dj_name)
    c.save()

    return out_path
