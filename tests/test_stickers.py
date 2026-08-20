from __future__ import annotations

import sqlite3

import pytest
from PIL import Image

from discql import db, stickers
from discql.discogs_api import ArtistData, LabelData, ReleaseData, TrackData
from discql.sync import upsert_release
from discql.web import repository


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.create_function("strip_accents", 1, db.strip_accents, deterministic=True)
    db.migrate(connection)
    yield connection
    connection.close()


def seed(conn, release_id=1):
    data = ReleaseData(
        id=release_id,
        title="Alpha",
        year=2020,
        genres=["Electronic"],
        styles=["Deep House"],
        formats=[{"name": "Vinyl", "qty": "1"}],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url="https://example.com/cover.jpg",
        artists=[ArtistData(id=release_id * 10, name="Artist A")],
        labels=[LabelData(id=release_id * 100, name="Some Label", catalog_number="CAT001")],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:45", artist=None),
            TrackData(position="A2", title="Track Two", duration="4:12", artist=None),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")


def test_shaped_reorders_arabic_for_left_to_right_drawing():
    # "Simon" (Latin) stays where it logically is; the Arabic word gets
    # reshaped into its joined letterforms and moved to visual (right-to-
    # left) reading order in front of it - reportlab's drawString can't do
    # either of these on its own.
    result = stickers._shaped("سيمون = Simon")
    assert result.startswith("Simon = ")
    assert result != "سيمون = Simon"  # actually transformed, not passed through
    assert "س" not in result  # original (unjoined) seen letter is gone, replaced by a joined presentation form


def test_shaped_is_a_noop_for_plain_latin_and_cyrillic_text():
    assert stickers._shaped("Acid Test") == "Acid Test"
    assert stickers._shaped("Мой Друг Тима") == "Мой Друг Тима"


def test_greek_and_georgian_need_no_special_handling():
    # Both are simple left-to-right alphabetic scripts (no reshaping/bidi
    # like Arabic, no missing glyphs like CJK) - already covered by the
    # bundled DejaVu Sans font (see FONT/FONT_BOLD) with zero extra code.
    assert stickers._shaped("Ελληνικά ΑΩ") == "Ελληνικά ΑΩ"
    assert stickers._shaped("ქართული აბგ") == "ქართული აბგ"
    assert stickers._has_cjk("Ελληνικά") is False
    assert stickers._has_cjk("ქართული") is False


def test_has_cjk_detects_chinese_japanese_and_korean():
    assert stickers._has_cjk("老虎的偉大秩序") is True  # Chinese (Traditional)
    assert stickers._has_cjk("これからの緊急災害") is True  # Japanese (Hiragana + Kanji)
    assert stickers._has_cjk("서울") is True  # Korean (Hangul)


def test_has_cjk_false_for_latin_cyrillic_and_arabic():
    assert stickers._has_cjk("Acid Test") is False
    assert stickers._has_cjk("Мой Друг Тима") is False
    assert stickers._has_cjk("سيمون") is False


def test_cjk_image_rasterizes_text_to_a_non_empty_rgba_image():
    # Any valid font file works mechanically here (this only exercises the
    # rasterization pipeline, not real CJK glyph shapes) - the bundled
    # DejaVu Sans is enough, no dependency on a system CJK font being
    # present on the machine running the tests.
    font_path = stickers.FONT_DIR / "DejaVuSans.ttf"

    img = stickers._cjk_image("老虎", font_path)

    assert img.mode == "RGBA"
    assert img.width > 0
    assert img.height > 0


def test_draw_text_uses_raster_path_when_cjk_font_configured(tmp_path):
    from reportlab.pdfgen import canvas as canvas_module

    font_path = stickers.FONT_DIR / "DejaVuSans.ttf"
    out_path = tmp_path / "test.pdf"
    c = canvas_module.Canvas(str(out_path))

    # Should not raise, regardless of whether cjk_font_path is set.
    stickers._draw_text(c, 10, 10, "老虎的偉大秩序", stickers.FONT, 10, font_path)
    stickers._draw_text(c, 10, 30, "老虎的偉大秩序", stickers.FONT, 10, None)
    stickers._draw_text(c, 10, 50, "Acid Test", stickers.FONT, 10, font_path)
    c.save()

    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_pdf_with_cjk_title_and_no_cjk_font_configured(conn, tmp_path):
    seed(conn)
    conn.execute("UPDATE releases SET title = ? WHERE id = 1", ("老虎的偉大秩序",))
    conn.commit()
    release = repository.get_release_detail(conn, 1)

    out_path = stickers.generate_release_sticker_pdf(
        release, tmp_path / "covers", tmp_path / "waveforms", tmp_path / "stickers"
    )

    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_pdf_with_cjk_title_and_cjk_font_configured(conn, tmp_path):
    seed(conn)
    conn.execute("UPDATE releases SET title = ? WHERE id = 1", ("老虎的偉大秩序",))
    conn.commit()
    release = repository.get_release_detail(conn, 1)
    font_path = stickers.FONT_DIR / "DejaVuSans.ttf"

    out_path = stickers.generate_release_sticker_pdf(
        release, tmp_path / "covers", tmp_path / "waveforms", tmp_path / "stickers", cjk_font_path=font_path
    )

    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_pdf_with_dj_name_and_date_added(conn, tmp_path):
    seed(conn)  # date_added="2024-01-01T00:00:00"
    release = repository.get_release_detail(conn, 1)

    out_path = stickers.generate_release_sticker_pdf(
        release, tmp_path / "covers", tmp_path / "waveforms", tmp_path / "stickers", dj_name="ƒDiode"
    )

    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_pdf_with_no_cover_or_analysis(conn, tmp_path):
    seed(conn)
    release = repository.get_release_detail(conn, 1)

    out_path = stickers.generate_release_sticker_pdf(
        release, tmp_path / "covers", tmp_path / "waveforms", tmp_path / "stickers"
    )

    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_pdf_with_cover_and_analysis(conn, tmp_path):
    seed(conn)

    cover_dir = tmp_path / "covers"
    cover_dir.mkdir()
    Image.new("RGB", (4, 4)).save(cover_dir / "1.jpg")

    waveform_dir = tmp_path / "waveforms"
    (waveform_dir / "1").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(waveform_dir / "1" / "A1.png")

    conn.execute(
        "UPDATE tracks SET bpm = ?, musical_key = ?, musical_key_scale = ?, waveform_path = ? "
        "WHERE release_id = 1 AND position = 'A1'",
        (128.0, "C", "major", "A1.png"),
    )
    conn.commit()

    release = repository.get_release_detail(conn, 1)

    out_path = stickers.generate_release_sticker_pdf(release, cover_dir, waveform_dir, tmp_path / "stickers")

    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_sheet_pdf_single_page_for_few_releases(conn, tmp_path):
    for release_id in (1, 2, 3):
        seed(conn, release_id)
    releases = [repository.get_release_detail(conn, i) for i in (1, 2, 3)]

    out_path = stickers.generate_sticker_sheet_pdf(
        releases, tmp_path / "covers", tmp_path / "waveforms", tmp_path / "stickers"
    )

    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_sheet_pdf_breaks_page_after_ten_releases(conn, tmp_path):
    for release_id in range(1, 12):
        seed(conn, release_id)
    releases = [repository.get_release_detail(conn, i) for i in range(1, 12)]

    out_path = stickers.generate_sticker_sheet_pdf(
        releases, tmp_path / "covers", tmp_path / "waveforms", tmp_path / "stickers"
    )

    assert out_path.is_file()
    pdf_bytes = out_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    # A second page was actually started for the 11th release (excludes the
    # "/Type /Pages" tree root - only an actual page object ends "Page\n").
    assert pdf_bytes.count(b"/Type /Page\n") == 2


def test_avery_preset_scale_is_1_0():
    # The reference layout was designed against this exact preset's size.
    assert stickers._scale_for(stickers.PRESETS["avery_l4744rev65"]) == 1.0


def test_herma_preset_scale_is_height_constrained_and_below_1():
    preset = stickers.PRESETS["herma_4222"]
    scale = stickers._scale_for(preset)
    assert 0 < scale < 1
    assert scale == preset.label_height / stickers.REFERENCE_LABEL_HEIGHT


def test_herma_grid_dimensions_add_up_exactly_to_a4():
    from reportlab.lib.pagesizes import A4

    preset = stickers.PRESETS["herma_4222"]
    total_width = 2 * preset.margin_left + preset.labels_per_row * preset.label_width
    total_width += (preset.labels_per_row - 1) * preset.horizontal_gap
    total_height = 2 * preset.margin_top + preset.labels_per_col * preset.label_height

    assert total_width == pytest.approx(A4[0], abs=0.1)
    assert total_height == pytest.approx(A4[1], abs=0.1)


def test_generate_sticker_pdf_with_herma_preset(conn, tmp_path):
    seed(conn)
    release = repository.get_release_detail(conn, 1)

    out_path = stickers.generate_release_sticker_pdf(
        release,
        tmp_path / "covers",
        tmp_path / "waveforms",
        tmp_path / "stickers",
        preset=stickers.PRESETS["herma_4222"],
    )

    assert out_path.read_bytes().startswith(b"%PDF")


def test_generate_sticker_sheet_pdf_with_herma_preset_fits_27_per_page(conn, tmp_path):
    for release_id in range(1, 29):
        seed(conn, release_id)
    releases = [repository.get_release_detail(conn, i) for i in range(1, 29)]

    out_path = stickers.generate_sticker_sheet_pdf(
        releases,
        tmp_path / "covers",
        tmp_path / "waveforms",
        tmp_path / "stickers",
        preset=stickers.PRESETS["herma_4222"],
    )

    pdf_bytes = out_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    # 28 releases at 27/page (3x9) -> 2 pages, not 3 (would be 3 pages at
    # the Avery preset's 10/page).
    assert pdf_bytes.count(b"/Type /Page\n") == 2
