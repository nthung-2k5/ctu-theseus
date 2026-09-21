from theseus.services.image_size import ImageDimensions, read_image_dimensions

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def test_reads_width_and_height_from_png_ihdr():
    png = (
        PNG_SIGNATURE
        + bytes([0x00, 0x00, 0x00, 0x0D])  # IHDR chunk length (13)
        + b"IHDR"
        + bytes([0x00, 0x00, 0x00, 0x64])  # width = 100
        + bytes([0x00, 0x00, 0x00, 0x32])  # height = 50
    )
    assert read_image_dimensions(png) == ImageDimensions(width=100, height=50)


def test_reads_width_and_height_from_jpeg_sof0():
    jpeg = bytes(
        [0xFF, 0xD8]  # SOI
        + [0xFF, 0xC0]  # SOF0
        + [0x00, 0x0B]  # segment length
        + [0x08]  # precision
        + [0x00, 0x32]  # height = 50
        + [0x00, 0x64]  # width = 100
    )
    assert read_image_dimensions(jpeg) == ImageDimensions(width=100, height=50)


def test_skips_non_sof_jpeg_segments_such_as_app0():
    jpeg = bytes(
        [0xFF, 0xD8]
        + [0xFF, 0xE0, 0x00, 0x04, 0xAB, 0xCD]  # APP0, length 4 (2 content bytes)
        + [0xFF, 0xC0, 0x00, 0x0B, 0x08]
        + [0x00, 0x0A]  # height = 10
        + [0x00, 0x14]  # width = 20
    )
    assert read_image_dimensions(jpeg) == ImageDimensions(width=20, height=10)


def test_dht_marker_is_not_mistaken_for_a_frame_header():
    # 0xC4 is DHT: it must be skipped, not read as dimensions.
    jpeg = bytes(
        [0xFF, 0xD8] + [0xFF, 0xC4, 0x00, 0x04, 0x00, 0x00] + [0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x0A, 0x00, 0x14]
    )
    assert read_image_dimensions(jpeg) == ImageDimensions(width=20, height=10)


def test_unrecognized_format_returns_none():
    assert read_image_dimensions(bytes([0x00, 0x01, 0x02, 0x03])) is None
    assert read_image_dimensions(b"") is None


def test_truncated_png_returns_none():
    assert read_image_dimensions(PNG_SIGNATURE) is None


def test_jpeg_with_garbage_marker_returns_none():
    assert read_image_dimensions(bytes([0xFF, 0xD8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])) is None
