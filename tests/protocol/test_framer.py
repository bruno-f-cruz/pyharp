import struct

import pytest
from harp.device.client._framer import HarpFramer
from harp.protocol._message_type import MessageType

from tests.fixtures import TIMESTAMP_1S, make_extended_frame_from_raw, make_frame_from_raw


def test_single_message():
    frame = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x01")
    msgs = HarpFramer.parse_bytes(frame)
    assert len(msgs) == 1
    assert msgs[0].address == 10
    assert msgs[0].payload_bytes == b"\x01"


def test_back_to_back_messages():
    f1 = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    f2 = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    f3 = make_frame_from_raw(0x03, 32, 0xFF, 0x11, b"\x7f", timestamp=TIMESTAMP_1S)
    msgs = HarpFramer.parse_bytes(f1 + f2 + f3)
    assert len(msgs) == 3
    assert msgs[0].message_type == MessageType.Read
    assert msgs[1].message_type == MessageType.Write
    assert msgs[2].message_type == MessageType.Event


def test_garbage_prefix_skipped():
    garbage = bytes([0x00, 0x05, 0xFF, 0x20, 0x00])
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    msgs = HarpFramer.parse_bytes(garbage + frame)
    assert len(msgs) == 1
    assert msgs[0].address == 8


def test_garbage_between_messages():
    f1 = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    f2 = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    noise = bytes([0xAA, 0xBB, 0xCC])
    msgs = HarpFramer.parse_bytes(f1 + noise + f2)
    assert len(msgs) == 2


def test_bad_checksum_skipped_recovery():
    # A frame with a bad checksum should be skipped; the next valid frame parses.
    bad = bytearray(make_frame_from_raw(0x01, 8, 0xFF, 0x04, b""))
    bad[-1] ^= 0xFF  # corrupt checksum
    good = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    msgs = HarpFramer.parse_bytes(bytes(bad) + good)
    assert len(msgs) == 1
    assert msgs[0].address == 10


def test_truncated_stream_returns_empty():
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    # Only the first 3 bytes, not enough for a complete frame.
    msgs = HarpFramer.parse_bytes(frame[:3])
    assert msgs == []


def test_incremental_feed():
    # Feeding data in small chunks still yields the complete message.
    frame = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x42")
    framer = HarpFramer()
    results = []
    for byte in frame:
        framer.feed(bytes([byte]))
        results.extend(framer.frames())
    assert len(results) == 1
    assert results[0].payload_bytes == b"\x42"


def test_all_scalar_types():
    # Framer correctly parses messages with each PayloadType.
    from harp.protocol._payload_type import PayloadType, encode_payload_type

    for pt in PayloadType:
        size = pt.numpy_dtype.itemsize
        payload = bytes(range(size))
        pt_byte = encode_payload_type(pt)
        frame = make_frame_from_raw(0x03, 32, 0xFF, pt_byte, payload)
        msgs = HarpFramer.parse_bytes(frame)
        assert len(msgs) == 1, f"Failed for {pt}"
        assert msgs[0].payload_type == pt


def test_array_payload():
    payload = struct.pack("<" + "H" * 5, *range(5))
    frame = make_frame_from_raw(0x03, 32, 0xFF, 0x02, payload)
    msgs = HarpFramer.parse_bytes(frame)
    assert len(msgs) == 1
    assert len(msgs[0].payload_bytes) == 10


def test_parse_file(tmp_path):
    frame = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    p = tmp_path / "test.bin"
    p.write_bytes(frame)
    msgs = HarpFramer.parse_file(p)
    assert len(msgs) == 1
    assert len(msgs) == 1


# --------------------------------------------------------------------------
# Extended-length frames
# --------------------------------------------------------------------------


def _extended_event(payload: bytes, address: int = 64) -> bytes:
    return make_extended_frame_from_raw(0x13, address, 0xFF, 0x01, payload, timestamp=TIMESTAMP_1S)


def test_extended_frame_parses_from_complete_buffer():
    frame = _extended_event(bytes(1000))
    msgs = HarpFramer.parse_bytes(frame)
    assert len(msgs) == 1
    assert msgs[0].is_extended_length
    assert len(msgs[0].payload_bytes) == 1000


def test_streaming_framer_rejects_extended_frames_by_default():
    framer = HarpFramer()
    framer.feed(_extended_event(bytes(1000)) + make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05"))
    msgs = list(framer.frames())
    assert [m.address for m in msgs] == [10]


def test_streaming_framer_accepts_extended_frames_within_bound():
    frame = _extended_event(bytes(1000))
    framer = HarpFramer(max_frame_length=len(frame))
    framer.feed(frame)
    assert [len(m.payload_bytes) for m in framer.frames()] == [1000]


def test_max_frame_length_cannot_reject_regular_frames():
    with pytest.raises(ValueError):
        HarpFramer(max_frame_length=100)


def test_extended_frame_fed_byte_by_byte():
    # Covers the U32 length arriving split across chunks.
    frame = _extended_event(bytes(range(256)) * 2)
    framer = HarpFramer(max_frame_length=len(frame))
    results = []
    for byte in frame:
        framer.feed(bytes([byte]))
        results.extend(framer.frames())
    assert len(results) == 1
    assert results[0].payload_bytes == bytes(range(256)) * 2


def test_regular_and_extended_frames_interleaved():
    f1 = make_frame_from_raw(0x01, 8, 0xFF, 0x04, b"")
    f2 = _extended_event(bytes(300), address=64)
    f3 = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    f4 = _extended_event(bytes(500), address=65)
    framer = HarpFramer(max_frame_length=1024)
    framer.feed(f1 + f2 + f3 + f4)
    msgs = list(framer.frames())
    assert [m.address for m in msgs] == [8, 64, 10, 65]
    assert [m.is_extended_length for m in msgs] == [False, True, False, True]


def test_false_extended_start_beyond_bound_does_not_stall_stream():
    # A stray byte that reads as an extended Event, followed by a length claiming
    # gigabytes, must not hold back the frame after it.
    noise = bytes([0x13]) + struct.pack("<I", 0xFFFFFF00)
    good = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    framer = HarpFramer(max_frame_length=1 << 20)
    framer.feed(noise + good)
    assert [m.address for m in framer.frames()] == [10]


def test_false_extended_start_in_complete_buffer_is_skipped():
    # In a complete buffer, a claimed frame running past the end cannot be completed,
    # so it is skipped rather than ending the parse.
    noise = bytes([0x13]) + struct.pack("<I", 5000)
    good = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    msgs = HarpFramer.parse_bytes(noise + good)
    assert [m.address for m in msgs] == [10]


def test_corrupt_extended_crc_recovers_next_frame():
    bad = bytearray(_extended_event(bytes(300)))
    bad[-1] ^= 0xFF
    good = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    framer = HarpFramer(max_frame_length=1024)
    framer.feed(bytes(bad) + good)
    assert [m.address for m in framer.frames()] == [10]


def test_extended_length_too_short_is_a_false_start():
    noise = bytes([0x13]) + struct.pack("<I", 3)
    good = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    framer = HarpFramer(max_frame_length=1024)
    framer.feed(noise + good)
    assert [m.address for m in framer.frames()] == [10]


def test_truncated_regular_tail_still_ends_complete_parse():
    # A recording cut mid-frame ends the parse at the cut, as it did before extended
    # framing, rather than rescanning the tail for frames.
    f1 = make_frame_from_raw(0x02, 10, 0xFF, 0x01, b"\x05")
    f2 = make_frame_from_raw(0x02, 11, 0xFF, 0x01, b"\x06")
    msgs = HarpFramer.parse_bytes(f1 + f2[:-1])
    assert [m.address for m in msgs] == [10]
