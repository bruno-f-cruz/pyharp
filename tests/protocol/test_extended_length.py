"""Extended-length framing, harp-tech/protocol#218."""

import struct
import zlib

import numpy as np
import pytest
from harp.device.client._framer import HarpFramer
from harp.protocol._builder import build_message_frame
from harp.protocol._checksum import compute_crc32, validate_crc32
from harp.protocol._message import HarpMessage, HarpParseError
from harp.protocol._message_type import (
    MessageType,
    is_extended_length,
    message_type_from_byte,
    message_type_to_byte,
)
from harp.protocol._payload_type import PayloadType
from harp.protocol._payload import Field, StructPayload, _IdentityConverter
from harp.protocol._register import (
    ExtendedLengthRegister,
    ExtendedMessageReceipt,
    RegisterBase,
    RegisterU8,
    RegisterU8Array,
    RegisterU16Array,
    RegisterU32Array,
)

from tests.fixtures import TIMESTAMP_1S, make_extended_frame_from_raw

# --------------------------------------------------------------------------
# CRC-32
# --------------------------------------------------------------------------


def test_crc32_check_value():
    # CRC-32/ISO-HDLC check value, as given by the proposal. The CRC field itself is
    # excluded, so append four placeholder bytes.
    assert compute_crc32(b"123456789" + b"\x00" * 4) == 0xCBF43926


def test_validate_crc32():
    data = b"123456789"
    assert validate_crc32(data + struct.pack("<I", 0xCBF43926))
    assert not validate_crc32(data + struct.pack("<I", 0xCBF43927))
    assert not validate_crc32(b"\x00" * 4)  # nothing to check


# --------------------------------------------------------------------------
# MessageType byte
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "byte, expected_type, expected_error",
    [
        (0x11, MessageType.Read, False),
        (0x12, MessageType.Write, False),
        (0x13, MessageType.Event, False),
        (0x19, MessageType.Read, True),
        (0x1A, MessageType.Write, True),
        (0x1B, MessageType.Event, True),
    ],
)
def test_extended_message_type_bytes_are_valid(byte, expected_type, expected_error):
    assert message_type_from_byte(byte) == (expected_type, expected_error)
    assert is_extended_length(byte)


@pytest.mark.parametrize("byte", [0x10, 0x14, 0x31, 0x51, 0x91])
def test_extended_flag_does_not_unreserve_other_bits(byte):
    with pytest.raises(ValueError):
        message_type_from_byte(byte)


def test_message_type_to_byte_extended():
    assert message_type_to_byte(MessageType.Write, extended_length=True) == 0x12
    assert message_type_to_byte(MessageType.Write, True, extended_length=True) == 0x1A
    assert not is_extended_length(message_type_to_byte(MessageType.Write))


# --------------------------------------------------------------------------
# Building and parsing frames
# --------------------------------------------------------------------------


def test_empty_extended_read_is_twelve_bytes():
    # An empty read is valid in extended framing: a U32 length of 7, covering
    # address, port, payload type and the CRC.
    header = bytes([0x11]) + struct.pack("<I", 7) + bytes([0x20, 0xFF, 0x02])
    expected = header + struct.pack("<I", zlib.crc32(header))
    frame = build_message_frame(MessageType.Read, 0x20, PayloadType.U16, extended_length=True)
    assert frame == expected
    assert len(frame) == 12

    msg = HarpMessage.parse(frame)
    assert msg.is_extended_length
    assert msg.message_type is MessageType.Read
    assert msg.address == 0x20
    assert msg.port == 0xFF
    assert msg.payload_type is PayloadType.U16
    assert msg.payload_bytes == b""
    assert msg.timestamp is None
    assert msg.checksum == zlib.crc32(header)


def test_builder_matches_hand_built_frame_with_timestamp():
    payload = bytes(range(256)) * 4
    frame = build_message_frame(
        MessageType.Event, 0x40, PayloadType.U8, payload, timestamp=1.0, extended_length=True
    )
    assert frame == make_extended_frame_from_raw(
        0x13, 0x40, 0xFF, 0x01, payload, timestamp=TIMESTAMP_1S
    )

    msg = HarpMessage.parse(frame)
    assert msg.is_extended_length
    assert msg.has_timestamp
    assert msg.timestamp == pytest.approx(1.0)
    assert msg.payload_bytes == payload


def test_builder_picks_framing_from_size_when_not_told():
    small = build_message_frame(MessageType.Write, 1, PayloadType.U8, b"\x00" * 251)
    large = build_message_frame(MessageType.Write, 1, PayloadType.U8, b"\x00" * 252)
    assert not HarpMessage.parse(small).is_extended_length
    assert HarpMessage.parse(large).is_extended_length


def test_builder_refuses_oversized_regular_frame():
    with pytest.raises(ValueError, match="extended_length=True"):
        build_message_frame(
            MessageType.Write, 1, PayloadType.U8, b"\x00" * 252, extended_length=False
        )


def test_small_payload_in_extended_framing():
    # The framing is the register's choice, so a small payload may still be extended.
    frame = build_message_frame(MessageType.Write, 1, PayloadType.U8, b"\x05", extended_length=True)
    msg = HarpMessage.parse(frame)
    assert msg.is_extended_length
    assert msg.payload_bytes == b"\x05"


def test_message_constructor_takes_framing():
    msg = HarpMessage(MessageType.Write, 1, PayloadType.U8, b"\x05", extended_length=True)
    assert msg.is_extended_length
    assert HarpMessage.parse(msg.bytes).payload_bytes == b"\x05"


def test_parse_rejects_corrupt_crc():
    frame = bytearray(make_extended_frame_from_raw(0x13, 0x40, 0xFF, 0x01, b"\x01\x02\x03"))
    frame[-1] ^= 0xFF
    with pytest.raises(HarpParseError, match="CRC"):
        HarpMessage.parse(bytes(frame))


def test_parse_rejects_corrupt_payload_under_crc():
    # A byte swap leaves a U8 sum unchanged but not a CRC-32.
    frame = bytearray(make_extended_frame_from_raw(0x13, 0x40, 0xFF, 0x01, b"\x01\x02\x03"))
    frame[8], frame[9] = frame[9], frame[8]
    with pytest.raises(HarpParseError, match="CRC"):
        HarpMessage.parse(bytes(frame))


def test_parse_rejects_length_mismatch():
    frame = bytearray(make_extended_frame_from_raw(0x13, 0x40, 0xFF, 0x01, b"\x01\x02\x03"))
    frame[1:5] = struct.pack("<I", 99)
    frame[-4:] = struct.pack("<I", zlib.crc32(frame[:-4]))
    with pytest.raises(HarpParseError, match="Length"):
        HarpMessage.parse(bytes(frame))


def test_parse_rejects_truncated_extended_frame():
    frame = make_extended_frame_from_raw(0x11, 0x40, 0xFF, 0x01, b"")
    with pytest.raises(HarpParseError, match="too short"):
        HarpMessage.parse(frame[:-1])


def test_regular_frame_is_not_extended():
    msg = HarpMessage(MessageType.Write, 1, PayloadType.U8, b"\x05")
    assert not msg.is_extended_length
    assert msg.checksum == msg.bytes[-1]


# --------------------------------------------------------------------------
# Registers
# --------------------------------------------------------------------------


class Waveform(RegisterU16Array, ExtendedLengthRegister):
    address = 0x64
    length = 4096


@pytest.mark.parametrize(
    "register, extended",
    [
        (RegisterU8(0x20), False),
        (RegisterU8Array(0x20, length=245), False),  # 245 + timestamp fits in a U8 length
        (RegisterU8Array(0x20, length=246), True),
        (RegisterU16Array(0x20, length=122), False),  # 244 bytes
        (RegisterU16Array(0x20, length=123), True),  # 246 bytes
        (RegisterU32Array(0x20, length=61), False),  # 244 bytes
        (RegisterU32Array(0x20, length=62), True),  # 248 bytes
    ],
)
def test_register_framing_is_derived_from_payload_size(register, extended):
    assert register.is_extended_length is extended


def test_marker_does_not_decide_framing():
    # The marker is for type checkers only; the size decides the framing.
    assert Waveform.is_extended_length is True
    assert RegisterU16Array(0x64, length=4096).is_extended_length is True


def test_struct_register_framing_is_derived():
    class Big(StructPayload, length=300):
        Data = Field(converter=_IdentityConverter("u1"), offset=0)

    class BigRegister(RegisterBase[Big]):
        address = 0x30
        payload_type = PayloadType.U8
        payload_class = Big

    assert BigRegister.is_extended_length is True


def test_largest_regular_register_fits_with_timestamp():
    register = RegisterU8Array(0x20, length=245)
    frame = register.format(np.zeros(245, dtype=np.uint8), timestamp=1.0)
    assert not HarpMessage.parse(frame).is_extended_length


def test_extended_register_frames_every_message_extended():
    register = Waveform
    values = np.arange(4096, dtype=np.uint16)

    read = HarpMessage.parse(register.format())
    assert read.is_extended_length
    assert read.message_type is MessageType.Read
    assert read.payload_bytes == b""

    write = HarpMessage.parse(register.format(values))
    assert write.is_extended_length
    assert len(write.payload_bytes) == 8192

    event = HarpMessage.parse(
        register.format(values, message_type=MessageType.Event, timestamp=2.5)
    )
    assert event.is_extended_length
    assert event.timestamp == pytest.approx(2.5)
    np.testing.assert_array_equal(event.decode(register).payload, values)


def test_format_bulk_extended_matches_format():
    register = RegisterU16Array(0x64, length=200)
    values = np.arange(3 * 200, dtype=np.uint16).reshape(3, 200)
    timestamps = [1.0, 2.0, 3.0]
    bulk = register.format_bulk(values, timestamps=timestamps, message_type=MessageType.Event)
    expected = b"".join(
        register.format(row, message_type=MessageType.Event, timestamp=ts)
        for row, ts in zip(values, timestamps)
    )
    assert bulk.tobytes() == expected

    msgs = HarpFramer.parse_bytes(bulk.tobytes())
    assert len(msgs) == 3
    for msg, row in zip(msgs, values):
        assert msg.is_extended_length
        np.testing.assert_array_equal(msg.decode(register).payload, row)


def test_format_bulk_regular_clears_extended_flag_of_given_types():
    register = RegisterU8(0x20)
    bulk = register.format_bulk([1, 2], message_type=[0x13, 0x03])
    msgs = HarpFramer.parse_bytes(bulk.tobytes())
    assert [m.is_extended_length for m in msgs] == [False, False]


def test_parse_bulk_rejects_extended_frames():
    register = RegisterU16Array(0x64, length=200)
    bulk = register.format_bulk(np.zeros((2, 200), dtype=np.uint16))
    with pytest.raises(NotImplementedError):
        register.parse_bulk(bulk.tobytes())
    # Also when the frames are extended but the register is not, as a file would be.
    with pytest.raises(NotImplementedError):
        RegisterU16Array(0x64, length=2).parse_bulk(bulk.tobytes())


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def test_receipt_decodes_a_write_reply():
    reply = HarpMessage(
        MessageType.Write, 0x64, PayloadType.U32, struct.pack("<I", 0xDEADBEEF), timestamp=1.0
    )
    typed = reply.decode(ExtendedMessageReceipt)
    assert typed.payload == ExtendedMessageReceipt(crc=0xDEADBEEF)
    assert typed.timestamp == pytest.approx(1.0)
    assert not typed.is_extended_length


def test_receipt_rejects_other_payload_types():
    reply = HarpMessage(MessageType.Write, 0x64, PayloadType.U16, bytes(2))
    with pytest.raises(HarpParseError):
        reply.decode(ExtendedMessageReceipt)
