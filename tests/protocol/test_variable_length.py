"""Variable-length array registers, declared with ``max_length``."""

import struct

import numpy as np
import pytest
from harp.device.client._framer import HarpFramer
from harp.protocol._message import HarpMessage, HarpParseError
from harp.protocol._message_type import MessageType
from harp.protocol._payload_type import PayloadType
from harp.protocol._register import (
    ExtendedLengthRegister,
    RegisterU8Array,
    RegisterU16,
    RegisterU16Array,
)


class Tag(RegisterU8Array):
    address = 0x11
    max_length = 64


class FirmwareImage(RegisterU8Array, ExtendedLengthRegister):
    address = 0xC8
    max_length = 1_048_576


def _reply(register, payload: bytes, *, extended: bool = False) -> HarpMessage:
    return HarpMessage.parse(
        HarpMessage(
            MessageType.Read,
            register.address,
            register.payload_type,
            payload,
            timestamp=1.0,
            extended_length=extended,
        ).bytes
    )


# --------------------------------------------------------------------------
# Parsing and formatting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 17, 64])
def test_round_trip_any_count_up_to_max(count):
    values = np.arange(count, dtype=np.uint8)
    msg = HarpMessage.parse(Tag.format(values))
    assert len(msg.payload_bytes) == count
    decoded = msg.decode(Tag).payload
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, values)


def test_multi_byte_elements_and_conversion():
    register = RegisterU16Array(0x20, max_length=10)
    msg = HarpMessage.parse(register.format([1, 2, 65535]))
    assert msg.payload_bytes == struct.pack("<3H", 1, 2, 65535)
    decoded = msg.decode(register).payload
    assert decoded.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(decoded, [1, 2, 65535])


def test_format_rejects_more_than_max():
    with pytest.raises(ValueError, match="up to 64"):
        Tag.format(np.zeros(65, dtype=np.uint8))


def test_format_rejects_non_1d_values():
    with pytest.raises(ValueError, match="1-D"):
        Tag.format(np.zeros((2, 2), dtype=np.uint8))


def test_decode_rejects_more_than_max():
    with pytest.raises(HarpParseError, match="up to 64"):
        _reply(Tag, bytes(65)).decode(Tag)


def test_decode_rejects_partial_element():
    register = RegisterU16Array(0x20, max_length=10)
    with pytest.raises(HarpParseError, match="up to 10"):
        _reply(register, b"\x01\x00\x02").decode(register)


def test_parse_rejects_partial_element():
    register = RegisterU16Array(0x20, max_length=10)
    with pytest.raises(HarpParseError):
        register.parse(b"\x01\x00\x02")


def test_empty_read_request():
    msg = HarpMessage.parse(Tag.format())
    assert msg.message_type is MessageType.Read
    assert msg.payload_bytes == b""


# --------------------------------------------------------------------------
# Declaration
# --------------------------------------------------------------------------


def test_call_and_class_body_declarations_agree():
    one_off = RegisterU8Array(0x11, max_length=64)
    values = np.arange(5, dtype=np.uint8)
    assert one_off.format(values) == Tag.format(values)


def test_call_requires_exactly_one_size():
    with pytest.raises(TypeError, match="exactly one"):
        RegisterU8Array(0x11)
    with pytest.raises(TypeError, match="exactly one"):
        RegisterU8Array(0x11, length=4, max_length=8)


def test_class_body_cannot_declare_both():
    with pytest.raises(TypeError, match="both"):

        class _Both(RegisterU8Array):
            address = 0x11
            length = 4
            max_length = 8


def test_max_length_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        RegisterU8Array(0x11, max_length=0)


def test_cannot_resize_a_sized_register():
    with pytest.raises(TypeError, match="redeclares"):

        class _Resized(Tag):
            max_length = 32

    with pytest.raises(TypeError, match="redeclares"):

        class _Fixed(Tag):
            length = 4


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "register, extended",
    [
        (Tag, False),
        (RegisterU8Array(0x20, max_length=245), False),
        (RegisterU8Array(0x20, max_length=246), True),
        (RegisterU16Array(0x20, max_length=122), False),
        (RegisterU16Array(0x20, max_length=123), True),  # up to 246 bytes
        (FirmwareImage, True),
    ],
)
def test_framing_is_derived_from_the_maximum(register, extended):
    assert register.is_extended_length is extended


def test_small_message_of_extended_register_is_still_extended():
    # The framing belongs to the register, so a short payload is framed extended too.
    msg = HarpMessage.parse(FirmwareImage.format(np.arange(3, dtype=np.uint8)))
    assert msg.is_extended_length
    np.testing.assert_array_equal(msg.decode(FirmwareImage).payload, [0, 1, 2])


def test_large_extended_variable_round_trip():
    image = np.random.default_rng(0).integers(0, 256, 300_000, dtype=np.uint8)
    frame = FirmwareImage.format(image, message_type=MessageType.Event, timestamp=3.0)
    (msg,) = HarpFramer.parse_bytes(frame)
    assert msg.is_extended_length
    np.testing.assert_array_equal(msg.decode(FirmwareImage).payload, image)


def test_variable_frames_of_different_sizes_stream_back_to_back():
    frames = [Tag.format(np.arange(n, dtype=np.uint8)) for n in (0, 3, 64, 1)]
    msgs = HarpFramer.parse_bytes(b"".join(frames))
    assert [len(m.decode(Tag).payload) for m in msgs] == [0, 3, 64, 1]


# --------------------------------------------------------------------------
# Bulk paths are not supported yet
# --------------------------------------------------------------------------


def test_bulk_paths_raise():
    with pytest.raises(NotImplementedError):
        Tag.format_bulk(np.zeros((2, 4), dtype=np.uint8))
    with pytest.raises(NotImplementedError):
        Tag.parse_bulk(Tag.format(np.arange(4, dtype=np.uint8)))


def test_reply_payload_type_is_still_checked():
    msg = HarpMessage(MessageType.Read, Tag.address, PayloadType.U16, b"\x00\x00")
    with pytest.raises(HarpParseError, match="declares"):
        msg.decode(Tag)


@pytest.mark.parametrize(
    "register, variable",
    [
        (Tag, True),
        (FirmwareImage, True),
        (RegisterU16Array(0x20, max_length=10), True),
        (RegisterU8Array(0x20, length=10), False),
        (RegisterU8Array, False),  # an unsized base
        (RegisterU16(0x20), False),
    ],
)
def test_is_variable_length(register, variable):
    assert register.is_variable_length is variable


def test_max_length_is_readable_on_the_register():
    assert Tag.max_length == 64
    assert RegisterU16Array(0x20, max_length=10).max_length == 10
