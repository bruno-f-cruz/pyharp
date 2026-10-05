"""Utilities for building outgoing Harp message frames."""

import struct
import zlib

from ._constants import _CHECKSUM_LEN, _CRC_LEN, _DEFAULT_PORT, _MAX_REGULAR_LENGTH, _TICK_PERIOD_S
from ._message_type import MessageType
from ._message_type import message_type_to_byte as _msg_type_byte
from ._payload_type import PayloadType, encode_payload_type

_MAX_EXTENDED_LENGTH = 0xFFFFFFFF
"""Largest value of the U32 length field of an extended-length frame."""


def build_message_frame(
    message_type: MessageType,
    address: int,
    payload_type: PayloadType,
    payload: bytes = b"",
    *,
    port: int = _DEFAULT_PORT,
    timestamp: float | None = None,
    extended_length: bool | None = None,
) -> bytes:
    """Build and return a complete Harp wire frame as bytes.

    ``extended_length`` selects the framing. ``True`` builds an extended-length frame,
    with a U32 length and a CRC-32, and ``False`` a regular one, with a U8 length and a
    U8 sum. ``None`` builds a regular frame unless the body does not fit in one. A
    register always passes the framing it declares, since that is fixed per register
    rather than chosen by the size of one payload.
    """
    if timestamp is not None:
        seconds = int(timestamp)
        microseconds = round((timestamp - seconds) / _TICK_PERIOD_S)
        ts_bytes = struct.pack("<IH", seconds, microseconds)
    else:
        ts_bytes = b""
    pt_byte = encode_payload_type(payload_type, has_timestamp=timestamp is not None)
    body = bytes([address, port, pt_byte]) + ts_bytes + payload
    if extended_length is None:
        extended_length = len(body) + _CHECKSUM_LEN > _MAX_REGULAR_LENGTH

    if extended_length:
        length = len(body) + _CRC_LEN
        if length > _MAX_EXTENDED_LENGTH:
            raise ValueError(
                f"A {len(payload)}-byte payload does not fit in an extended-length frame."
            )
        frame = bytearray([_msg_type_byte(message_type, extended_length=True)])
        frame += struct.pack("<I", length)
        frame += body
        frame += struct.pack("<I", zlib.crc32(frame))
        return bytes(frame)

    length = len(body) + _CHECKSUM_LEN
    if length > _MAX_REGULAR_LENGTH:
        raise ValueError(
            f"A {len(payload)}-byte payload does not fit in a regular frame; "
            "build it with extended_length=True."
        )
    header = bytes([_msg_type_byte(message_type), length])
    frame = header + body
    checksum = sum(frame) & 0xFF
    return frame + bytes([checksum])
