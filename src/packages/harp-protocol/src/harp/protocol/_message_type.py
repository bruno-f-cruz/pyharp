from enum import IntEnum

from ._constants import _EXTENDED_LENGTH_FLAG


class MessageType(IntEnum):
    """Represents the a message type from the harp protocol"""

    Read = 1
    Write = 2
    Event = 3


_RESERVED_MASK = 0b11100100
"""Bits 7, 6, 5 and 2 must be 0. Bit 4 is extended length, bit 3 is error and bits 1:0
are the type."""

_VALID_TYPES = frozenset(t.value for t in MessageType)


def _message_type_from_byte_safe(b: int) -> "tuple[MessageType, bool] | None":
    """Decode a MessageType byte into ``(MessageType, has_error)``, or ``None`` if invalid."""
    if b & _RESERVED_MASK:
        return None
    type_bits = b & 0x03
    if type_bits not in _VALID_TYPES:
        return None
    return MessageType(type_bits), bool(b & 0x08)


def message_type_from_byte(b: int) -> tuple["MessageType", bool]:
    """Decode a MessageType byte into ``(MessageType, has_error)``. Raises ``ValueError`` on invalid input.

    The framing the byte selects is read separately, with :func:`is_extended_length`.
    """
    result = _message_type_from_byte_safe(b)
    if result is None:
        type_bits = b & 0x03
        if b & _RESERVED_MASK:
            raise ValueError(f"Reserved bits set in MessageType byte: 0x{b:02x}")
        raise ValueError(f"Invalid MessageType value {type_bits} in byte: 0x{b:02x}")
    return result


def message_type_to_byte(
    message_type: MessageType, has_error: bool = False, *, extended_length: bool = False
) -> int:
    """Encode MessageType + error flag + extended-length flag to a single byte."""
    return (
        message_type.value
        | (0x08 if has_error else 0)
        | (_EXTENDED_LENGTH_FLAG if extended_length else 0)
    )


def is_extended_length(b: int) -> bool:
    """Return True if a MessageType byte selects extended-length framing."""
    return bool(b & _EXTENDED_LENGTH_FLAG)
