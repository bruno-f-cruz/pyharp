"""Harp message container."""

import struct
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from typing_extensions import Sentinel

from ._builder import build_message_frame
from ._checksum import validate as _validate_checksum
from ._checksum import validate_crc32 as _validate_crc32
from ._constants import (
    _CHECKSUM_LEN,
    _CRC_LEN,
    _DEFAULT_PORT,
    _EXTENDED_HEADER_LEN,
    _EXTENDED_LENGTH_FLAG,
    _HEADER_LEN,
    _MIN_EXTENDED_FRAME_LEN,
    _MIN_FRAME_LEN,
    _TICK_PERIOD_S,
    _TIMESTAMP_FLAG,
    _TIMESTAMP_LEN,
)
from ._message_type import MessageType, _message_type_from_byte_safe
from ._payload import PayloadBase
from ._payload_type import PayloadType, decode_payload_type

P = TypeVar("P")
_P = TypeVar("_P")
_P_co = TypeVar("_P_co", covariant=True)

_UNDECODED = Sentinel("_UNDECODED")
"""Marks a message whose payload no register has decoded yet."""


class HarpParseError(Exception):
    """An exception raised for errors encountered during message parsing"""

    pass


_ADDRESS = _HEADER_LEN - 3
"""Byte offset of the address in a regular frame, after the type and the U8 length."""

_EXT_ADDRESS = _EXTENDED_HEADER_LEN - 3
"""Byte offset of the address in an extended-length frame, after the type and the U32
length."""


def _validate_regular_frame(raw: bytes) -> None:
    """Check the U8 length and U8 sum of a regular frame and its payload type byte."""
    if len(raw) < _MIN_FRAME_LEN:
        raise HarpParseError(f"Frame too short: {len(raw)} bytes (minimum {_MIN_FRAME_LEN})")

    if not _validate_checksum(raw):
        raise HarpParseError("Checksum mismatch")

    length = raw[1]
    if len(raw) != length + 2:
        raise HarpParseError(f"Length field {length} inconsistent with buffer size {len(raw)}")

    _validate_payload_type(raw[4], len(raw), _MIN_FRAME_LEN)


def _validate_extended_frame(raw: bytes) -> None:
    """Check the U32 length and CRC-32 of an extended-length frame and its payload type."""
    if len(raw) < _MIN_EXTENDED_FRAME_LEN:
        raise HarpParseError(
            f"Frame too short: {len(raw)} bytes (minimum {_MIN_EXTENDED_FRAME_LEN})"
        )

    if not _validate_crc32(raw):
        raise HarpParseError("CRC-32 checksum mismatch")

    length = int.from_bytes(raw[1:5], "little")
    if len(raw) != length + 5:
        raise HarpParseError(f"Length field {length} inconsistent with buffer size {len(raw)}")

    _validate_payload_type(raw[_EXTENDED_HEADER_LEN - 1], len(raw), _MIN_EXTENDED_FRAME_LEN)


def _validate_payload_type(pt_byte: int, frame_len: int, min_len: int) -> None:
    try:
        decode_payload_type(pt_byte)
    except ValueError as exc:
        raise HarpParseError(str(exc)) from exc

    if pt_byte & _TIMESTAMP_FLAG and frame_len < min_len + _TIMESTAMP_LEN:
        raise HarpParseError("Frame too short to contain timestamp")


class PayloadDecoder(Protocol[_P_co]):
    """Reads a payload of type ``_P_co`` out of a message.

    Structural rather than nominal, so a message never has to know about registers, and
    anything declaring a payload type, a payload class and a ``parse`` satisfies it. Every
    ``RegisterBase`` does. ``payload_class.payload_dtype`` is what fixes how many payload
    bytes the decoder consumes, and it is the same quantity ``parse`` reads the frame
    with, so the two cannot disagree about the extent of a payload.
    """

    payload_type: ClassVar["PayloadType"]
    payload_class: ClassVar[type[PayloadBase[Any]]]

    @classmethod
    def parse(cls, value: Any) -> _P_co: ...


class HarpMessage(Generic[P]):
    """A Harp message backed by its raw frame bytes, parameterized by its payload type.

    Build with the constructor or parse from wire bytes with ``HarpMessage.parse()``.
    A message off the wire is a ``HarpMessage[Any]``, since a frame declares only how
    its payload is encoded and not which register contract it satisfies. Decoding it
    with a register yields a ``HarpMessage[P]``, whose ``payload`` is that contract.
    """

    __slots__ = ("_bytes", "_payload")

    def __init__(
        self,
        message_type: MessageType,
        address: int,
        payload_type: PayloadType,
        payload_bytes: bytes = b"",
        *,
        port: int = _DEFAULT_PORT,
        timestamp: float | None = None,
        extended_length: bool | None = None,
    ) -> None:
        self._bytes: bytes = build_message_frame(
            message_type,
            address,
            payload_type,
            payload_bytes,
            port=port,
            timestamp=timestamp,
            extended_length=extended_length,
        )
        self._payload: P | _UNDECODED = _UNDECODED

    @classmethod
    def parse(cls, data: bytes | bytearray | memoryview) -> "HarpMessage[Any]":
        """Parse and validate a complete Harp Message from a byte sequence. Raises ``HarpParseError`` on failure."""
        raw = data if isinstance(data, bytes) else bytes(data)

        if not raw:
            raise HarpParseError("Frame is empty")

        # Validate MessageType byte (bits 7,6,5,2 must be 0; bits 1:0 are type). It is
        # read first, since its extended-length bit decides how the rest is framed.
        b0 = raw[0]
        if _message_type_from_byte_safe(b0) is None:
            raise HarpParseError(f"Invalid MessageType byte: 0x{b0:02x}")

        if b0 & _EXTENDED_LENGTH_FLAG:
            _validate_extended_frame(raw)
        else:
            _validate_regular_frame(raw)

        obj = cls.__new__(cls)
        obj._bytes = raw
        obj._payload = _UNDECODED
        return obj

    @property
    def message_type(self) -> MessageType:
        """Return the MessageType of this message."""
        return MessageType(self._bytes[0] & 0x03)

    @property
    def has_error(self) -> bool:
        """Return True if the error flag is set in this message."""
        return bool(self._bytes[0] & 0x08)

    @property
    def is_extended_length(self) -> bool:
        """Return True if this message uses extended-length framing.

        An extended-length frame has a U32 length and a CRC-32 checksum, in place of the
        U8 length and the U8 sum of a regular frame.
        """
        return bool(self._bytes[0] & _EXTENDED_LENGTH_FLAG)

    # The fields below sit 3 bytes further into an extended-length frame, whose length
    # field is a U32 rather than a U8. The flag is tested inline rather than through a
    # helper, since these are read for every message.

    @property
    def address(self) -> int:
        """Return the address byte of this message."""
        raw = self._bytes
        return raw[_EXT_ADDRESS] if raw[0] & _EXTENDED_LENGTH_FLAG else raw[_ADDRESS]

    @property
    def port(self) -> int:
        """Return the port byte of this message."""
        raw = self._bytes
        return raw[_EXT_ADDRESS + 1] if raw[0] & _EXTENDED_LENGTH_FLAG else raw[_ADDRESS + 1]

    @property
    def payload_type(self) -> PayloadType:
        """Return the PayloadType of this message."""
        raw = self._bytes
        pt_byte = raw[_EXT_ADDRESS + 2] if raw[0] & _EXTENDED_LENGTH_FLAG else raw[_ADDRESS + 2]
        return decode_payload_type(pt_byte).payload_type

    @property
    def has_timestamp(self) -> bool:
        """Return True if the timestamp flag is set in this message."""
        raw = self._bytes
        pt_byte = raw[_EXT_ADDRESS + 2] if raw[0] & _EXTENDED_LENGTH_FLAG else raw[_ADDRESS + 2]
        return bool(pt_byte & _TIMESTAMP_FLAG)

    @property
    def timestamp(self) -> float | None:
        """Return the timestamp of this message, or None if not present."""
        raw = self._bytes
        offset = _EXTENDED_HEADER_LEN if raw[0] & _EXTENDED_LENGTH_FLAG else _HEADER_LEN
        if not raw[offset - 1] & _TIMESTAMP_FLAG:
            return None
        seconds, microseconds = struct.unpack_from("<IH", raw, offset)
        return cast(int, seconds) + cast(int, microseconds) * _TICK_PERIOD_S

    @property
    def payload_bytes(self) -> memoryview:
        """Payload bytes, excluding timestamp and checksum."""
        raw = self._bytes
        if raw[0] & _EXTENDED_LENGTH_FLAG:
            offset, end = _EXTENDED_HEADER_LEN, -_CRC_LEN
        else:
            offset, end = _HEADER_LEN, -_CHECKSUM_LEN
        if raw[offset - 1] & _TIMESTAMP_FLAG:
            offset += _TIMESTAMP_LEN
        return memoryview(raw)[offset:end]

    @property
    def checksum(self) -> int:
        """Return the checksum field of this message.

        That is the U8 sum of a regular frame, or the CRC-32 of an extended-length one.
        """
        raw = self._bytes
        if raw[0] & _EXTENDED_LENGTH_FLAG:
            return int.from_bytes(raw[-_CRC_LEN:], "little")
        return raw[-1]

    @property
    def has_payload(self) -> bool:
        """Return True if a register has decoded the payload of this message."""
        return self._payload is not _UNDECODED

    @property
    def payload(self) -> P:
        """The decoded payload, as the register that parsed this message defines it.

        Only a register knows which contract a frame satisfies, so a message read from
        the wire carries no payload until one decodes it. Raises ``ValueError`` in that
        case; test with ``has_payload`` first, or read ``payload_bytes`` instead.
        """
        if self._payload is _UNDECODED:
            raise ValueError(
                "No register has decoded this message, so it has no payload. "
                "Parse it with a register, or read payload_bytes instead."
            )
        return self._payload

    def decode(self, decoder: type[PayloadDecoder[_P]]) -> "HarpMessage[_P]":
        """Return a copy of this message with its payload decoded by ``decoder``.

        The payload is derived from the frame in the same call, so the two cannot
        disagree. The payload type and the byte count are both checked, since together
        they decide whether these bytes can be read as this payload at all. The address
        is not, so a frame may be decoded by anything describing the same layout.
        """
        if self.payload_type is not decoder.payload_type:
            raise HarpParseError(
                f"{decoder.__name__} declares {decoder.payload_type!r} but this "
                f"message declares {self.payload_type!r}."
            )
        payload_class = decoder.payload_class
        actual = len(self.payload_bytes)
        # The fixed case is compared inline, since every reply and event is decoded.
        if payload_class._max_length is None:
            accepted = actual == payload_class.payload_dtype.itemsize
        else:
            accepted = payload_class._accepts_payload_size(actual)
        if not accepted:
            itemsize = payload_class.payload_dtype.itemsize
            if payload_class._max_length is None:
                expected = f"{itemsize} payload bytes"
            else:
                expected = f"up to {payload_class._max_length} elements of {itemsize} payload bytes"
            raise HarpParseError(
                f"{decoder.__name__} reads {expected} but this message carries {actual}."
            )
        obj: HarpMessage[_P] = HarpMessage.__new__(HarpMessage)
        obj._bytes = self._bytes
        obj._payload = decoder.parse(self)
        return obj

    @property
    def bytes(self) -> bytes:
        """The complete raw message frame, including checksum."""
        return self._bytes

    def __str__(self) -> str:
        return (
            f"HarpMessage(message_type={self.message_type!r}, address={self.address:#04x}, "
            f"payload_type={self.payload_type!r}, timestamp={self.timestamp!r})"
        )
