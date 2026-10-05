"""Package-wide Harp protocol constants."""

_TICK_PERIOD_S: float = 32e-6
"""Harp timestamp clock tick period in seconds, 32 microseconds per tick."""

_TIMESTAMP_FLAG: int = 0x10
"""PayloadType byte bit that signals a timestamp is present in the frame."""

_EXTENDED_LENGTH_FLAG: int = 0x10
"""MessageType byte bit that signals ``ExtendedLength`` framing: a U32 ``Length`` and a
CRC-32 ``Checksum`` in place of the U8 length and the U8 sum.

The same value as :data:`_TIMESTAMP_FLAG`, but a bit of a different byte.
"""

_DEFAULT_PORT: int = 0xFF
"""Default Harp port value, meaning broadcast."""

_HEADER_LEN: int = 5
"""Fixed header size in bytes of a regular frame.

That is msg_type + length + address + port + payload_type.
"""

_EXTENDED_HEADER_LEN: int = 8
"""Fixed header size in bytes of an extended-length frame, whose length field is a U32."""

_CHECKSUM_LEN: int = 1
"""Checksum size in bytes of a regular frame, a U8 sum."""

_CRC_LEN: int = 4
"""Checksum size in bytes of an extended-length frame, a CRC-32."""

_MIN_FRAME_LEN: int = 6
"""Smallest regular frame on the wire in bytes, the fixed header plus the checksum."""

_MIN_EXTENDED_FRAME_LEN: int = _EXTENDED_HEADER_LEN + _CRC_LEN
"""Smallest extended-length frame on the wire in bytes, the fixed header plus the CRC."""

_MAX_REGULAR_LENGTH: int = 0xFF
"""Largest value of the U8 length field, which counts every byte after itself."""

_MAX_FRAME_LEN: int = _MAX_REGULAR_LENGTH + 2
"""Largest regular frame on the wire in bytes, the U8 length plus the two bytes before it."""

_TIMESTAMP_LEN: int = 6
"""Timestamp field size in bytes: 4-byte seconds as u32 plus 2-byte microseconds as u16."""

_MAX_REGULAR_PAYLOAD_LEN: int = _MAX_REGULAR_LENGTH - 3 - _TIMESTAMP_LEN - _CHECKSUM_LEN
"""Largest payload in bytes that a regular frame carries with a timestamp.

A register whose payload can exceed it uses extended-length framing for every message,
whether or not a given message carries a timestamp, since any of them may.
"""

_TS_MICROS_OFFSET: int = 9
"""Byte offset of the timestamp microseconds field, which is ``_HEADER_LEN + 4``."""

_TIMESTAMPED_PAYLOAD_OFFSET: int = 11
"""Byte offset of the payload when a timestamp is present, ``_HEADER_LEN + _TIMESTAMP_LEN``."""
