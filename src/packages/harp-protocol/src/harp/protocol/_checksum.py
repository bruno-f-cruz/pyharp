import zlib

from ._constants import _CRC_LEN


def compute(data: bytes | bytearray | memoryview) -> int:
    """Wrapping u8 sum of all bytes except the last (the checksum byte itself)."""
    return sum(memoryview(data)[:-1]) & 0xFF


def validate(data: bytes | bytearray | memoryview) -> bool:
    """Return True if the last byte equals the checksum of all preceding bytes."""
    mv = memoryview(data)
    if len(mv) < 2:
        return False
    return compute(mv) == mv[-1]


def compute_crc32(data: bytes | bytearray | memoryview) -> int:
    """CRC-32 of all bytes except the last four (the CRC field itself).

    The CRC is CRC-32/ISO-HDLC, the IEEE 802.3 polynomial with reflected input and
    output, which is what :func:`zlib.crc32` computes.
    """
    return zlib.crc32(memoryview(data)[:-_CRC_LEN])


def validate_crc32(data: bytes | bytearray | memoryview) -> bool:
    """Return True if the last four bytes, read as a little-endian U32, equal the CRC-32
    of all preceding bytes."""
    mv = memoryview(data)
    if len(mv) <= _CRC_LEN:
        return False
    return compute_crc32(mv) == int.from_bytes(mv[-_CRC_LEN:], "little")
