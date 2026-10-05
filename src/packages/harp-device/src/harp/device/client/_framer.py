from collections.abc import Iterator
from pathlib import Path

from harp.protocol._constants import _CRC_LEN, _EXTENDED_LENGTH_FLAG, _MAX_FRAME_LEN
from harp.protocol._message import HarpMessage, HarpParseError
from harp.protocol._message_type import message_type_from_byte as _validate_message_type

_EXTENDED_LENGTH_FIELD_LEN = 4
"""Size in bytes of the U32 length field of an extended-length frame."""

_MIN_EXTENDED_LENGTH = 3 + _CRC_LEN
"""Smallest valid length of an extended-length frame: address, port, payload type, CRC."""


class HarpFramer:
    """Stateful Harp message stream parser.

    Feed raw bytes incrementally with feed(), then drain complete frames with
    next_frame() or by iterating. Suitable for both file parsing and streaming
    sources (e.g. serial ports) where data arrives in chunks.

    Recovery: on checksum or PayloadType failure, the framer skips exactly the
    bad MessageType byte and retries from the next byte, matching the C#
    StreamTransport resynchronisation strategy.

    ``max_frame_length`` bounds the frames the framer waits for, in bytes. A regular
    frame never exceeds it, since the default is the largest regular frame, so only
    extended-length frames are affected. An extended-length frame declares a U32
    length, and a byte that only looks like the start of one, such as noise while
    resynchronising, can declare gigabytes and hold every later frame back until that
    many bytes arrive. A declared length beyond the bound is treated as a false start
    instead. The default accepts no extended-length frame at all; raise it to the
    largest frame the device can send.
    """

    def __init__(self, max_frame_length: int = _MAX_FRAME_LEN) -> None:
        if max_frame_length < _MAX_FRAME_LEN:
            raise ValueError(
                f"max_frame_length must be at least {_MAX_FRAME_LEN} bytes, the largest "
                f"regular frame, but was {max_frame_length}."
            )
        self.max_frame_length = max_frame_length
        """The largest frame, in bytes, that the framer waits for."""
        self._buf: bytearray = bytearray()
        self._pos: int = 0
        # Set when the buffer holds the whole stream, so an extended-length frame
        # running past its end can never complete, and is a false start rather than
        # one still arriving.
        self._complete: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, data: bytes | bytearray) -> None:
        """Append new bytes to the internal buffer."""
        self._buf.extend(data)
        # Compact once we've consumed a decent chunk to avoid unbounded growth.
        if self._pos > 4096:
            del self._buf[: self._pos]
            self._pos = 0

    def next_frame(self) -> HarpMessage | None:
        """Return the next complete, valid HarpMessage, or None if not enough data."""
        buf = self._buf
        pos = self._pos

        while pos < len(buf):
            # --- State 1: Seek ----------------------------------------------
            # Find a byte that looks like a valid MessageType.
            try:
                _validate_message_type(buf[pos])
            except ValueError:
                pos += 1
                continue

            msg_type_pos = pos

            # --- State 2: ReadLength ---------------------------------------
            if buf[pos] & _EXTENDED_LENGTH_FLAG:
                length_end = pos + 1 + _EXTENDED_LENGTH_FIELD_LEN
                if length_end > len(buf):
                    if self._complete:
                        pos += 1
                        continue
                    break  # need more data
                length = int.from_bytes(buf[pos + 1 : length_end], "little")
                frame_end = length_end + length
                if length < _MIN_EXTENDED_LENGTH or frame_end - pos > self.max_frame_length:
                    # Too short to hold a frame, or longer than any frame expected.
                    pos += 1
                    continue
                if frame_end > len(buf) and self._complete:
                    # A whole stream cannot complete it, so it is a false start. A
                    # regular frame past the end is a truncated tail, and still ends
                    # the parse below as it always has.
                    pos += 1
                    continue
            else:
                if pos + 1 >= len(buf):
                    break  # need more data

                length = buf[pos + 1]
                if length == 0:
                    # Length=0 is invalid (no remaining bytes, not even checksum).
                    pos += 1
                    continue
                frame_end = pos + 2 + length

            # --- State 3: ReadBody -----------------------------------------
            if frame_end > len(buf):
                break  # frame not yet complete

            frame = bytes(buf[pos:frame_end])
            try:
                msg = HarpMessage.parse(frame)
                pos = frame_end
                self._pos = pos
                return msg
            except HarpParseError:
                # Recovery: skip the bad MessageType byte, retry from pos+1.
                pos = msg_type_pos + 1
                continue

        self._pos = pos
        return None

    def frames(self) -> Iterator[HarpMessage]:
        """Yield all complete frames currently available in the buffer."""
        while (msg := self.next_frame()) is not None:
            yield msg

    def __iter__(self) -> Iterator[HarpMessage]:
        return self.frames()

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def parse_bytes(cls, data: bytes | bytearray) -> list[HarpMessage]:
        """Parse all Harp messages from a byte buffer.

        The buffer is taken to be the whole stream, so frames of any length are
        accepted, and an extended-length frame running past its end is a false start
        to skip rather than one to wait for.
        """
        framer = cls(max(len(data), _MAX_FRAME_LEN))
        framer._complete = True
        framer.feed(data)
        return list(framer.frames())

    @classmethod
    def parse_file(cls, path: str | Path) -> list[HarpMessage]:
        """Parse all Harp messages from a binary file."""
        with open(path, "rb") as f:
            data = f.read()
        return cls.parse_bytes(data)
