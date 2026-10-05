import zlib
from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, cast, final, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Sentinel

from ._builder import build_message_frame
from ._constants import (
    _CHECKSUM_LEN,
    _CRC_LEN,
    _DEFAULT_PORT,
    _EXTENDED_HEADER_LEN,
    _EXTENDED_LENGTH_FLAG,
    _HEADER_LEN,
    _MAX_REGULAR_PAYLOAD_LEN,
    _MIN_FRAME_LEN,
    _TICK_PERIOD_S,
    _TIMESTAMP_FLAG,
    _TIMESTAMP_LEN,
    _TIMESTAMPED_PAYLOAD_OFFSET,
    _TS_MICROS_OFFSET,
)
from ._message import HarpMessage, HarpParseError
from ._message_type import MessageType, message_type_to_byte
from ._payload import (
    AnonymousPayload,
    Batch,
    PayloadBase,
    PayloadFloat,
    PayloadFloatArray,
    PayloadS8,
    PayloadS8Array,
    PayloadS16,
    PayloadS16Array,
    PayloadS32,
    PayloadS32Array,
    PayloadS64,
    PayloadS64Array,
    PayloadU8,
    PayloadU8Array,
    PayloadU16,
    PayloadU16Array,
    PayloadU32,
    PayloadU32Array,
    PayloadU64,
    PayloadU64Array,
)
from ._payload_type import PayloadType, encode_payload_type

_MISSING = Sentinel("_MISSING")


def _encode_message_types(message_type: MessageType | ArrayLike, nrows: int) -> NDArray[np.uint8]:
    """Resolve a scalar/array ``message_type`` argument to N message-type bytes.

    A single :class:`MessageType` (error-bit aware) fills all frames; a scalar int
    is used verbatim; an array (e.g. the msgtype view from ``parse_bulk``, or a
    list of ``MessageType``/ints) becomes the per-frame bytes.
    """
    if isinstance(message_type, MessageType):
        return np.full(nrows, message_type_to_byte(message_type), dtype=np.uint8)
    values = np.asarray(message_type)
    if values.ndim == 0:
        return np.full(nrows, int(values.item()), dtype=np.uint8)
    return values.astype(np.uint8)


U = TypeVar("U")
_R = TypeVar("_R")
_AR = TypeVar("_AR", bound="RegisterBase[Any]")


class _LazyTimestamps:
    """Seconds + microseconds timestamp views, combined into float64 on first use.

    Combining the raw views costs an O(n) pass over every frame, two ``astype``
    casts plus a multiply-add, independent of the register payload, so eagerly
    computing it in ``parse_bulk`` taxes every call even when the caller never
    reads the timestamps. Deferring the combine until the array is actually
    accessed, and caching the result, avoids that cost in the common case where
    only the payload is needed.
    """

    __slots__ = ("_ts_s", "_ts_us", "_values")

    def __init__(self, ts_s: np.ndarray, ts_us: np.ndarray) -> None:
        self._ts_s = ts_s
        self._ts_us = ts_us
        self._values: np.ndarray | None = None

    def _resolve(self) -> np.ndarray:
        if self._values is None:
            out = np.multiply(self._ts_us, _TICK_PERIOD_S, dtype=np.float64)
            np.add(self._ts_s, out, out=out)
            self._values = out
        return self._values

    def __array__(self, dtype: "np.dtype | None" = None) -> np.ndarray:
        arr = self._resolve()
        return arr if dtype is None else arr.astype(dtype)

    def __len__(self) -> int:
        return len(self._ts_s)

    def __iter__(self):
        return iter(self._resolve())

    def __getitem__(self, item: Any) -> Any:
        return self._resolve()[item]

    def __repr__(self) -> str:
        return repr(self._resolve())


class _RegisterBaseMeta(ABCMeta):
    """The metaclass of every register, rendering its name and address."""

    def __repr__(cls) -> str:
        address = getattr(cls, "address", None)
        return super().__repr__() if address is None else f"<{cls.__name__} @{address}>"

    @property
    def is_extended_length(cls) -> bool:
        """Whether every message of this register uses extended-length framing.

        Derived from the payload size: a register whose payload does not fit in a
        regular frame together with a timestamp uses extended-length framing, a U32
        length and a CRC-32, for every message to and from it. That holds for a message
        without a timestamp too, and for a ``Read`` request with no payload at all, so
        the framing is a property of the register rather than of any one message.
        """
        # Cached on the class itself, not inherited, since a subclass may resize the
        # payload; formatting reads it for every frame.
        cached = cls.__dict__.get("_extended_length")
        if cached is None:
            payload_class = getattr(cls, "payload_class", None)
            cached = (
                payload_class is not None
                and payload_class._max_payload_size() > _MAX_REGULAR_PAYLOAD_LEN
            )
            setattr(cls, "_extended_length", cached)
        return cached

    @property
    def is_variable_length(cls) -> bool:
        """Whether a message of this register may carry any number of elements.

        True for an array register declared with ``max_length``, whose messages carry
        from none up to that many elements, and False for every register of fixed size.
        """
        payload_class = getattr(cls, "payload_class", None)
        return payload_class is not None and payload_class._max_length is not None


def _require_no_address(cls: Any) -> None:
    address = getattr(cls, "address", None)
    if address is not None:
        raise TypeError(
            f"{cls.__name__} already declares address {address} and cannot be reassigned."
        )


class _ScalarRegisterMeta(_RegisterBaseMeta):
    """Calling a register base with an address creates a one-off subclass: ``RegisterU32(0x08)``."""

    def __call__(cls: "type[_R]", address: int) -> "type[_R]":
        _require_no_address(cls)
        return cast(
            "type[_R]",
            type(f"{cls.__name__}_{address:#04x}", (cls,), {"address": address}),
        )


class RegisterBase(ABC, Generic[U], metaclass=_RegisterBaseMeta):
    """Abstract base for all typed Harp registers.

    The generic parameter ``U`` is the static return type of :meth:`parse`, the
    user-facing value, *not* necessarily ``payload_class``, which is the wire
    encoding. The two coincide only for multi-member struct payloads:

    * scalar registers -> a numpy scalar, for example ``np.uint16``;
    * array registers  -> ``NDArray[...]`` of fixed length;
    * multi-member struct registers -> the payload class itself;
    * single-member registers that unwrap on parse -> the inner value type, for
      example ``RegisterBase[str]`` for DeviceName, ``RegisterBase[HarpVersion]``,
      or ``RegisterBase[ClockConfigurationFlags]`` for a whole-register
      ``BitMask`` or ``GroupMask``, even though each still has a ``payload_class``.

    Subclasses must define ``address``, ``payload_type``, and ``payload_class`` as
    ``ClassVar``s. The extent of a payload is always read from ``payload_class``.
    """

    address: ClassVar[int]
    payload_type: ClassVar[PayloadType]
    payload_class: ClassVar[type[PayloadBase[Any]]]

    @classmethod
    def parse(cls, value: HarpMessage | bytes | bytearray | memoryview) -> U:
        """Parse a single message into the user-facing payload value.

        Struct payloads return a typed wrapper (descriptor access like
        ``payload.Channel0`` works). Anonymous payloads (scalar / array
        registers) return the raw numpy scalar or ndarray directly.
        """
        buf = value.payload_bytes if isinstance(value, HarpMessage) else value
        payload_class = cls.payload_class
        if payload_class._max_length is not None:
            # A variable-length array: every whole element present, up to the maximum.
            if not payload_class._accepts_payload_size(len(buf)):
                raise HarpParseError(
                    f"{cls.__name__} reads up to {payload_class._max_length} elements of "
                    f"{cls.payload_type!r} but {len(buf)} payload bytes are not that."
                )
            elements = np.frombuffer(buf, dtype=payload_class.payload_dtype)
            return cast(U, payload_class._unwrap(elements))
        expected = payload_class.payload_dtype.itemsize
        if len(buf) < expected:
            raise HarpParseError(
                f"{cls.__name__} reads {expected} payload bytes as {cls.payload_type!r} "
                f"but only {len(buf)} are available."
            )
        record = np.frombuffer(buf, dtype=cls.payload_class.payload_dtype, count=1)[0]
        return cast(U, cls.payload_class._unwrap(record))

    @classmethod
    def parse_bulk(
        cls,
        source: bytes | bytearray | memoryview,
        *,
        parse_timestamp: bool = True,
    ) -> "tuple[np.ndarray, _LazyTimestamps | None, np.ndarray | None, Batch[Any]]":
        """Parse a bulk buffer containing one or more frames of this register type. Returns (data, timestamps, msgtype_view, payload)."""
        # Returns (data, timestamps, msgtype_view, payload). ``data`` is
        payload_cls = cls.payload_class
        data = np.frombuffer(source, dtype=np.uint8)

        if payload_cls._max_length is not None:
            raise NotImplementedError(
                f"{cls.__name__}: bulk parsing of variable-length frames is not supported yet."
            )
        if cls.is_extended_length or (len(data) > 0 and int(data[0]) & _EXTENDED_LENGTH_FLAG):
            # A recording of such a register mixes frames of different sizes, since a
            # write is answered with a regular frame carrying only a CRC, so no single
            # stride describes it.
            raise NotImplementedError(
                f"{cls.__name__}: bulk parsing of extended-length frames is not supported yet."
            )

        if len(data) == 0:
            # No frames, but still need to return a Batch with the right dtype.
            payload = payload_cls._from_array(np.empty(0, dtype=payload_cls.payload_dtype))
            return data, None, None, cast("Batch[Any]", payload)

        if len(data) < _MIN_FRAME_LEN:
            raise HarpParseError(
                f"{cls.__name__} reads frames of at least {_MIN_FRAME_LEN} bytes "
                f"but only {len(data)} are available."
            )

        stride = (
            int(data[1]) + 2
        )  # TODO this assumes all frames have the same length but we may want to revisit in the future.
        if len(data) < stride:
            raise HarpParseError(
                f"{cls.__name__} reads frames of {stride} bytes as declared by the first "
                f"length byte, but only {len(data)} are available."
            )
        nrows = len(data) // stride
        is_timestamped = bool(int(data[4]) & _TIMESTAMP_FLAG)
        payload_offset = _TIMESTAMPED_PAYLOAD_OFFSET if is_timestamped else _HEADER_LEN

        if is_timestamped and parse_timestamp:
            ts_s = np.ndarray(nrows, dtype="<u4", buffer=data, offset=_HEADER_LEN, strides=stride)
            ts_us = np.ndarray(
                nrows, dtype="<u2", buffer=data, offset=_TS_MICROS_OFFSET, strides=stride
            )
            timestamps = _LazyTimestamps(ts_s, ts_us)
        # TODO we may want to check if the timestamp is not present and users ask to be parsed. In that case we can either raise an error or return a nan-filled array
        else:
            timestamps = None

        msgtype_view = np.ndarray(nrows, dtype=np.uint8, buffer=data, offset=0, strides=stride)

        payload_arr = np.ndarray(
            nrows,
            dtype=payload_cls.payload_dtype,
            buffer=data,
            offset=payload_offset,
            strides=stride,
        )

        payload = payload_cls._from_array(payload_arr)
        return data, timestamps, msgtype_view, cast("Batch[Any]", payload)

    @classmethod
    def format_bulk(
        cls,
        values: PayloadBase | ArrayLike,
        *,
        timestamps: ArrayLike | None = None,
        message_type: MessageType | ArrayLike = MessageType.Event,
        port: int = _DEFAULT_PORT,
    ) -> NDArray[np.uint8]:
        """Build a flat buffer of N frames of this register type, the inverse of
        :meth:`parse_bulk`.

        ``values`` is a payload, either scalar or :class:`Batch`, or an ndarray of
        the ``payload_class.payload_dtype`` of the register. ``timestamps``, a
        length-N array of seconds, makes every frame timestamped. ``message_type``
        is one :class:`MessageType` for all frames, or a length-N array of
        message-type bytes or values, for example the ``msgtype`` view returned by
        ``parse_bulk``.
        """
        payload_cls = cls.payload_class
        if payload_cls._max_length is not None:
            raise NotImplementedError(
                f"{cls.__name__}: bulk formatting of variable-length frames is not supported yet."
            )
        record_dtype = payload_cls.payload_dtype
        itemsize = record_dtype.itemsize
        subdtype = record_dtype.subdtype
        if isinstance(values, PayloadBase):
            records = np.atleast_1d(np.asarray(values.payload_array))
        else:
            records = np.atleast_1d(np.asarray(values))
        if subdtype is not None and records.dtype != record_dtype:
            # An array register: convert against the element type and let the declared
            # shape decide the frame count, as `format` does for a single frame.
            element_dtype, shape = subdtype
            records = np.asarray(records, dtype=element_dtype)
            if records.shape[-len(shape) :] != shape:
                raise ValueError(
                    f"{cls.__name__}.format_bulk: values of shape {records.shape} do not end "
                    f"in the declared payload shape {shape}"
                )
            records = records.reshape(-1, *shape)
        elif (
            # Coerce the element type only for plain scalar payloads (e.g. an int
            # list for a scalar register). Struct records already carry the right
            # byte layout and must not be re-cast.
            records.dtype.names is None
            and records.dtype.subdtype is None
            and record_dtype.names is None
            and subdtype is None
            and records.dtype != record_dtype
        ):
            records = records.astype(record_dtype)
        nrows = len(records)
        flat = np.ascontiguousarray(records).tobytes()
        if len(flat) != nrows * itemsize:
            raise ValueError(
                f"{cls.__name__}.format_bulk: {len(flat)} payload bytes for {nrows} frames "
                f"is not a multiple of itemsize {itemsize}; check the values shape/dtype"
            )

        is_timestamped = timestamps is not None
        extended = cls.is_extended_length
        header_len = _EXTENDED_HEADER_LEN if extended else _HEADER_LEN
        checksum_len = _CRC_LEN if extended else _CHECKSUM_LEN
        payload_offset = header_len + _TIMESTAMP_LEN if is_timestamped else header_len
        ts_micros_offset = header_len + 4
        stride = payload_offset + itemsize + checksum_len

        buf = np.zeros((nrows, stride), dtype=np.uint8)
        # The framing is the register's, whatever flags the given message types carry.
        msg_types = _encode_message_types(message_type, nrows)
        if extended:
            buf[:, 0] = msg_types | _EXTENDED_LENGTH_FLAG
            length = np.array([stride - (header_len - 3)], dtype="<u4")
            buf[:, 1 : header_len - 3] = length.view(np.uint8)
        else:
            buf[:, 0] = msg_types & np.uint8(~_EXTENDED_LENGTH_FLAG & 0xFF)
            buf[:, 1] = stride - 2
        buf[:, header_len - 3] = cls.address
        buf[:, header_len - 2] = port
        buf[:, header_len - 1] = encode_payload_type(cls.payload_type, has_timestamp=is_timestamped)

        if is_timestamped:
            ts = np.atleast_1d(np.asarray(timestamps, dtype=np.float64))
            seconds = ts.astype(np.uint32)
            micros = np.round((ts - seconds.astype(np.float64)) / _TICK_PERIOD_S).astype(np.uint16)
            buf[:, header_len:ts_micros_offset] = np.frombuffer(
                seconds.astype("<u4").tobytes(), dtype=np.uint8
            ).reshape(nrows, 4)
            buf[:, ts_micros_offset:payload_offset] = np.frombuffer(
                micros.astype("<u2").tobytes(), dtype=np.uint8
            ).reshape(nrows, 2)

        payload_bytes = np.frombuffer(flat, dtype=np.uint8).reshape(nrows, itemsize)
        buf[:, payload_offset : payload_offset + itemsize] = payload_bytes
        if extended:
            crcs = np.fromiter(
                (zlib.crc32(row) for row in buf[:, :-_CRC_LEN]), dtype="<u4", count=nrows
            )
            buf[:, -_CRC_LEN:] = crcs.view(np.uint8).reshape(nrows, _CRC_LEN)
        else:
            buf[:, -1] = buf[:, :-1].sum(axis=1, dtype=np.uint64).astype(np.uint8)
        return buf.reshape(-1)

    @overload
    @classmethod
    def format(
        cls,
        *,
        message_type: MessageType = MessageType.Read,
        timestamp: float | None = None,
        port: int = _DEFAULT_PORT,
    ) -> bytes: ...

    @overload
    @classmethod
    def format(
        cls,
        value: U,
        *,
        message_type: MessageType = MessageType.Write,
        timestamp: float | None = None,
        port: int = _DEFAULT_PORT,
    ) -> bytes: ...
    # We go with "U" for typing but it is worth noting that we accept "Any" below.
    # However for API ergonomics we want to keep the type hint for symmetry
    @final
    @classmethod
    def format(
        cls,
        value: Any = _MISSING,
        *,
        message_type: MessageType | None = None,
        timestamp: float | None = None,
        port: int = _DEFAULT_PORT,
    ) -> bytes:
        """Build a Harp frame for this register. No value gives a Read, a value gives a Write."""
        if value is _MISSING:
            mt = MessageType.Read if message_type is None else message_type
            return build_message_frame(
                mt,
                cls.address,
                cls.payload_type,
                port=port,
                timestamp=timestamp,
                extended_length=cls.is_extended_length,
            )
        else:
            mt = MessageType.Write if message_type is None else message_type
            if isinstance(value, PayloadBase):
                raw = value.payload_array.tobytes()
            elif cls.payload_class._max_length is not None:
                # The payload class checks the element count and converts the elements.
                raw = cls.payload_class(value).payload_array.tobytes()
            elif isinstance(value, np.ndarray):
                raw = value.tobytes()
            else:
                # A bare high-level value (the symmetric counterpart of what
                # parse() returns): let the payload class encode it, so any
                # converter (e.g. a str via StringConverter) is applied.
                raw = cls.payload_class(value).payload_array.tobytes()
            return build_message_frame(
                mt,
                cls.address,
                cls.payload_type,
                raw,
                port=port,
                timestamp=timestamp,
                extended_length=cls.is_extended_length,
            )


class ExtendedLengthRegister:
    """Marks a register as extended-length for type checkers.

    A device answers a write to an extended-length register with an
    :class:`ExtendedMessageReceipt` rather than the written value, which
    :meth:`~harp.device.client.Device.write` can only express in its return type when
    the register is marked::

        class Waveform(RegisterU16Array, ExtendedLengthRegister):
            address = 0x64
            length = 4096

    The marker has no effect at runtime. Framing is always derived from the payload
    size (see ``is_extended_length``), so an unmarked large register is still framed and
    answered correctly, only typed as returning its own payload.

    Extended-length framing is proposed in harp-tech/protocol#218 and is not yet part
    of the protocol specification, so this API may change.
    """


@dataclass(frozen=True)
class ExtendedMessageReceipt:
    """The payload of the device's answer to a write of an extended-length register.

    A device does not echo the written value back, which may be megabytes long, but
    the CRC-32 it computed over the request it received, as proposed in
    harp-tech/protocol#221. A receipt matching the CRC of the request sent is
    evidence the device received the frame intact.

    ``ExtendedMessageReceipt`` also decodes such a message itself, as
    ``message.decode(ExtendedMessageReceipt)``.
    """

    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class: ClassVar[type[PayloadBase[Any]]] = PayloadU32

    crc: int
    """The CRC-32 the device computed over the request it received."""

    @classmethod
    def parse(cls, value: HarpMessage | bytes | bytearray | memoryview) -> "ExtendedMessageReceipt":
        """Read a receipt out of a message or its payload bytes."""
        return cls(int(RegisterU32.parse(value)))


class RegisterU8(RegisterBase[np.uint8], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint8 payload. ``parse()`` returns ``np.uint8``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = PayloadU8


class RegisterU16(RegisterBase[np.uint16], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint16 payload. ``parse()`` returns ``np.uint16``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = PayloadU16


class RegisterU32(RegisterBase[np.uint32], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint32 payload. ``parse()`` returns ``np.uint32``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class = PayloadU32


class RegisterU64(RegisterBase[np.uint64], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a uint64 payload. ``parse()`` returns ``np.uint64``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U64
    payload_class = PayloadU64


class RegisterS8(RegisterBase[np.int8], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with an int8 payload. ``parse()`` returns ``np.int8``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S8
    payload_class = PayloadS8


class RegisterS16(RegisterBase[np.int16], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with an int16 payload. ``parse()`` returns ``np.int16``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S16
    payload_class = PayloadS16


class RegisterS32(RegisterBase[np.int32], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with an int32 payload. ``parse()`` returns ``np.int32``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S32
    payload_class = PayloadS32


class RegisterS64(RegisterBase[np.int64], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with an int64 payload. ``parse()`` returns ``np.int64``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S64
    payload_class = PayloadS64


class RegisterFloat(RegisterBase[np.float32], metaclass=_ScalarRegisterMeta):
    """A simple scalar register with a float32 payload. ``parse()`` returns ``np.float32``."""

    payload_type: ClassVar[PayloadType] = PayloadType.Float
    payload_class = PayloadFloat


class _ArrayRegisterMeta(_RegisterBaseMeta):
    """A declared ``length`` or ``max_length`` sizes the payload, and calling a register
    base with an address and either creates a one-off subclass:
    ``RegisterU16Array(0x28, length=3)`` or ``RegisterU8Array(0x11, max_length=64)``.

    ``length`` fixes the element count of every message. ``max_length`` makes the
    register variable-length instead: a message carries any number of elements from
    none up to ``max_length``, and ``parse`` returns an array of as many as arrived.
    Framing is derived from the maximum, so a variable-length register whose maximum
    does not fit in a regular frame uses extended-length framing for every message.

    Both are declared here rather than on ``RegisterBase``, so only an array register
    carries them, and a register declares exactly one. They are element counts, and
    nothing reads them to size a payload.
    """

    length: int
    max_length: int
    payload_class: type[AnonymousPayload[Any]]

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        # The namespace holds this class body only, not inherited values, so a plain
        # subclass reads None and keeps the payload already sized by its base.
        length = namespace.get("length")
        max_length = namespace.get("max_length")
        if length is None and max_length is None:
            return
        if length is not None and max_length is not None:
            raise TypeError(f"{name} declares both length and max_length; declare one.")
        base_payload = cls.payload_class
        if base_payload.payload_dtype.subdtype is not None or base_payload._max_length is not None:
            raise TypeError(f"{name} redeclares a length already applied by its base class.")
        if length is not None:
            # A sub-array dtype, so reading one buffer element gives an ndarray of that shape.
            cls.payload_class = type(
                f"{base_payload.__name__}_{length}",
                (base_payload,),
                {"payload_dtype": np.dtype((base_payload.payload_dtype, (length,)))},
            )
            return
        assert max_length is not None  # declaring neither returned above
        if max_length < 1:
            raise ValueError(f"{name} declares max_length={max_length}; it must be at least 1.")
        # The dtype stays a single element, read as many times as the payload holds.
        cls.payload_class = type(
            f"{base_payload.__name__}_max{max_length}",
            (base_payload,),
            {"payload_dtype": base_payload.payload_dtype, "_max_length": max_length},
        )

    # The overloads let a type checker require exactly one of the two sizes, which the
    # implementation checks again at runtime for untyped callers.
    @overload
    def __call__(  # type: ignore[override, misc]
        cls: "type[_AR]",  # type: ignore[misc]
        address: int,
        *,
        length: int,
    ) -> "type[_AR]": ...

    @overload
    def __call__(  # type: ignore[override, misc]
        cls: "type[_AR]",  # type: ignore[misc]
        address: int,
        *,
        max_length: int,
    ) -> "type[_AR]": ...

    def __call__(  # type: ignore[override, misc]
        cls: "type[_AR]",  # type: ignore[misc]
        address: int,
        *,
        length: int | None = None,
        max_length: int | None = None,
    ) -> "type[_AR]":
        _require_no_address(cls)
        if (length is None) == (max_length is None):
            raise TypeError(f"{cls.__name__}() takes exactly one of length= or max_length=.")
        sizing = {"length": length} if length is not None else {"max_length": max_length}
        return cast(
            "type[_AR]",
            type(f"{cls.__name__}_{address:#04x}", (cls,), {"address": address, **sizing}),
        )


class RegisterU8Array(RegisterBase[NDArray[np.uint8]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint8 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterU8Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint8]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U8
    payload_class = PayloadU8Array


class RegisterU16Array(RegisterBase[NDArray[np.uint16]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint16 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterU16Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint16]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = PayloadU16Array


class RegisterU32Array(RegisterBase[NDArray[np.uint32]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint32 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterU32Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint32]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U32
    payload_class = PayloadU32Array


class RegisterU64Array(RegisterBase[NDArray[np.uint64]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a uint64 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterU64Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.uint64]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.U64
    payload_class = PayloadU64Array


class RegisterS8Array(RegisterBase[NDArray[np.int8]], metaclass=_ArrayRegisterMeta):
    """A simple array register with an int8 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterS8Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int8]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S8
    payload_class = PayloadS8Array


class RegisterS16Array(RegisterBase[NDArray[np.int16]], metaclass=_ArrayRegisterMeta):
    """A simple array register with an int16 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterS16Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int16]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S16
    payload_class = PayloadS16Array


class RegisterS32Array(RegisterBase[NDArray[np.int32]], metaclass=_ArrayRegisterMeta):
    """A simple array register with an int32 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterS32Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int32]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S32
    payload_class = PayloadS32Array


class RegisterS64Array(RegisterBase[NDArray[np.int64]], metaclass=_ArrayRegisterMeta):
    """A simple array register with an int64 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterS64Array(0x28, length=3)``. ``parse()`` returns ``NDArray[np.int64]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.S64
    payload_class = PayloadS64Array


class RegisterFloatArray(RegisterBase[NDArray[np.float32]], metaclass=_ArrayRegisterMeta):
    """A simple array register with a float32 array payload. It must be sized with a fixed ``length`` or a variable ``max_length``: ``RegisterFloatArray(0x28, length=3)``. ``parse()`` returns ``NDArray[np.float32]``."""

    payload_type: ClassVar[PayloadType] = PayloadType.Float
    payload_class = PayloadFloatArray
