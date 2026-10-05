import queue
import struct
import threading
import time
import types
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from harp.device import core
from harp.device.client import Device, DeviceError, TransportError, WriteVerificationError
from harp.protocol import (
    ExtendedLengthRegister,
    ExtendedMessageReceipt,
    HarpMessage,
    MessageType,
    PayloadType,
    RegisterU8Array,
    RegisterU16Array,
)
from tests.fixtures import make_frame_from_raw

_U16 = 0x02
"""Payload-type byte of a U16 payload, as the size nibble alone."""


class _NullTransport:
    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> None: ...

    def read(self) -> bytes:
        return b""


class _ScriptedTransport:
    """A transport replying with whatever ``on_write`` returns for each request.

    Frames are queued from inside ``write``, which runs only once the request is
    registered, so a reply cannot be dispatched before there is a waiter to receive it.
    Setting ``failing`` makes the next read fail, as a removed port would.
    """

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.failing = False
        self.on_write: Callable[[bytes], Iterable[bytes]] | None = None
        self._inbox: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        if self.on_write is not None:
            for frame in self.on_write(data):
                self._inbox.put(frame)

    def read(self) -> bytes:
        if self.failing:
            raise TransportError("simulated transport failure")
        try:
            return self._inbox.get(timeout=0.01)
        except queue.Empty:
            return b""


class _ShortTimeoutDevice(Device[None]):
    """A device that gives up on a reply quickly, so a test never waits five seconds."""

    REPLY_TIMEOUT = 0.5


def _module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.DEVICE_NAME = name
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_zero_whoami_skips_validation():
    # 0 marks an unregistered device, so opening must not read WhoAmI at all.
    device = Device(_NullTransport(), _module("Unregistered", WHO_AM_I=0, REGISTER_MAP={}))
    with device:
        assert device.module.WHO_AM_I == 0


def test_module_property_returns_given_module():
    module = _module("Behavior", WHO_AM_I=0, REGISTER_MAP={})
    assert Device(_NullTransport(), module).module is module
    assert Device(_NullTransport()).module is None


def test_write_reply_does_not_satisfy_read():
    # A reply carries the message type of its request, so a write reply on the same
    # register must not answer a pending read.
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (
        core.WhoAmI.format(np.uint16(7), message_type=MessageType.Write),
        core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),
    )
    with _ShortTimeoutDevice(transport) as device:
        assert int(device.read(core.WhoAmI).payload) == 9


def test_event_does_not_satisfy_read():
    # Events are unsolicited, so one for the same register must not answer a read.
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (
        core.WhoAmI.format(np.uint16(7), message_type=MessageType.Event),
        core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),
    )
    with _ShortTimeoutDevice(transport) as device:
        assert int(device.read(core.WhoAmI).payload) == 9


def test_concurrent_reads_share_one_reply():
    # The wire carries no request identifier, so two reads in flight on one register
    # cannot be told apart in the reply. Both are answered by it rather than one
    # replacing the waiter of the other.
    # Writes are serialized, so replying only to the second one guarantees the first
    # read is still waiting when the reply arrives.
    transport = _ScriptedTransport()

    def reply_to_second(_: bytes) -> Iterable[bytes]:
        first = len(transport.writes) == 1
        return () if first else (core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),)

    transport.on_write = reply_to_second
    with _ShortTimeoutDevice(transport) as device, ThreadPoolExecutor(2) as pool:
        replies = [pool.submit(device.read, core.WhoAmI) for _ in range(2)]
        assert [int(reply.result(timeout=2).payload) for reply in replies] == [9, 9]


def test_concurrent_requests_do_not_interleave_writes():
    # A frame is one transport write, and a second request must not start writing
    # while the first is still on the wire.
    transport = _ScriptedTransport()
    writing = threading.Lock()
    overlapped = threading.Event()

    def slow_write(frame: bytes) -> Iterable[bytes]:
        if not writing.acquire(blocking=False):
            overlapped.set()
            return ()
        try:
            time.sleep(0.05)
        finally:
            writing.release()
        return (core.WhoAmI.format(np.uint16(9), message_type=MessageType.Read),)

    transport.on_write = slow_write
    with _ShortTimeoutDevice(transport) as device, ThreadPoolExecutor(4) as pool:
        replies = [pool.submit(device.read, core.WhoAmI) for _ in range(4)]
        for reply in replies:
            reply.result(timeout=2)
    assert not overlapped.is_set()


def test_transport_failure_faults_pending_read():
    transport = _ScriptedTransport()

    def fail(_: bytes) -> Iterable[bytes]:
        transport.failing = True
        return ()

    transport.on_write = fail
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(TransportError):
            device.read(core.WhoAmI)


def test_read_after_transport_failure_raises_transport_error():
    # A failure that has already stopped the reader is reported to the next request
    # rather than leaving it to time out on a reply that cannot arrive.
    transport = _ScriptedTransport()

    def fail(_: bytes) -> Iterable[bytes]:
        transport.failing = True
        return ()

    transport.on_write = fail
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(TransportError):
            device.read(core.WhoAmI)
        with pytest.raises(TransportError):
            device.read(core.WhoAmI)
    assert len(transport.writes) == 1  # the second request was never written to the transport


def test_error_reply_raises_device_error():
    frame = make_frame_from_raw(MessageType.Read | 0x08, 0, 255, _U16, b"\x07\x00")
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (frame,)
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(DeviceError) as error:
            device.read(core.WhoAmI)
    assert error.value.reply.bytes == frame  # the frame is kept, not formatted away


def test_error_reply_returned_when_not_raising():
    frame = make_frame_from_raw(MessageType.Read | 0x08, 0, 255, _U16, b"\x07\x00")
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (frame,)
    with _ShortTimeoutDevice(transport, raise_on_error=False) as device:
        reply = device.read(core.WhoAmI)
    assert reply.has_error
    assert int(reply.payload) == 7


def test_read_multi_element_register_returns_payload():
    # read decodes the reply through the register, so a payload of several elements has to
    # survive that step.
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (
        core.DeviceName.format("Behavior", message_type=MessageType.Read),
    )
    with _ShortTimeoutDevice(transport) as device:
        assert device.read(core.DeviceName).payload == "Behavior"


def test_write_multi_element_register_returns_payload():
    transport = _ScriptedTransport()
    transport.on_write = lambda data: (data,)  # a device echoing the write
    with _ShortTimeoutDevice(transport) as device:
        assert device.write(core.DeviceName, "Behavior").payload == "Behavior"


# --------------------------------------------------------------------------
# Extended-length registers
# --------------------------------------------------------------------------


class _Waveform(RegisterU16Array, ExtendedLengthRegister):
    address = 0x64
    length = 4096


_WAVEFORM = np.arange(4096, dtype=np.uint16)


def _receipt(crc: int, *, error: bool = False, address: int = _Waveform.address) -> bytes:
    frame = bytearray(
        HarpMessage(
            MessageType.Write,
            address,
            PayloadType.U32,
            struct.pack("<I", crc),
            timestamp=1.0,
        ).bytes
    )
    if error:
        frame[0] |= 0x08
        frame[-1] = sum(frame[:-1]) & 0xFF
    return bytes(frame)


def _echo_crc(frame: bytes) -> Iterable[bytes]:
    return (_receipt(int.from_bytes(frame[-4:], "little")),)


def test_extended_write_returns_verified_receipt():
    transport = _ScriptedTransport()
    transport.on_write = _echo_crc
    with _ShortTimeoutDevice(transport) as device:
        reply = device.write(_Waveform, _WAVEFORM)
    sent = transport.writes[-1]
    assert HarpMessage.parse(sent).is_extended_length
    assert reply.payload == ExtendedMessageReceipt(crc=int.from_bytes(sent[-4:], "little"))
    assert reply.timestamp == pytest.approx(1.0)


def test_extended_write_receipt_mismatch_raises():
    transport = _ScriptedTransport()
    transport.on_write = lambda frame: (_receipt(int.from_bytes(frame[-4:], "little") ^ 1),)
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(WriteVerificationError) as info:
            device.write(_Waveform, _WAVEFORM)
    assert info.value.actual == info.value.expected ^ 1
    assert info.value.reply.payload.crc == info.value.actual


def test_extended_write_error_reply_raises_device_error():
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (_receipt(0, error=True),)
    with _ShortTimeoutDevice(transport) as device:
        with pytest.raises(DeviceError):
            device.write(_Waveform, _WAVEFORM)


def test_extended_write_error_reply_is_returned_unchecked_without_raise_on_error():
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (_receipt(0, error=True),)
    with _ShortTimeoutDevice(transport, raise_on_error=False) as device:
        reply = device.write(_Waveform, _WAVEFORM)
    assert reply.has_error
    assert reply.payload == ExtendedMessageReceipt(crc=0)


def test_extended_read_round_trip_without_module():
    # The framer grows to fit a register passed to read, so no module is needed.
    transport = _ScriptedTransport()
    transport.on_write = lambda _: (
        _Waveform.format(_WAVEFORM, message_type=MessageType.Read, timestamp=2.0),
    )
    with _ShortTimeoutDevice(transport) as device:
        reply = device.read(_Waveform)
    assert HarpMessage.parse(transport.writes[-1]).is_extended_length
    np.testing.assert_array_equal(reply.payload, _WAVEFORM)


def test_unmarked_extended_register_is_still_framed_and_answered():
    # The marker only types the reply; the framing and the receipt follow the size.
    register = RegisterU16Array(0x64, length=4096)
    transport = _ScriptedTransport()
    transport.on_write = _echo_crc
    with _ShortTimeoutDevice(transport) as device:
        reply = device.write(register, _WAVEFORM)
    assert isinstance(reply.payload, ExtendedMessageReceipt)


def test_framer_bound_derived_from_module():
    module = _module("Quac", WHO_AM_I=0, REGISTER_MAP={_Waveform.address: _Waveform})
    device = Device(_NullTransport(), module)
    assert device._framer.max_frame_length == 8 + 6 + 8192 + 4


def test_framer_bound_override():
    device = Device(_NullTransport(), max_frame_length=1 << 20)
    assert device._framer.max_frame_length == 1 << 20


def test_per_call_timeout():
    transport = _ScriptedTransport()  # never replies
    with _ShortTimeoutDevice(transport) as device:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="0.05s"):
            device.read(core.WhoAmI, timeout=0.05)
        assert time.monotonic() - start < _ShortTimeoutDevice.REPLY_TIMEOUT


def test_subscription_receives_extended_write_reply_as_receipt():
    transport = _ScriptedTransport()
    transport.on_write = _echo_crc
    received: queue.SimpleQueue[HarpMessage] = queue.SimpleQueue()
    with _ShortTimeoutDevice(transport) as device:
        device.subscribe(_Waveform, received.put, message_types=MessageType.Write)
        device.write(_Waveform, _WAVEFORM)
        delivered = received.get(timeout=2)
    assert isinstance(delivered.payload, ExtendedMessageReceipt)


# --------------------------------------------------------------------------
# Variable-length registers
# --------------------------------------------------------------------------


class _Tag(RegisterU8Array):
    address = 0x11
    max_length = 64


class _FirmwareImage(RegisterU8Array, ExtendedLengthRegister):
    address = 0xC8
    max_length = 1_048_576


def _echo_as_write_reply(frame: bytes) -> Iterable[bytes]:
    # A regular write is answered with the value written, whatever its length.
    sent = HarpMessage.parse(frame)
    return (
        HarpMessage(
            MessageType.Write,
            sent.address,
            sent.payload_type,
            bytes(sent.payload_bytes),
            timestamp=1.0,
        ).bytes,
    )


def test_variable_write_reply_carries_the_written_elements():
    transport = _ScriptedTransport()
    transport.on_write = _echo_as_write_reply
    with _ShortTimeoutDevice(transport) as device:
        reply = device.write(_Tag, np.frombuffer(b"rig-7", dtype=np.uint8))
    assert bytes(reply.payload) == b"rig-7"


def test_extended_variable_write_returns_receipt_and_read_returns_elements():
    image = np.arange(10_000, dtype=np.uint8)
    transport = _ScriptedTransport()

    def answer(frame: bytes) -> Iterable[bytes]:
        if HarpMessage.parse(frame).message_type is MessageType.Write:
            crc = int.from_bytes(frame[-4:], "little")
            return (_receipt(crc, address=_FirmwareImage.address),)
        return (_FirmwareImage.format(image, message_type=MessageType.Read, timestamp=1.0),)

    transport.on_write = answer
    with _ShortTimeoutDevice(transport) as device:
        receipt = device.write(_FirmwareImage, image)
        assert isinstance(receipt.payload, ExtendedMessageReceipt)
        np.testing.assert_array_equal(device.read(_FirmwareImage).payload, image)


def test_framer_bound_uses_the_variable_maximum():
    module = _module("Quac", WHO_AM_I=0, REGISTER_MAP={_FirmwareImage.address: _FirmwareImage})
    device = Device(_NullTransport(), module)
    assert device._framer.max_frame_length == 8 + 6 + 1_048_576 + 4
