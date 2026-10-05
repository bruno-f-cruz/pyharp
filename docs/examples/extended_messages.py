from typing import ClassVar

import numpy as np
from harp import serial
from harp.device import client
from harp.protocol import (
    ArrayConverter,
    ExtendedLengthRegister,
    Field,
    IdentityConverter,
    PayloadType,
    RegisterBase,
    RegisterU8Array,
    RegisterU16Array,
    StructPayload,
)
from numpy.typing import NDArray

SERIAL_PORT = "/dev/ttyUSB0"  # or "COMx" in Windows, where "x" is the serial port number


# A fixed-length array register. 4096 U16 samples are 8192 bytes, far more than the
# 245 a regular frame holds, so every message to and from it is extended-length.
# Deriving from ExtendedLengthRegister tells a type checker that writing it returns
# a receipt; the framing itself follows from the size either way.
class Waveform(RegisterU16Array, ExtendedLengthRegister):
    address = 100
    length = 4096


# A struct payload is declared field by field, as for any register. Its size in base
# elements, two U16 settings plus 4096 U16 samples here, decides the framing.
class WaveformPresetPayload(StructPayload[np.uint16], length=2 + 4096):
    channel: np.uint16 = Field(IdentityConverter(np.uint16), offset=0)
    sample_rate: np.uint16 = Field(IdentityConverter(np.uint16), offset=1)
    samples: NDArray[np.uint16] = Field(ArrayConverter(np.uint16, 4096), offset=2)


class WaveformPreset(RegisterBase[WaveformPresetPayload], ExtendedLengthRegister):
    address: ClassVar[int] = 101
    payload_type: ClassVar[PayloadType] = PayloadType.U16
    payload_class = WaveformPresetPayload


# A variable-length register carries anywhere from none up to max_length elements.
# The maximum decides the framing, so this one is extended even for a short image.
class FirmwareImage(RegisterU8Array, ExtendedLengthRegister):
    address = 102
    max_length = 1_048_576


# Variable-length but small: its largest message fits in a regular frame, so it is
# not extended, and needs no marker.
class Tag(RegisterU8Array):
    address = 103
    max_length = 64


for register in (Waveform, WaveformPreset, FirmwareImage, Tag):
    print(
        f"{register.__name__}: extended={register.is_extended_length}, "
        f"variable={register.is_variable_length}"
    )

samples = (2048 + 2000 * np.sin(np.linspace(0, 2 * np.pi, 4096))).astype(np.uint16)

with serial.open_device(port=SERIAL_PORT) as device:
    # The device answers an extended write with the CRC-32 of the request it received,
    # not with the samples, and write checks it against the request it sent. A slow
    # device may need longer than the default five seconds to answer.
    try:
        receipt = device.write(Waveform, samples, timeout=120.0)
    except client.WriteVerificationError as error:
        print(f"Corrupted in transit: sent 0x{error.expected:08x}, got 0x{error.actual:08x}")
        raise
    print(f"Waveform received with CRC-32 0x{receipt.payload.crc:08x}")

    # A read of an extended register returns its whole payload.
    stored = device.read(Waveform, timeout=30.0).payload
    print("Waveform read back intact:", np.array_equal(stored, samples))

    # A struct payload is built and read like any other.
    preset = WaveformPresetPayload(
        channel=np.uint16(0), sample_rate=np.uint16(1000), samples=samples
    )
    device.write(WaveformPreset, preset, timeout=120.0)
    print("Preset sample rate:", device.read(WaveformPreset).payload.sample_rate)

    # A variable-length write may carry fewer elements than the maximum.
    image = np.fromfile("firmware.bin", dtype=np.uint8)
    device.write(FirmwareImage, image, timeout=300.0)

    # A small variable-length register is answered with the value written, at
    # whatever length it was written.
    tag = device.write(Tag, np.frombuffer(b"rig-7", dtype=np.uint8)).payload
    print("Tag:", bytes(tag))
