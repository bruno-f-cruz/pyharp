# Send and receive extended messages

!!! warning "Experimental"
    Extended-length messages are proposed in [harp-tech/protocol#218](https://github.com/harp-tech/protocol/issues/218), [#220](https://github.com/harp-tech/protocol/issues/220) and [#221](https://github.com/harp-tech/protocol/issues/221), and are not yet part of the Harp specification. This API may change, and a device must implement the proposals for it to work.

A regular Harp message holds at most 245 payload bytes with a timestamp, since its length field is a single byte. A register with a larger payload, such as a stimulus waveform or a firmware image, uses *extended-length* messages instead. These have a 4-byte length field, so they can carry up to 4 GB, and a CRC-32 checksum in place of the 1-byte sum.

This example declares four device-specific registers and reads and writes them:

- an array register too large for a regular frame;
- a struct payload too large for one;
- a variable-length register for a firmware image;
- a small variable-length register that stays regular.

## Declaring registers

You don't choose the framing of a register; it follows from the size of its payload. A register whose largest payload exceeds 245 bytes uses extended-length framing for *every* message to and from it, including a read request with no payload. Each register reports this at runtime as `is_extended_length`, and `HarpMessage.is_extended_length` tells the same for a single message.

The payload is declared the same way as for any register:

- **Fixed-length arrays** take a `length`, as `Waveform` does.
- **Struct payloads** declare their fields with `Field`, using `ArrayConverter` for a member that spans many elements, as `WaveformPresetPayload` does. The `length` of the struct, in base elements, decides the framing.
- **Variable-length arrays** take a `max_length` instead of a `length`. A message then carries anywhere from no elements up to that many, and reading one returns an array of as many as arrived. The maximum decides the framing, so `FirmwareImage` is extended even for a short image, while `Tag` stays regular. Such registers report `is_variable_length`.

Deriving a register from `ExtendedLengthRegister` changes nothing at runtime. It tells a type checker that writing the register returns an `ExtendedMessageReceipt` rather than the written value.

## Writing and reading

A device answers an extended-length write with the CRC-32 of the request it received, not with the written payload, which may be megabytes long. `write` returns that receipt after checking it against the CRC of the request it sent. A mismatch means the message was corrupted in transit, and raises `WriteVerificationError` with both values.

Sending megabytes over serial and storing them can take far longer than the default reply timeout of five seconds. Pass a `timeout`, in seconds, to `read` and `write` to wait longer. The timeout only starts once the whole message has been sent.

A read of an extended-length register returns its full payload, as does an event, which `subscribe` receives as usual.

## Receiving frames from unknown registers

The device only accepts extended-length frames up to the size it expects. Any frame claiming more is treated as noise, so a corrupted length can't stall the stream.

- **Expected size:** it is the largest register of the device module, if one is given. It also grows to fit any register you pass to `read`, `write` or `subscribe`.
- **Raising it yourself:** to record extended-length events from registers the device has never been given, for example with `subscribe_all`, pass `max_frame_length` to `serial.open_device`.

## Limitations

- Recordings that contain extended-length frames can't be read with `harp.data` yet.
- Bulk formatting and parsing of variable-length registers aren't supported yet.

{% include-markdown "includes/serial-port.md" %}

<!--codeinclude-->
```python
[](./extended_messages.py)
```
<!--/codeinclude-->
