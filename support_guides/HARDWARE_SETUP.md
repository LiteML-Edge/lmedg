# Hardware Setup

This file records the hardware assumptions documented by the current LiteML-Edge technical support package. The package includes a dedicated hardware-schematic folder: `LiteML_Edge ESP32 Electrical Schematic/`.

## Primary embedded target

The preserved firmware source references target:

- ESP32 family
- Wemos / Lolin32-class boards
- Arduino-based PlatformIO projects in the original repository

## Serial assumptions recovered from the original PlatformIO configuration

The preserved project evidence indicates the following local-machine settings during the recorded runs:

- upload port: `COM6`
- monitor port: `COM6`
- upload speed: `115200`
- monitor speed: `115200`

These values are host-specific and may differ on another machine.

## Packaged hardware schematic

The package includes `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`, which documents the reported ESP32 sensing-node wiring. The schematic shows the ESP32 board, the indoor and outdoor DHT22 sensors, the INA219 power monitor, shared SDA/SCL lines for OLED and INA219, GND, the USB/+5V rail, and the GPIO assignment table used in the packaged hardware context.

## Measurement instrumentation reflected in the logs

Representative log files contain lines such as:

- `[BENCH][INA219] calib=16V/400mA | assumed_shunt=0.100 ohm`

This indicates that the measured deployment workflow includes INA219-based current and power instrumentation for the benchmark and power-reporting path used in the paper.

## Operational modes

The package contains replay-specific exported sample headers such as:

- `environment_quantized_samples_replay_raw_2plus47_mlp.h`
- `environment_quantized_samples_replay_raw_2plus47_lstm.h`
- `environment_quantized_samples_replay_raw_2plus47_Conv1D_Tiny.h`

This supports inspection of the firmware-side replay execution contract without requiring live sensing for the core validation path.

## Practical documentation boundary

A reader can inspect the Python-side artifacts, preserved firmware source references, and workbook outputs without hardware. Hardware is required to independently inspect the on-device execution path and to retrace the memory, timing, and electrical measurements reported in the firmware logs.
