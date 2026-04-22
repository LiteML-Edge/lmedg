# Firmware Build Context

This guide explains what firmware-side documentation is included in the package and what host-side prerequisites are still needed for direct rebuilding on another machine.

## 1. Firmware source references included here

The package includes preserved firmware source references for the three model families under:

- `core_source_reference/firmwares/mlp/`
- `core_source_reference/firmwares/lstm/`
- `core_source_reference/firmwares/Conv1D_Tiny/`

These folders expose the main embedded logic used by the reported system and document the firmware-side implementation context discussed in the manuscript, including the embedded use of TensorFlow Lite Micro.

## 2. Supporting firmware-side materials

The firmware-side documentation in this package is complemented by:

- replay headers under `replay_evidence/`
- representative device logs under `firmware_logs/`
- workflow descriptors under `pipelines/`
- hardware-context notes under `support_guides/HARDWARE_SETUP.md`
- the packaged schematic PDF under `LiteML_Edge ESP32 Electrical Schematic/`

Together, these materials help a reviewer understand the firmware-side contract without requiring the full public code-release stage to be bundled here.

In particular, the current package documents TensorFlow Lite Micro as part of the firmware-side runtime context through preserved source references, logs, and surrounding guides. It does not claim that a ready-to-use bundled TensorFlow Lite Micro code release is already part of this initial-submission package.

## 3. What is still host-dependent for direct rebuilding

A reader who wants to rebuild firmware on another machine will still need host-side prerequisites such as:

- PlatformIO installation and the relevant ESP32 toolchain
- local serial-port selection
- compatible ESP32 hardware
- the sensor and measurement setup reflected in the packaged documentation

These requirements are normal host-side conditions for direct rebuilding and do not change the role of the current package as initial-submission technical supporting documentation. They are also consistent with separating the current documentation package from the later Link-to-Code stage, where the operational firmware-side code layout can be provided more completely.

## 4. Interpretation boundary

The current package already contains the firmware-side technical documentation and preserved source references needed for technical review and implementation understanding. What remains host-dependent is the machine and hardware setup required for direct rebuilding and on-device retracing on a new system, together with the later Link-to-Code packaging of the operational TensorFlow Lite Micro integration.

## 5. Recommended cross-check files

When reviewing the firmware side, inspect these files together:

- `core_source_reference/firmwares/*/`
- `replay_evidence/*/*.h`
- `firmware_logs/*/*.log`
- `support_guides/HARDWARE_SETUP.md`
- `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`
