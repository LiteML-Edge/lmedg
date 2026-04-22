# TECHNICAL_SUPPORT_GUIDE

This guide presents the current package as a technical support package for LiteML-Edge.

Its role is to organize the packaged materials that help a reviewer understand the method described in the manuscript, inspect the implementation context, and follow the evidence paths that support the manuscript during the initial submission stage.

The package does not claim to be the final post-acceptance Link-to-Code release. Instead, it documents the implementation context through workflow descriptors, preserved source references, datasets, exported artifacts, workbooks, logs, and hardware-context materials.

## 1. What this package contains

The package combines the main documentation-facing layers needed to understand and technically inspect the LiteML-Edge workflow:

1. `datasets/` for source climate files, prepared datasets, and dataset preparation scripts;
2. `core_source_reference/trainings/` for training, pruning, quantization, header export, and scaler export source references;
3. `core_source_reference/firmwares/` for preserved firmware source references for MLP, LSTM, and Conv1D Tiny;
4. `pipelines/` for workflow descriptors and the orchestration runner;
5. `python_firmware_validation/`, `replay_evidence/`, and `rolling24_evidence/` for the Python-to-firmware validation path and exported contract artifacts;
6. `firmware_logs/` for representative REPLAY and FIELD execution evidence;
7. `LiteML_Edge ESP32 Electrical Schematic/` for the embedded wiring and GPIO mapping context.

## 2. Recommended reading order

For a reader who wants to understand the package efficiently, the recommended order is:

1. `START_HERE.md`
2. `PACKAGE_CONTENTS.md`
3. `README_SUPPORT_PACKAGE.md`
4. `support_guides/SOFTWARE_STACK.md`
5. `support_guides/AUTOMATION_AND_WORKSPACE.md`
6. `support_guides/FIRMWARE_BUILD_CONTEXT.md`
7. `datasets/README_DATASET.md`
8. `support_guides/HARDWARE_SETUP.md`

Then inspect the documentation and evidence layers directly:

- `core_source_reference/trainings/`
- `core_source_reference/firmwares/`
- `replay_evidence/`
- `rolling24_evidence/`
- `python_firmware_validation/`
- `firmware_logs/`

## 3. How to interpret the package

This package is intended to support implementation understanding, technical review, and technology-transfer-oriented documentation of the reported work. It is also suitable for manuscript-facing inspection because the technical-support artifacts and the paper evidence are aligned to the same contract.

The package is not presented as a host-independent universal image that automatically reproduces every original development operation with zero setup. Instead, it provides the technical documentation, preserved source references, datasets, exported artifacts, logs, and hardware context needed to inspect how LiteML-Edge is organized and how the reported results are supported.

## 4. Dependency interpretation boundary

The package documents the software stack in two complementary ways.

On the Python side, `environment/requirements.txt` and `environment/packages_report.md` document the intended training and export environment. In this package, TensorFlow and TensorFlow Model Optimization should be read as training-side dependencies expected to be installed through the requirements baseline.

On the firmware side, the preserved sources and logs document the use of TensorFlow Lite Micro as the embedded inference runtime. That runtime is part of the implementation context documented here, but its operational ready-to-use code-release packaging is deferred to the later Link-to-Code stage rather than claimed as a complete initial-submission deliverable.

## 5. Practical documentation layers

### Architecture and workflow

Use:

- `images_generator/architecture_image/`
- `pipelines/`
- `EVIDENCE_MAP.md`
- `PAPER_TO_ARTIFACT_TRACE.md`

### Data and preprocessing

Use:

- `datasets/`
- `datasets/README_DATASET.md`
- `core_source_reference/trainings/*/base_model/`

### Model-generation and export logic

Use:

- `core_source_reference/trainings/*/`
- `core_source_reference/utils/global_utils/`
- `replay_evidence/`

### Firmware-side documentation

Use:

- `core_source_reference/firmwares/`
- `support_guides/FIRMWARE_BUILD_CONTEXT.md`
- `firmware_logs/`

### Validation and deployment evidence

Use:

- `python_firmware_validation/`
- `rolling24_evidence/`
- `replay_evidence/`
- `table_I_doi_support/` when reviewing the manuscript's related-work matrix only

## 6. Minimum host-side prerequisites

For direct script inspection and limited package-side execution, the package documents:

- a `.venv` Python environment;
- the package-local dependency baseline in `environment/requirements.txt`;
- the VSCode workspace convention in `environment/LiteML.code-workspace`.

For independent firmware rebuilding or on-device retracing on another machine, the reader additionally needs host-side tooling and hardware resources such as:

- PlatformIO and the appropriate ESP32 toolchain;
- an ESP32 board compatible with the preserved firmware sources;
- the sensor and power-measurement setup reflected in the packaged schematic and logs;
- local serial-port selection and upload configuration.

That boundary is a practical host-setup condition and one reason the package should be read as technical supporting documentation for the initial submission stage rather than as the final Link-to-Code release.
