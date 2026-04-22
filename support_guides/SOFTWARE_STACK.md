# Software Stack

This file records the software and toolchain information documented by the current LiteML-Edge technical support package.

It has five complementary goals:

1. document the intended `.venv`-based Python environment;
2. document the included VSCode workspace context;
3. document the preserved automation descriptors and source-reference utilities; and
4. document the main third-party Python packages exposed by `environment/requirements.txt` and `environment/packages_report.md`; and
5. distinguish the Python-side training dependencies from the embedded TensorFlow Lite Micro runtime that is only documented here as firmware context.

## Package-level execution environment

The documented execution environment for package inspection is:

- **Visual Studio Code** as the primary IDE
- `environment/LiteML.code-workspace` as the packaged workspace file
- a **project-local Python virtual environment** named `.venv`
- dependency installation from `environment/requirements.txt`
- workflow inspection through `pipelines/runner.py` and the YAML pipeline files

This package-level environment description is intentionally about the Python-side training, export, and inspection workflow. It should not be read as a claim that the embedded TensorFlow Lite Micro runtime is already packaged here as the final operational Link-to-Code release.

### Practical inspection contract

For package inspection, the intended order is:

1. open the package root in VSCode
2. create or reuse `.venv`
3. let VSCode select the `.venv` interpreter
4. install the Python dependencies from `environment/requirements.txt`
5. inspect the dataset, replay, Rolling-24, validation, and source-reference artifacts

## Training-side dependencies versus embedded runtime

The packaged environment files serve two different purposes.

- `environment/requirements.txt` and `environment/packages_report.md` document the Python-side environment used for dataset preparation, training, pruning, quantization, export, workbook generation, and related inspection tasks. In this context, TensorFlow-based dependencies are expected to be installed through the requirements baseline.
- The firmware-side sources and logs document the use of TensorFlow Lite Micro on the embedded target. In the current package, TensorFlow Lite Micro is described as part of the firmware implementation context, while its operational ready-to-use distribution is reserved for the later Link-to-Code stage.

## Workspace-backed execution context

The included `environment/LiteML.code-workspace` defines the `.venv` interpreter convention and terminal activation behavior. It should be treated as environment evidence, even though this package does not mirror the full executable repository tree or the later Link-to-Code release structure.

## Automation entry points preserved in the package

The current package exposes the following workflow files:

- `pipelines/runner.py`
- `pipelines/environment_mlp_pipeline.yaml`
- `pipelines/environment_lstm_pipeline.yaml`
- `pipelines/environment_Conv1D_pipeline.yaml`

These files document the model-family orchestration structure used by the LiteML-Edge project. Within this package, they should be interpreted primarily as workflow descriptors and implementation-support artifacts.

For direct package inspection, the safest usage is a non-destructive plan listing such as `python pipelines/runner.py --pipeline pipelines/environment_mlp_pipeline.yaml --list`. This preserves the inspection function of the descriptors without implying that the current package is a full standalone rebuild tree.

A direct firmware rebuild on another host would additionally require host-side prerequisites such as PlatformIO and toolchain provisioning, ESP32 hardware availability, and local serial-port selection. See `support_guides/FIRMWARE_BUILD_CONTEXT.md` for the firmware-side boundary.

## Core Python packages documented by the package

The included environment files document the Python-side use of packages such as:

- `numpy`
- `pandas`
- `scikit-learn`
- `openpyxl`
- `matplotlib`
- `tensorflow`
- `tensorflow_model_optimization`
- `platformio`

These entries document the host-side Python stack and related tooling. They do not by themselves claim that the embedded TensorFlow Lite Micro runtime has already been packaged here as a complete public code-release bundle.

## Preserved project-side Python modules

Selected project modules are included under:

- `core_source_reference/utils/global_utils/`

Representative internal modules include:

- `global_seed.py`
- `paths_mlp.py`
- `paths_lstm.py`
- `paths_Conv1D_Tiny.py`
- `versioning.py`
- `report_env.py`
- `pio_build.py`
- `pio_monitor.py`
- `pio_pull_headers.py`
- `pio_upload.py`

## Workbook comparison and validation scripts

The package also includes comparison scripts and workbook helpers under:

- `python_firmware_validation/compare_scripts/` (immediate model I/O)
- `python_firmware_validation/compare_prediction_vs_log_scripts/` (predictions and Rolling-24 metrics)
- `core_source_reference/utils/workbook_mlp/`
- `core_source_reference/utils/workbook_lstm/`
- `core_source_reference/utils/workbook_Conv1D_Tiny/`

These scripts are part of the technical-support trail used to document the stage-wise Python-versus-firmware validation path in the paper.
