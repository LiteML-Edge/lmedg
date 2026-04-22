# Technical Review Quickstart

This guide is the fastest orientation for reading the current LiteML-Edge technical support package.

## 1. Open the package correctly

Open the package root in Visual Studio Code. The included workspace file is:

- `environment/LiteML.code-workspace`

It preserves the intended `.venv` interpreter convention and the VSCode-based workflow context used by the LiteML-Edge project.

## 2. Use the documented Python environment

The documented Python environment is the package-local `.venv` convention.

Recommended setup for Python-side inspection and training-environment documentation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

## 3. Verify the environment snapshot

The package exposes two complementary Python-environment artifacts:

- `environment/requirements.txt` as the intended installation baseline
- `environment/packages_report.md` as an observed `.venv` package snapshot

These files document the host-side TensorFlow-based training environment. The embedded TensorFlow Lite Micro runtime is instead documented through the firmware-side sources and logs and is not presented here as a ready-to-use Link-to-Code bundle.

## 4. Inspect the documentation layer first

Start with:

- `TECHNICAL_SUPPORT_GUIDE.md`
- `PACKAGE_CONTENTS.md`
- `datasets/README_DATASET.md`
- `support_guides/FIRMWARE_BUILD_CONTEXT.md`
- `support_guides/HARDWARE_SETUP.md`

## 5. Choose the inspection depth

### Minimal technical review

Use this when the goal is to inspect the methodology and the evidence behind the paper without rerunning the full workflow.

Inspect:

- `README_SUPPORT_PACKAGE.md`
- `EVIDENCE_MAP.md`
- `rolling24_evidence/`
- `replay_evidence/`
- `python_firmware_validation/`
- `firmware_logs/`
- `core_source_reference/firmwares/`
- `core_source_reference/trainings/`
- `table_I_doi_support/` when reviewing the related-work matrix only
- the guides under `support_guides/`

### Source-trace inspection

Use this when the goal is to inspect how the included included files relate to the method described in the manuscript.

Primary areas:

- `datasets/`
- `core_source_reference/trainings/`
- `core_source_reference/utils/`
- `core_source_reference/firmwares/`
- `pipelines/`

## 6. Inspect the workflow descriptors

The preserved pipeline descriptors are:

- `pipelines/environment_mlp_pipeline.yaml`
- `pipelines/environment_lstm_pipeline.yaml`
- `pipelines/environment_Conv1D_pipeline.yaml`

Useful planning commands:

```powershell
python pipelines/runner.py --pipeline pipelines/environment_mlp_pipeline.yaml --list
python pipelines/runner.py --pipeline pipelines/environment_lstm_pipeline.yaml --list
python pipelines/runner.py --pipeline pipelines/environment_Conv1D_pipeline.yaml --list
```

In this package, the YAML descriptors should be read as workflow-preservation artifacts that document the original project-relative orchestration logic. The `--list` mode is the recommended direct use because this package is curated for technical review at the initial-submission stage, not presented as the final standalone Link-to-Code tree.

## 6A. Inspect versioning and run selection

To inspect how saved output workspaces are versioned and how the most recent output is resolved, review:

- `core_source_reference/trainings/*/*/run/`
- `core_source_reference/trainings/*/*/latest.txt`
- `core_source_reference/trainings/*/*/run/manifest.json`
- `core_source_reference/utils/global_utils/versioning.py`
- `pipelines/*.yaml` for `--selector latest` and `--runs-base`

## 7. Inspect the firmware and hardware context

Open:

- `core_source_reference/firmwares/`
- `support_guides/FIRMWARE_BUILD_CONTEXT.md`
- `support_guides/HARDWARE_SETUP.md`
- `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`

## 8. Package orientation path

For the quickest orientation path, use this order:

1. `START_HERE.md`
2. `TECHNICAL_SUPPORT_GUIDE.md`
3. `PACKAGE_CONTENTS.md`
4. `README_SUPPORT_PACKAGE.md`
5. `EVIDENCE_MAP.md`
6. `PAPER_TO_ARTIFACT_TRACE.md`
7. `datasets/`
8. `core_source_reference/`
9. `replay_evidence/`
10. `rolling24_evidence/`
11. `python_firmware_validation/`
12. `firmware_logs/`
13. `table_I_doi_support/`

## Additional entry points

For a manuscript-oriented inspection path, also open:

- `START_HERE.md`
- `PAPER_TO_ARTIFACT_TRACE.md`
- `REPLICATION_MODES.md`

## Figure 2 evidence note

Figure 2 should be inspected as a figure produced by the plotting script. The associated workbook is an input artifact, but the primary evidence chain is:

- `images_generator/graphic_image/plot_rolling24_scatter_offline_ondevice.py`
- `images_generator/graphic_image/data.xlsx`
- `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.png`
- `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.pdf`
