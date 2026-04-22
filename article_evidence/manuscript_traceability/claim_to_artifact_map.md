# CLAIM_TO_ARTIFACT_MAP

This file maps the main classes of manuscript claims to the technical-included files included in this package.

| Claim class | Primary evidence | Supporting files |
|---|---|---|
| The package documents model-family workflow automation | `pipelines/runner.py`, the three YAML descriptors | `support_guides/AUTOMATION_AND_WORKSPACE.md` |
| The intended Python environment is `.venv` under VSCode | `environment/requirements.txt`, `environment/packages_report.md`, `environment/LiteML.code-workspace` | `support_guides/SOFTWARE_STACK.md`, `support_guides/TECHNICAL_REVIEW_QUICKSTART.md` |
| Dataset provenance and per-model dataset preparation are inspectable | `datasets/singapore_dataset/`, `datasets/environment_mlp/`, `datasets/environment_lstm/`, `datasets/environment_Conv1D_Tiny/` | `support_guides/ARTIFACT_MAP.md` |
| Python-versus-firmware immediate I/O comparison is part of the methodology | `python_firmware_validation/compare_scripts/` (immediate model I/O), `python_firmware_validation/reference_workbook/`, `python_firmware_validation/workbook/` | `firmware_logs/`, `support_guides/ARTIFACT_MAP.md` |
| Rolling-24 is the evaluation protocol used to summarize prediction quality | `rolling24_evidence/*/environment_quantized_metrics_rolling24_*.xlsx`, `rolling24_evidence/*/environment_quantized_predictions_rolling24_*.xlsx` | `rolling24_evidence/*/environment_quantized_samples_rolling24_*.xlsx` |
| Replay 2+47 is exported into firmware and inspectable | `replay_evidence/*/environment_quantized_samples_replay_raw_2plus47_*.h`, `replay_evidence/*/environment_quantized_replay_reference_2plus47_*.xlsx` | selected replay monitor logs under `firmware_logs/` |
| Firmware execution is inspectable in REPLAY and FIELD modes | `firmware_logs/*/*.log` | selected firmware source references under `core_source_reference/firmwares/` |
| Hardware wiring and peripheral-to-GPIO mapping are inspectable | `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf` | `support_guides/HARDWARE_SETUP.md`, `support_guides/ARTIFACT_MAP.md` |
| Utility code exists to prepare headers, inspect versioned runs, and compare logs/workbooks | `core_source_reference/utils/global_utils/`, `core_source_reference/utils/workbook_*/` | pipeline YAML files |

## Notes

- This package is an initial-submission technical support package, not the final Link-to-Code release.
- The included dataset layer is part of the documented methodology context and should be considered when evaluating transferability and contract traceability.
