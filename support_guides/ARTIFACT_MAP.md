# Artifact Map

This document maps the main claims and manuscript-facing tables to the artifacts included in the current package.

## A. Package-level execution and environment entry points

Primary package-control artifacts:

- `environment/LiteML.code-workspace`
- `pipelines/runner.py`
- `pipelines/environment_mlp_pipeline.yaml`
- `pipelines/environment_lstm_pipeline.yaml`
- `pipelines/environment_Conv1D_pipeline.yaml`
- `environment/requirements.txt`
- `environment/packages_report.md`

These files document the intended VSCode workspace, the `.venv`-based Python environment, and the preserved workflow descriptors used by the full LiteML-Edge project. For explicit run-versioning interpretation, also see `support_guides/VERSIONING_AND_RUN_SELECTION.md`.

## B. Dataset provenance and prepared datasets

Supporting artifacts:

- `datasets/singapore_dataset/Singapore_Temp.csv`
- `datasets/singapore_dataset/Singapore_RH.csv`
- `datasets/singapore_dataset/README.rtf`
- `datasets/environment_mlp/environment_dataset_mlp.py`
- `datasets/environment_mlp/environment_dataset_mlp.csv`
- `datasets/environment_lstm/environment_dataset_lstm.py`
- `datasets/environment_lstm/environment_dataset_lstm.csv`
- `datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.py`
- `datasets/environment_Conv1D_Tiny/environment_dataset_Conv1D_Tiny.csv`

## C. Training, pruning, quantization, and export source references

### MLP

- `core_source_reference/trainings/environment_mlp/base_model/environment_base_model_mlp.py`
- `core_source_reference/trainings/environment_mlp/pruned_model/environment_pruned_model_mlp.py`
- `core_source_reference/trainings/environment_mlp/quantized_model/environment_quantized_model_mlp.py`
- `core_source_reference/trainings/environment_mlp/header_generator/header_generator_mlp.py`
- `core_source_reference/trainings/environment_mlp/scalers_exporter/scale_vector_exporter_mlp.py`

### LSTM

- `core_source_reference/trainings/environment_lstm/base_model/environment_base_model_lstm.py`
- `core_source_reference/trainings/environment_lstm/pruned_model/environment_pruned_model_lstm.py`
- `core_source_reference/trainings/environment_lstm/quantized_model/environment_quantized_model_lstm.py`
- `core_source_reference/trainings/environment_lstm/header_generator/header_generator_lstm.py`
- `core_source_reference/trainings/environment_lstm/scalers_exporter/scale_vector_exporter_lstm.py`

### Conv1D Tiny

- `core_source_reference/trainings/environment_Conv1D_Tiny/base_model/environment_base_model_Conv1D_Tiny.py`
- `core_source_reference/trainings/environment_Conv1D_Tiny/pruned_model/environment_pruned_model_Conv1D_Tiny.py`
- `core_source_reference/trainings/environment_Conv1D_Tiny/quantized_model/environment_quantized_model_Conv1D_Tiny.py`
- `core_source_reference/trainings/environment_Conv1D_Tiny/header_generator/header_generator_Conv1D_Tiny.py`
- `core_source_reference/trainings/environment_Conv1D_Tiny/scalers_exporter/scale_vector_exporter_Conv1D_Tiny.py`

## D. Exported deployment-contract artifacts

The firmware-facing contract is materialized through files such as:

- `environment_quantized_samples_replay_raw_2plus47_*.h`
- replay reference tables under `replay_evidence/`
- Rolling-24 metric and prediction exports under `rolling24_evidence/`
- reference workbooks under `python_firmware_validation/reference_workbook/`

## E. Desktop-to-firmware validation workbooks

Python reference outputs and comparison artifacts are concentrated under:

- `python_firmware_validation/reference_workbook/`
- `python_firmware_validation/workbook/`
- `python_firmware_validation/compare_scripts/` (immediate model I/O)
- `python_firmware_validation/compare_prediction_vs_log_scripts/` (predictions and Rolling-24 metrics)

Important artifact families include:

- input-reference workbooks
- decoded output-reference workbooks
- raw-output reference workbooks
- model I/O comparison workbooks
- prediction/metrics comparison workbooks

## F. Hardware schematic and peripheral mapping

Hardware-setup evidence is under:

- `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`

This artifact documents the ESP32 sensing-node wiring used by the package, including the indoor and outdoor DHT22 connections, the INA219 module, the shared I2C lines used by the OLED and INA219, the USB/+5V rail labeling, and the GPIO assignment table embedded in the schematic.

## G. Firmware execution and representative logs

Representative firmware evidence is under:

- `firmware_logs/mlp/`
- `firmware_logs/lstm/`
- `firmware_logs/Conv1D_Tiny/`

Supporting firmware source references are under:

- `core_source_reference/firmwares/mlp/`
- `core_source_reference/firmwares/lstm/`
- `core_source_reference/firmwares/Conv1D_Tiny/`

## H. Table reconstruction support

### Table IV

Use the stage-wise comparison workbooks and the comparison scripts under:

- `python_firmware_validation/compare_scripts/` (immediate model I/O)
- `python_firmware_validation/compare_prediction_vs_log_scripts/` (predictions and Rolling-24 metrics)
- `core_source_reference/utils/workbook_mlp/`
- `core_source_reference/utils/workbook_lstm/`
- `core_source_reference/utils/workbook_Conv1D_Tiny/`

### Tables V, VI, and VII

Use the firmware `device-monitor-*.log` files for on-device memory, arena, heap, latency, power, and energy readings, cross-checking against:

- `rolling24_evidence/*/environment_quantized_metrics_rolling24_*.xlsx`
- the workbook outputs under `python_firmware_validation/workbook/`

## I. Utility scripts supporting implementation traceability

Package utility modules are organized under:

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

## J. Environment evidence

The attached environment evidence consists of:

- `environment/requirements.txt` as the installation baseline
- `environment/packages_report.md` as an observed `.venv` snapshot
- `environment/LiteML.code-workspace` as the packaged workspace file


## K. Table I bibliographic audit support

Use the DOI-indexed audit folder when inspecting the related-work matrix in Table I:

- `table_I_doi_support/table_I_final_matrix.csv`
- `table_I_doi_support/table_I_final_matrix.xlsx`
- `table_I_doi_support/table_I_evidence_log.csv`
- `table_I_doi_support/table_I_evidence_log.xlsx`
- `table_I_doi_support/table_I_source_index.csv`
- `table_I_doi_support/table_I_source_index.xlsx`
- `table_I_doi_support/Table_I_Audit_Combined.xlsx`
- `table_I_doi_support/README.md`

These files provide the DOI-indexed evidence matrix, the criterion-level evidence log, and the source index used to support the conservative marking policy adopted for Table I.

## Figure 2 visual evidence support

Primary evidence for Figure 2 should be interpreted as the following figure script and source files:

- `images_generator/graphic_image/plot_rolling24_scatter_offline_ondevice.py`
- `images_generator/graphic_image/data.xlsx`
- `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.png`
- `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.pdf`
