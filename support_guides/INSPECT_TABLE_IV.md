# Inspect Table IV

Table IV is the stage-wise conformance table. In this package, the paper-level status is reconstructed by reading the included comparison artifacts for each model family.

## Package context

Use the packaged environment evidence in `environment/` and inspect the workflow descriptors under `pipelines/` as orchestration references. The current bundle is a technical support package for the initial submission stage, so the most direct evidence comes from the included workbooks and logs.

## Files to inspect

### MLP

Reference workbooks:

- `python_firmware_validation/reference_workbook/reference_workbook_mlp/environment_quantized_dbg_model_input_reference_mlp.xlsx`
- `python_firmware_validation/reference_workbook/reference_workbook_mlp/environment_quantized_dbg_model_output_reference_mlp.xlsx`
- `python_firmware_validation/reference_workbook/reference_workbook_mlp/environment_quantized_dbg_model_output_raw_reference_mlp.xlsx`

Comparison workbooks:

- `python_firmware_validation/workbook/workbook_mlp/model_io_comparison_mlp.xlsx`
- `python_firmware_validation/workbook/workbook_mlp/predictions_metrics_vs_log_comparison_mlp.xlsx`

Comparison scripts:

- `python_firmware_validation/compare_prediction_vs_log_scripts/compare_predictions_metrics_to_log_mlp.py`
- `core_source_reference/utils/workbook_mlp/compare_model_io_to_log_mlp.py`
- `core_source_reference/utils/workbook_mlp/compare_predictions_metrics_to_log_mlp.py`

### LSTM

Reference workbooks:

- `python_firmware_validation/reference_workbook/reference_workbook_lstm/environment_quantized_dbg_model_input_reference_lstm.xlsx`
- `python_firmware_validation/reference_workbook/reference_workbook_lstm/environment_quantized_dbg_model_output_reference_lstm.xlsx`
- `python_firmware_validation/reference_workbook/reference_workbook_lstm/environment_quantized_dbg_model_output_raw_reference_lstm.xlsx`

Comparison workbooks:

- `python_firmware_validation/workbook/workbook_lstm/model_io_comparison_lstm.xlsx`
- `python_firmware_validation/workbook/workbook_lstm/predictions_metrics_vs_log_comparison_lstm.xlsx`

Comparison scripts:

- `python_firmware_validation/compare_prediction_vs_log_scripts/compare_predictions_metrics_to_log_lstm.py`
- `core_source_reference/utils/workbook_lstm/compare_model_io_to_log_lstm.py`
- `core_source_reference/utils/workbook_lstm/compare_predictions_metrics_to_log_lstm.py`

### Conv1D Tiny

Reference workbooks:

- `python_firmware_validation/reference_workbook/reference_workbook_Conv1D_Tiny/environment_quantized_dbg_model_input_reference_Conv1D_Tiny.xlsx`
- `python_firmware_validation/reference_workbook/reference_workbook_Conv1D_Tiny/environment_quantized_dbg_model_output_reference_Conv1D_Tiny.xlsx`
- `python_firmware_validation/reference_workbook/reference_workbook_Conv1D_Tiny/environment_quantized_dbg_model_output_raw_reference_Conv1D_Tiny.xlsx`

Comparison workbooks:

- `python_firmware_validation/workbook/workbook_Conv1D_Tiny/model_io_comparison_Conv1D_Tiny.xlsx`
- `python_firmware_validation/workbook/workbook_Conv1D_Tiny/predictions_metrics_vs_log_comparison_Conv1D_Tiny.xlsx`

Comparison scripts:

- `python_firmware_validation/compare_scripts/compare_model_io_to_log_Conv1D_Tiny.py`
- `python_firmware_validation/compare_prediction_vs_log_scripts/compare_predictions_metrics_to_log_Conv1D_Tiny.py`
- `core_source_reference/utils/workbook_Conv1D_Tiny/compare_model_io_to_log_Conv1D_Tiny.py`
- `core_source_reference/utils/workbook_Conv1D_Tiny/compare_predictions_metrics_to_log_Conv1D_Tiny.py`

## Reconstruction logic

1. Use the input-reference workbook to confirm the input-side contract.
2. Use the decoded and raw output-reference workbooks to evaluate immediate tensor agreement.
3. Use the comparison workbooks to evaluate postprocess and final-prediction alignment.
4. Summarize each model family into the Table IV stage-status cells.

## Expected paper interpretation

- MLP: agreement across the validation path, with final status taken from the workbook summaries.
- Conv1D Tiny: agreement across the validation path, including immediate model I/O and final predictions.
- LSTM: input-side agreement with earliest persistent mismatch localized at the immediate-output/raw stage, then reflected in downstream prediction comparison.
