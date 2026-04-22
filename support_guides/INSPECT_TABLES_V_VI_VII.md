# Inspect Tables V, VI, and VII

Tables V, VI, and VII are deployment-oriented tables. In this package, the values are reconstructed primarily from the included firmware logs and then cross-checked against selected workbook and Rolling-24 artifacts.

## Main firmware logs

### MLP

- `firmware_logs/mlp/device-monitor-*.log`

### LSTM

- `firmware_logs/lstm/device-monitor-*.log`

### Conv1D Tiny

- `firmware_logs/Conv1D_Tiny/device-monitor-*.log`

## What these logs expose

The logs contain lines with tags such as:

- `[TFLM]`
- `[ARENA]`
- `[MEM]`
- `[quant]`
- `[MODEL]`
- `[BENCH]`
- `[PWR]`

These lines are the direct sources for model memory footprint, arena allocation, heap observations, latency, current, power, and accumulated energy values.

## Cross-check files

Use the following as secondary support:

### MLP

- `rolling24_evidence/mlp/environment_quantized_metrics_rolling24_mlp.xlsx`
- `python_firmware_validation/workbook/workbook_mlp/predictions_metrics_vs_log_comparison_mlp.xlsx`

### LSTM

- `rolling24_evidence/lstm/environment_quantized_metrics_rolling24_lstm.xlsx`
- `python_firmware_validation/workbook/workbook_lstm/predictions_metrics_vs_log_comparison_lstm.xlsx`

### Conv1D Tiny

- `rolling24_evidence/Conv1D_Tiny/environment_quantized_metrics_rolling24_Conv1D_Tiny.xlsx`
- `python_firmware_validation/workbook/workbook_Conv1D_Tiny/predictions_metrics_vs_log_comparison_Conv1D_Tiny.xlsx`

## Reconstruction procedure

1. Open the representative `device-monitor-*.log` file for the model family.
2. Extract the `[MEM]` and `[ARENA]` values for model and arena footprint.
3. Extract `[BENCH]` and INA219-related values for latency, current, power, and energy.
4. Use the workbook and Rolling-24 artifacts as consistency checks for the final paper table entries.
5. Record the selected representative values in Tables V, VI, and VII.
