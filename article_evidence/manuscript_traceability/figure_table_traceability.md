# Figure / Table / Claim Traceability

This file provides a package-aligned traceability template for the current package.

| Manuscript evidence class | Primary package location | Notes |
|---|---|---|
| Stage-wise Python-versus-firmware validation figures or tables | `python_firmware_validation/`, `firmware_logs/` | Use the comparison workbooks and selected logs. |
| Rolling-24 evaluation figures or tables | `rolling24_evidence/` | Use metrics, predictions, and sample tables by model family. |
| Replay 2+47 figures or tables | `replay_evidence/`, `firmware_logs/` | Use replay headers, replay reference tables, and REPLAY logs. |
| Dataset and preprocessing traceability statements | `datasets/`, `core_source_reference/trainings/`, `pipelines/` | Use the source climate files, prepared datasets, and source-reference scripts. |
| Environment and workflow auditability statements | `environment/`, `pipelines/`, `support_guides/` | Use the packaged workspace, requirements, package snapshot, and guides. |
| Hardware schematic and peripheral-mapping statements | `LiteML_Edge ESP32 Electrical Schematic/`, `support_guides/HARDWARE_SETUP.md` | Use the schematic PDF and package guides to inspect sensor wiring, shared I2C lines, USB/+5V rail labeling, and GPIO assignment. |
| Firmware implementation traceability statements | `core_source_reference/firmwares/` | Use the selected firmware source references. |
