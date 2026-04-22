# PACKAGE_CONTENTS

This file summarizes the technical role of each top-level folder in the current LiteML-Edge package.

| Folder or file | Technical role |
|---|---|
| `START_HERE.md` | Entry point for package navigation |
| `TECHNICAL_SUPPORT_GUIDE.md` | Central guide for technical-support reading |
| `README_SUPPORT_PACKAGE.md` | Package scope, interpretation, and documentation boundary |
| `PACKAGE_CONTENTS.md` | Top-level manifest of package contents |
| `EVIDENCE_MAP.md` | Claim-to-artifact map across the package |
| `PAPER_TO_ARTIFACT_TRACE.md` | Manuscript element to artifact mapping |
| `datasets/` | Source climate files, prepared datasets, and dataset preparation scripts |
| `core_source_reference/trainings/` | Training, pruning, quantization, header generation, and scaler export source references |
| `core_source_reference/firmwares/` | Preserved firmware source references for the three model families |
| `core_source_reference/utils/` | Shared project utilities used in the packaged workflow |
| `pipelines/` | Workflow descriptors and DAG runner |
| `python_firmware_validation/` | Python-to-firmware comparison scripts, reference workbooks, and comparison workbooks |
| `replay_evidence/` | Replay headers and replay reference artifacts exported toward firmware |
| `rolling24_evidence/` | Rolling-24 samples, predictions, and metrics artifacts |
| `firmware_logs/` | Representative REPLAY and FIELD device logs |
| `LiteML_Edge ESP32 Electrical Schematic/` | Embedded wiring diagram and GPIO mapping context |
| `images_generator/` | Figure and architecture source assets |
| `tables_generator/` | Scripts and outputs for manuscript table generation |
| `table_I_doi_support/` | DOI-indexed support materials for the manuscript's related-work table |
| `support_guides/` | Focused guides for software stack, automation, hardware, and table inspection |
| `article_evidence/` | Manuscript-facing traceability notes |
| `appendix_support/` | Additional orientation notes for package reading |
| `environment/` | VSCode workspace file, Python-side dependency baseline, and observed environment snapshot |
| `REPLICATION_MODES.md` | Package use modes for inspection, regeneration, and source-level retracing |

## Main interpretation rule

For initial technical review, treat the package as a structured technical support package with aligned manuscript evidence. It should not be interpreted as the final Link-to-Code release or as a random archive of unrelated files.
