# START_HERE

This package contains the technical support documents for LiteML-Edge.

It helps editors and reviewers see how the method is organized, what files are included, and how the packaged evidence connects to the manuscript during the initial submission stage.

This package should not be interpreted as the journal's post-acceptance Link-to-Code release. Instead, it provides structured technical documentation that supports implementation understanding and technology transfer while the formal code-release stage is handled separately.

## Recommended reading order

1. Read `TECHNICAL_SUPPORT_GUIDE.md` for the package-wide technical-support overview.
2. Read `PACKAGE_CONTENTS.md` for a concise top-level manifest.
3. Read `README_SUPPORT_PACKAGE.md` for package scope and interpretation.
4. Read `PAPER_TO_ARTIFACT_TRACE.md` for direct mapping from manuscript elements to packaged artifacts.
5. Read `support_guides/TECHNICAL_REVIEW_QUICKSTART.md` for a short guided inspection path.
6. Inspect the specific artifact groups listed below, including the hardware schematic folder when reviewing the physical sensing setup.

## Fast path by documentation layer

### Workflow and architecture
Open:
- `TECHNICAL_SUPPORT_GUIDE.md`
- `pipelines/`
- `images_generator/architecture_image/`

Purpose:
- understand the contract-centered workflow;
- identify how dataset preparation, model export, replay artifacts, firmware execution, and validation are connected.

### Dataset and preprocessing
Open:
- `datasets/README_DATASET.md`
- `datasets/singapore_dataset/`
- `datasets/environment_mlp/`
- `datasets/environment_lstm/`
- `datasets/environment_Conv1D_Tiny/`

Purpose:
- inspect dataset provenance;
- inspect prepared datasets and per-model dataset preparation scripts.

### Firmware and hardware support documentation
Open:
- `core_source_reference/firmwares/`
- `support_guides/FIRMWARE_BUILD_CONTEXT.md`
- `support_guides/HARDWARE_SETUP.md`
- `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`

Purpose:
- inspect the preserved embedded logic and surrounding context;
- inspect the documented deployment hardware assumptions and GPIO mapping;
- understand that Python-side TensorFlow dependencies are documented through `environment/requirements.txt`, while TensorFlow Lite Micro is documented here as the embedded runtime context that the later Link-to-Code stage can package operationally.

## Fast path by manuscript element

### Table I
Open:
- `table_I_doi_support/table_I_final_matrix.xlsx`
- `table_I_doi_support/table_I_evidence_log.xlsx`
- `table_I_doi_support/table_I_source_index.xlsx`

Purpose:
- verify the DOI-indexed related-work evidence matrix;
- verify that check marks are assigned only when the corresponding element is explicitly reported in the reviewed source;
- verify that blank cells mean "not explicitly identified in the reviewed scope", not a negative quality judgment.

### Table IV
Open:
- `support_guides/INSPECT_TABLE_IV.md`
- the packaged model-I/O comparison workbooks
- the packaged prediction/metrics comparison workbooks
- the selected replay logs under `firmware_logs/`

Purpose:
- verify stage-wise replay conformance;
- verify that the methodology localizes agreement or divergence by stage.

### Tables V-VII
Open:
- `support_guides/INSPECT_TABLES_V_VI_VII.md`
- selected firmware logs under `firmware_logs/`
- the packaged table-generation scripts and their outputs, when included

Purpose:
- verify energy, memory, and IDLE values against the selected Replay and Field logs.

### Hardware schematic
Open:
- `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`

Purpose:
- verify the reported ESP32-based sensor-node wiring used by the LiteML-Edge hardware setup;
- verify the indoor/outdoor DHT22 mapping, shared I2C lines for OLED and INA219, USB/+5V rail labeling, and GPIO assignment table.

### Figure 2
Inspect the figure as a figure produced by the plotting script, not as a workbook-only artifact.

Primary evidence chain:
- `images_generator/graphic_image/plot_rolling24_scatter_offline_ondevice.py`
- `images_generator/graphic_image/data.xlsx`
- `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.png`
- `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.pdf`

Purpose:
- verify that Figure 2 is a visual summary produced by the plotting script of Rolling-24 offline versus on-device replay agreement;
- verify that Table IV remains the primary stage-wise conformance evidence.

## Package intent

This package is designed for:
- technical review of the method described in the manuscript;
- implementation understanding through supporting documentation;
- evidence traceability across manuscript claims and packaged artifacts;
- technology-transfer-oriented documentation at the initial submission stage.

It is not presented as a final public Link-to-Code release or as a universal zero-intervention rebuild image for every host environment.
