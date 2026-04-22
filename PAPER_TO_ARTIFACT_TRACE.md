# PAPER_TO_ARTIFACT_TRACE

This file maps key manuscript elements to the packaged artifacts used to inspect them.

| Paper element | Manuscript location | Primary supporting artifact(s) | What to inspect |
|---|---|---|---|
| Table I | Related Work / Table I | `table_I_doi_support/table_I_final_matrix.*`, `table_I_doi_support/table_I_evidence_log.*`, `table_I_doi_support/table_I_source_index.*` | DOI-indexed evidence matrix, criterion-level evidence, source list |
| Figure 1 | Methodology / framework figure | `images_generator/architecture_image/Framework LiteML-Edge.png`, `images_generator/architecture_image/Framework LiteML-Edge.vsdx`, and surrounding manuscript text | architecture source asset, editable source, and contract-centered workflow depiction |
| Figure 2 | Results / visual evidence for final-prediction agreement | `images_generator/graphic_image/plot_rolling24_scatter_offline_ondevice.py`, `images_generator/graphic_image/data.xlsx`, `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.png`, `images_generator/graphic_image/rolling24_scatter_offline_ondevice_T_in_ab.pdf` | script logic, data source, and rendered figure |
| Table IV | Results / replay conformance table | packaged model-I/O comparison workbooks, prediction/metrics comparison workbooks, selected replay logs | stage-wise agreement and localized divergence evidence |
| Tables V-VII | Cost and deployment tables | selected Replay and Field logs, packaged table-generation scripts, and generated table outputs when included | energy, latency, memory, heap headroom, and IDLE baselines |
| Replay headers | Methodology and replay evidence | packaged replay headers and replay reference workbooks | replay payload exported to firmware |
| Firmware validation path | Validation workflow | selected firmware logs, compare scripts, and preserved firmware implementation sources | Python-to-firmware comparison path |
| Hardware setup context | Hardware setup / sensor-node context | `LiteML_Edge ESP32 Electrical Schematic/LiteML_Edge ESP32 Electrical Schematic.pdf`, `support_guides/HARDWARE_SETUP.md` | ESP32 sensing-node wiring, shared I2C lines, USB/+5V rail labeling, and GPIO assignment table |
| Implementation reading path | Package orientation | `TECHNICAL_SUPPORT_GUIDE.md`, `PACKAGE_CONTENTS.md`, `support_guides/SOFTWARE_STACK.md`, `support_guides/FIRMWARE_BUILD_CONTEXT.md` | current package scope for technical review, implementation understanding, and host-side prerequisites |

## Figure 2 note

Figure 2 should be read as a figure produced by the plotting script. The associated workbook is an input artifact, not the sole primary evidence item. The verification chain is:

1. plotting script,
2. input Excel data,
3. exported figure file,
4. manuscript placement and caption.
