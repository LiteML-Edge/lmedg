# REPLICATION_MODES

This note clarifies the intended use modes of the packaged materials.

## 1. Inspection mode

Goal:
- inspect the methodology and manuscript evidence without rerunning the full workflow.

Use:
- `START_HERE.md`
- `TECHNICAL_SUPPORT_GUIDE.md`
- package guides and maps
- packaged workbooks, logs, replay headers, and source references

Expected outcome:
- understand the reported technology and verify the included files.

## 2. Table-regeneration mode

Goal:
- regenerate manuscript tables from packaged workbooks and logs.

Use:
- packaged workbook-comparison outputs;
- selected firmware logs;
- packaged table-generation scripts.

Expected outcome:
- reproduce the main manuscript tables from the packaged evidence.

## 3. Source-level retrace mode

Goal:
- retrace selected Python or firmware stages within the packaged workflow scope.

Use:
- `pipelines/runner.py` and the packaged YAML workflow files;
- the included training, utility, and firmware source references;
- the documented Python environment under `environment/`;
- the replay artifacts, workbooks, and logs used by the packaged validation path.

Expected outcome:
- inspect or retrace targeted workflow stages and regenerate selected artifacts within the current package scope.

## Scope note

The package is designed to support technical review, implementation understanding, and selected retracing. It is not presented as the final Link-to-Code release or as a universal zero-edit rebuild image for every host environment. Host-side provisioning and hardware availability still apply when an external reader attempts full on-device rebuilding.
