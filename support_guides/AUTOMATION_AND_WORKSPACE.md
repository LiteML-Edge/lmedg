# Automation and Workspace Guide

This document consolidates the role of the included `environment/LiteML.code-workspace`, `pipelines/runner.py`, `environment/requirements.txt`, `environment/packages_report.md`, and the three YAML pipeline descriptors.

## 1. Why the workspace matters

`environment/LiteML.code-workspace` is included as the VSCode navigation and environment-reference file for this package. It records the intended `.venv` interpreter convention and preserves the workspace-based execution context used by the LiteML-Edge project.

## 2. Runner role

`pipelines/runner.py` is the preserved orchestrator for model-family workflow description and technical traceability.

It loads a YAML descriptor and computes a DAG-based execution plan. The attached source documents support for:

- `--pipeline`
- `--only`
- `--no-upstream`
- `--from`
- `--to`
- `--after`
- `--before`
- `--list`
- `--dry-run`
- `--keep-going`

## 3. Model-family descriptors

### MLP descriptor

`pipelines/environment_mlp_pipeline.yaml` documents the MLP workflow from dataset preparation through firmware-side monitoring.

### LSTM descriptor

`pipelines/environment_lstm_pipeline.yaml` documents the LSTM workflow from dataset preparation through firmware-side monitoring.

### Conv1D Tiny descriptor

`pipelines/environment_Conv1D_pipeline.yaml` documents the Conv1D Tiny workflow from dataset preparation through firmware-side monitoring.

## 4. Installation baseline and observed environment

Use:

- `environment/requirements.txt` as the installation baseline;
- `environment/packages_report.md` as one observed `.venv` snapshot.

This pair gives both a prescriptive dependency source and a descriptive environment record for the Python-side training and export workflow.

It should not be read as a claim that the embedded TensorFlow Lite Micro runtime is fully packaged in this initial-submission bundle. That operational firmware-side packaging belongs to the later Link-to-Code stage.

## 5. Safe inspection command

For this package, the safest direct command is a workflow-plan listing. The YAML descriptors preserve the original project-relative workflow logic and are included here as workflow descriptors and technical-support artifacts, rather than as a claim that the packaged layout mirrors the original executable repository tree path-for-path:

```powershell
python pipelines/runner.py --pipeline pipelines/environment_mlp_pipeline.yaml --list
```

Equivalent commands apply to `pipelines/environment_lstm_pipeline.yaml` and `pipelines/environment_Conv1D_pipeline.yaml`.

## 6. Package interpretation boundary

The included YAML and runner files should be read primarily as automation and workflow descriptors. The YAML files preserve the original project-relative workflow structure, while this package provides a curated set of source references and result artifacts for technical review at the initial-submission stage. They should not be interpreted as a claim that the packaged layout reproduces the original executable repository tree path-for-path or that the final public code-release stage is already bundled here.

## 7. Versioning and selector interpretation

The package contains explicit examples of training-side versioned workspaces, including `run/`, `latest.txt`, and `manifest.json`, together with the shared helper implementation in `core_source_reference/utils/global_utils/versioning.py`. These materials show how the project preserves prior output files and resolves the most recent run.

In this workflow, metrics-side run directories are not independent authoring spaces that need to be packaged in full for inspection. They are produced by execution of the corresponding training, pruning, and quantization stages and by the downstream helper flow those stages drive. For that reason, this bundle preserves the generation linkage and selector logic, while the metrics-side evidence itself is provided through the curated replay, Rolling-24, workbook, and log artifacts elsewhere in the package.
