# Singapore Dataset README for LiteML-Edge

*Dataset note for the LiteML-Edge technical-support submission package*

## Purpose

This document describes the origin, construction logic, and auditability boundaries of the Singapore environmental dataset used in LiteML-Edge. It is intended to support technical inspection, dataset traceability, and independent review of the dataset preparation methodology included in the submission package.

## Source Files

The dataset is derived from two original synchronized CSV files:

- `Singapore_Temp.csv`
- `Singapore_RH.csv`

These files contain timestamped temperature and relative humidity measurements collected in Singapore.

## Original Column Semantics

Each original file contains one `datetime` column, one outdoor measurement column without numeric suffix, and multiple indoor sensor columns identified by numeric suffixes.

- Outdoor temperature: `T(C)`
- Outdoor relative humidity: `RH(%)`
- Indoor temperature sensors: `T(C)_11 ... T(C)_15`
- Indoor relative humidity sensors: `RH(%)_11 ... RH(%)_15`

## Final Variable Mapping

- `T_out = T(C)`
- `H_out = RH(%)`
- `T_in = mean(T(C)_11 ... T(C)_15)`
- `H_in = mean(RH(%)_11 ... RH(%)_15)`

Indoor values are aggregated using arithmetic mean across the available indoor sensors at each timestamp.

## Dataset Construction Procedure

1. Load the original temperature and humidity CSV files.
2. Identify outdoor columns as the columns without numeric suffix.
3. Identify indoor sensor columns as the columns with underscore plus numeric suffix.
4. Compute `T_in` and `H_in` using NaN-safe arithmetic mean across available indoor sensors at each timestamp.
5. Merge the temperature and humidity tables using an inner join on `datetime`.
6. Remove rows with missing outdoor values.
7. Remove rows where indoor aggregation is undefined because all indoor sensor values are missing.
8. Keep the final ordered columns: `datetime`, `T_out`, `T_in`, `H_out`, `H_in`.
9. Round all numeric values to two decimal places.

## Missing-Data Policy

- If some indoor sensors are missing at a timestamp, aggregation is performed using the available indoor values.
- If all indoor sensor values for a variable are missing at a timestamp, that timestamp is discarded.
- Timestamps with missing outdoor values are discarded.
- No interpolation is applied.
- No smoothing or filtering is applied during this base dataset preparation stage.

## Temporal Alignment

Temperature and humidity tables are merged by strict inner temporal intersection on `datetime`. This preserves only timestamps that exist in both source files and avoids temporal drift or artificial alignment.

## Final Dataset Structure

- `datetime`
- `T_out`
- `T_in`
- `H_out`
- `H_in`

All numerical values are rounded to two decimal places.

## Resulting Sample Count

The resulting synchronized dataset contains **9,932 valid observations** after removal of incomplete timestamps.

## Methodological Rationale

- Indoor environmental state is represented by arithmetic mean across indoor sensors.
- Outdoor measurements remain unaggregated because only one outdoor column is used per variable.
- Strict temporal intersection preserves synchronized multivariate observations.
- No interpolation, filtering, or smoothing is applied at this base stage.
- Deterministic rounding to two decimal places helps maintain consistency across downstream artifacts.

## Auditability and Deterministic Reconstruction

- No stochastic operations are used.
- Aggregation follows a fixed arithmetic rule.
- Temporal alignment uses a fixed inner-join rule.
- Missing-data handling is rule-based.
- Final rounding is deterministic.

As a result, the final CSV can be reconstructed consistently from the original source files when the same procedure is applied.

## Relationship to LiteML-Edge

This dataset serves as the environmental reference basis for the LiteML-Edge workflow. The prepared per-model datasets, replay artifacts, and downstream validation materials derive from this base data layer through model-specific preparation scripts.

## Included Scope

This note is intended as the primary dataset-orientation document for the package. It supports inspection of dataset provenance and preparation logic without requiring the full development repository.
