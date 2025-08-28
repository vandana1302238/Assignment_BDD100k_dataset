# BDD Dataset Analysis Report — Enhanceds

## Executive Summary

This report provides a comprehensive analysis of the Berkeley
 DeepDrive (BDD) dataset for object detection tasks, extended
 with data quality checks, drift analysis, spatial/context
 insights, and model-ready priors.

### Dataset Overview
- **Training Images**: 69,863
- **Validation Images**: 10,000
- **Total Training Objects**: 1,286,871
- **Total Validation Objects**: 185,526

## Class Distribution Analysis

### Training Set Distribution
- **car**: 713,211 objects (55.42%)
- **traffic sign**: 239,686 objects (18.63%)
- **traffic light**: 186,117 objects (14.46%)
- **person**: 91,349 objects (7.10%)
- **truck**: 29,971 objects (2.33%)
- **bus**: 11,672 objects (0.91%)
- **bike**: 7,210 objects (0.56%)
- **rider**: 4,517 objects (0.35%)
- **motor**: 3,002 objects (0.23%)
- **train**: 136 objects (0.01%)

### Validation Set Distribution
- **car**: 102,506 objects (55.25%)
- **traffic sign**: 34,908 objects (18.82%)
- **traffic light**: 26,885 objects (14.49%)
- **person**: 13,262 objects (7.15%)
- **truck**: 4,245 objects (2.29%)
- **bus**: 1,597 objects (0.86%)
- **bike**: 1,007 objects (0.54%)
- **rider**: 649 objects (0.35%)
- **motor**: 452 objects (0.24%)
- **train**: 15 objects (0.01%)

## Key Findings

### Class Imbalance
- Most frequent class: **car**
 with 713,211 instances
- Least frequent class: **train**
with 136 instances
- Imbalance ratio: 5244.2:1

### Anomalies Detected

#### Train Set Anomalies
- **traffic light**: 16586 outliers (8.91%)
  - Very small objects:21196
  - Very large objects: 17
- **traffic sign**: 26137 outliers (10.90%)
  - Very small objects:10296
  - Very large objects: 128
- **person**: 9599 outliers (10.51%)
  - Very small objects:796
  - Very large objects: 487
- **bike**: 774 outliers (10.74%)
  - Very small objects:26
  - Very large objects: 74
- **truck**: 4203 outliers (14.02%)
  - Very small objects:55
  - Very large objects: 4350
- **motor**: 342 outliers (11.39%)
  - Very small objects:35
  - Very large objects: 74
- **car**: 103612 outliers (14.53%)
  - Very small objects:10585
  - Very large objects: 35532
- **train**: 19 outliers (13.97%)
  - Very small objects:5
  - Very large objects: 31
- **rider**: 548 outliers (12.13%)
  - Very small objects:54
  - Very large objects: 102
- **bus**: 1714 outliers (14.68%)
  - Very small objects:14
  - Very large objects: 2032

#### Val Set Anomalies
- **traffic light**: 2464 outliers (9.16%)
  - Very small objects:3157
  - Very large objects: 0
- **traffic sign**: 3806 outliers (10.90%)
  - Very small objects:1413
  - Very large objects: 26
- **person**: 1396 outliers (10.53%)
  - Very small objects:131
  - Very large objects: 69
- **bike**: 104 outliers (10.33%)
  - Very small objects:7
  - Very large objects: 7
- **truck**: 624 outliers (14.70%)
  - Very small objects:12
  - Very large objects: 615
- **motor**: 51 outliers (11.28%)
  - Very small objects:7
  - Very large objects: 11
- **car**: 14908 outliers (14.54%)
  - Very small objects:1444
  - Very large objects: 5128
- **train**: 2 outliers (13.33%)
  - Very small objects:0
  - Very large objects: 3
- **rider**: 82 outliers (12.63%)
  - Very small objects:7
  - Very large objects: 9
- **bus**: 233 outliers (14.59%)
  - Very small objects:0
  - Very large objects: 283

## Data Quality & Attributes
- Invalid/Out-of-bounds/Empty boxes detected: **0** (see `data_quality_issues.csv`).
- Occluded/Truncated/Crowd distributions saved to `object_attributes.json`.

## Scenario Coverage & Hard Negatives
- Scenario coverage table: `scenario_coverage_counts.csv` (low-count cells in `scenario_coverage_lowcells.csv`).
- Images with zero detection classes logged to `negative_frames.csv`.

## Recommendations

1. **Addressing Class Imbalance**: Consider data
        augmentation or weighted sampling for underrepresented classes
2. **Quality Control**: Review outlier samples and flagged
quality issues for potential label errors
3. **Environmental Bias**: Ensure model training accounts
for weather/time-of-day/scene distributions
4. **Scale Handling**: Implement multi-scale training
and small-object strategies given S/M/L mix
6. **Crowding**: Prefer Soft-NMS / cluster-NMS when same-class IoU is high
