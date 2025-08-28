# BDD Dataset Analysis Tool

A comprehensive Python tool for analyzing the Berkeley DeepDrive (BDD) dataset for object detection tasks. This tool provides in-depth analysis of class distributions, bounding box statistics, environmental conditions, data quality checks, and generates detailed reports with visualizations.

## Features

- **Class Distribution Analysis**: Object counts and distribution across 10 detection classes
- **Bounding Box Statistics**: Area, aspect ratio, and geometric analysis
- **Environmental Analysis**: Performance across weather, time of day, and scene conditions
- **Data Quality Validation**: Detection of invalid annotations and out-of-bounds boxes
- **Anomaly Detection**: Identification of outliers and unusual patterns
- **Scenario Coverage**: Analysis of class × environment combinations
- **Image Quality Metrics**: Blur, brightness, and contrast analysis (optional)
- **Comprehensive Reporting**: Automated markdown reports with actionable insights
- **Rich Visualizations**: Publication-ready charts and statistical plots

## Installation

### Prerequisites

```bash
Python 3.7+
```

### Dependencies

install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Ensure your BDD dataset is organized as follows:

```
your_bdd_directory/
├── labels/
│   ├── bdd100k_labels_images_train.json
│   └── bdd100k_labels_images_val.json
└── images/100k/  # Optional for image quality analysis
    ├── train/
    │   ├── image1.jpg
    │   └── ...
    └── val/
        ├── image1.jpg
        └── ...
```

## Quick Start

### Basic Usage

```python
from bdd_analyzer import BDDDatasetAnalyzer

# Initialize analyzer
analyzer = BDDDatasetAnalyzer(
    data_dir="path/to/bdd_dataset",
    output_dir="analysis_results"
)

# Run complete analysis
analyzer.run_complete_analysis()
```

## Output Files

The analysis generates the following outputs in your specified directory:

### Reports and Statistics
- `BDD_Dataset_Analysis_Report.md` - Comprehensive analysis report
- `detailed_statistics.json` - All analysis results in JSON format
- `analysis.log` - Execution log with timestamps

### Data Quality Files
- `data_quality_issues.csv` - Invalid/out-of-bounds annotations
- `object_attributes.json` - Occluded/truncated/crowd statistics
- `negative_frames.csv` - Images with zero detection objects

### Coverage Analysis
- `scenario_coverage_counts.csv` - Class × environment coverage matrix
- `scenario_coverage_lowcells.csv` - Under-represented scenarios

### Image Quality (Optional)
- `image_quality_train.csv` - Training set image quality metrics
- `image_quality_val.csv` - Validation set image quality metrics

### Visualizations
- `class_distribution_comparison.png` - Train vs validation class distribution
- `bbox_area_by_class_top5_boxplot.png` - Bounding box area analysis
- `weather_distribution_training.png` - Environmental condition analysis
- `aspect_ratio_distribution_training.png` - Aspect ratio distributions
- `train_val_split_analysis.png` - Dataset split comparison
- `dataset_summary_table.png` - Key statistics summary

## Detection Classes

The tool analyzes these 10 BDD object detection classes:

1. **traffic light**
2. **traffic sign** 
3. **person**
4. **bike**
5. **truck**
6. **motor** (motorcycle)
7. **car**
8. **train**
9. **rider**
10. **bus**

## Configuration

### Environment Variables

```bash
# Dataset paths
BDD_DATA_DIR="path/to/bdd/dataset"      # Required: Path to labels directory
BDD_OUTPUT_DIR="analysis_results"       # Optional: Output directory
BDD_IMAGES_DIR="path/to/images/100k"    # Optional: For image quality analysis
```

## Example Output

### Sample Analysis Report

```markdown
# BDD Dataset Analysis Report

## Executive Summary
- Training Images: 70,000
- Validation Images: 10,000
- Total Training Objects: 1,395,102
- Total Validation Objects: 200,718

## Key Findings
### Class Imbalance
- Most frequent class: car with 965,435 instances
- Least frequent class: train with 2,045 instances
- Imbalance ratio: 472.0:1

### Anomalies Detected
- car: 15,234 outliers (1.58%)
- person: 8,945 outliers (2.34%)
```
## Results

# Dataset summary
![Class Distribution](bdd_analysis_results\dataset_summary_table.png)

# Train val split analysis
![Class Distribution](bdd_analysis_results\train_val_split_analysis.png)

# class distribution comparison
![Class Distribution](bdd_analysis_results\class_distribution_comparison.png)

- Cars, traffic signs, traffic lights dominate.  
- Rare classes (train, rider, motor, bike, bus) <1%.

# bounding box area distribution

![Class Distribution](bdd_analysis_results\bbox_area_distribution.png)

- Most categories show a long-tailed distribution → many small objects, very few large ones.
- Traffic lights & signs: overwhelmingly tiny, which makes detection difficult.
- Persons, riders, bikes, motors: mostly small/medium; also fewer samples → harder to learn.
- Cars: largest and most diverse category (~713k boxes), spanning tiny to very large.
- Trucks & buses: fewer instances but generally large, easier to detect.
- Train: extremely rare (~136 boxes), contributes little to training.

# Bounding box area by class (top 5)

![Class Distribution](bdd_analysis_results\bbox_area_by_class_top5_boxplot.png)
- Tiny objects (traffic lights/signs) → hardest for YOLOv8n.  
- Cars/trucks have larger and more varied boxes.

# Aspect ratio distribution
![Class Distribution](bdd_analysis_results\aspect_ratio_distribution_training.png)

- Most boxes near-square or vertical.  
- Very few elongated boxes.

# Objects per image distribution
![Class Distribution](bdd_analysis_results\objects_per_image_distribution.png)
- 10–40 objects per image typical.  
- Dense scenes → higher NMS conflict risk.

# Weather distribution
![Class Distribution](bdd_analysis_results\weather_distribution_heatmap.png)

- Clear weather dominates (51%). Foggy/snowy/rainy underrepresented.  
- Likely weaker generalization to rare weather conditions.
- Further plan is to choose different models according to various weather conditions, so that it could give spectrum of results with best suitable accuracies
## Contact

- Author: Vandana R D (DDB1COB)