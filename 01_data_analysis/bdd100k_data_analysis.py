"""
BDD Dataset Analysis for Object Detection
This module provides comprehensive analysis of the BDD dataset
for object detection tasks.It analyzes the distribution
of training samples, train/val splits, identifies anomalies,
and creates visualizations.
Author: Vandana R D -DDB1COB
"""

from __future__ import annotations

import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import argparse


class BDDDatasetAnalyzer:
    """
    Comprehensive analyzer for BDD dataset object detection task.

    This class handles parsing, analysis, and visualization of the BDD dataset
    focusing on the 10 object detection classes with bounding boxes.
    """

    # Default BDD image size (used when actual image dims are unavailable)
    DEFAULT_W = 1280
    DEFAULT_H = 720

    def __init__(self, data_dir: str, output_dir:
                 str = "analysis_output",
                 images_dir: Optional[str] = None):
        """
        Initialize the BDD Dataset Analyzer.

        Args:
            data_dir (str): Path to the BDD dataset
            directory (containing labels/)
            output_dir (str): Directory to save
            analysis outputs
            images_dir (Optional[str]): Path to
            images/100k (with train/val). Optional.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.train_json_path = \
            self.data_dir / "bdd100k_labels_images_train.json"
        self.val_json_path = \
            self.data_dir / "bdd100k_labels_images_val.json"

        # BDD Object Detection Classes
        self.detection_classes = [
            'traffic light', 'traffic sign', 'person', 'bike', 'truck',
            'motor', 'car', 'train', 'rider', 'bus'
        ]
        self.images_dir = Path(images_dir)

        self.logger = self._setup_logging()
        self.train_data: List[Dict[str, Any]] = []
        self.val_data: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}

    # Setup / IO
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("bdd_analyzer")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(self.output_dir / 'analysis.log')
        sh = logging.StreamHandler()
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def load_dataset(self) -> None:
        """
        Load and parse the BDD dataset JSON files.

        Loads both training and validation annotations and filters for
        object detection relevant data.
        """
        self.logger.info("Loading BDD dataset...")

        # Load training data
        if self.train_json_path.exists():
            with open(self.train_json_path,
                      'r', encoding="utf-8") as f:
                self.train_data = json.load(f)
            self.logger.info(
                f"Loaded {len(self.train_data)} training samples")
        else:
            self.logger.error("Training "
                              f"labels not found at {self.train_json_path}")

        # Load validation data
        if self.val_json_path.exists():
            with open(self.val_json_path, 'r', encoding="utf-8") as f:
                self.val_data = json.load(f)
            self.logger.info(f"Loaded {len(self.val_data)} validation samples")
        else:
            self.logger.error("Validation "
                              f"labels not found at {self.val_json_path}")

    # Parsing
    def parse_annotations(self, data: List[Dict]) -> pd.DataFrame:
        """
        Parse BDD annotations into a structured DataFrame.

        Args:
            data (List[Dict]): Raw BDD annotation data

        Returns:
            pd.DataFrame: Structured annotation data
            (row per object; placeholder rows for empty images)
        """
        parsed_data: List[Dict[str, Any]] = []

        for sample in data:
            image_name = sample.get('name')
            image_attrs = sample.get('attributes', {}) or {}

            # Extract image-level attributes
            weather = image_attrs.get('weather', 'unknown')
            scene = image_attrs.get('scene', 'unknown')
            timeofday = image_attrs.get('timeofday', 'unknown')

            # Parse object labels
            labels = sample.get('labels', []) or []
            det_labels = [lab for lab in
                          labels if lab.get('category')
                          in self.detection_classes]

            if not det_labels:  # No objects in image
                parsed_data.append({
                    'image_name': image_name,
                    'weather': weather,
                    'scene': scene,
                    'timeofday': timeofday,
                    'category': None,
                    'bbox_x1': None, 'bbox_y1': None,
                    'bbox_x2': None, 'bbox_y2': None,
                    'bbox_width': None, 'bbox_height': None,
                    'bbox_area': None,
                    'occluded': None, 'truncated': None, 'crowd': None,
                    'object_count': 0
                })
            else:
                obj_count = sum(1 for lab in det_labels
                                if lab.get('box2d'))
                for label in det_labels:
                    bbox = label.get('box2d') or {}
                    if bbox:  # Only process if bounding box exists
                        x1, y1 = bbox.get('x1'), bbox.get('y1')
                        x2, y2 = bbox.get('x2'), bbox.get('y2')
                        width = (x2 - x1) if (x1 is not None
                                              and x2 is not None) \
                            else None
                        height = (y2 - y1) if (y1 is not None
                                               and y2 is not None) \
                            else None
                        area = (width * height) if (width is not None
                                                    and height
                                                    is not None)\
                            else None
                    else:
                        x1 = y1 = x2 = y2 = width = height = area = None

                    obj_attrs = label.get('attributes', {}) or {}
                    parsed_data.append({
                        'image_name': image_name,
                        'weather': weather,
                        'scene': scene,
                        'timeofday': timeofday,
                        'category': label.get('category'),
                        'bbox_x1': x1, 'bbox_y1': y1,
                        'bbox_x2': x2, 'bbox_y2': y2,
                        'bbox_width': width,
                        'bbox_height': height,
                        'bbox_area': area,
                        'occluded': obj_attrs.get('occluded'),
                        'truncated': obj_attrs.get('truncated'),
                        'crowd': obj_attrs.get('crowd'),
                        'object_count': obj_count
                    })

        return pd.DataFrame(parsed_data)

    # Existing analyses
    def analyze_class_distribution(self, train_df, val_df):
        """
        Analyze the distribution of object detection classes.

        Returns:
            Dict[str, Any]: Class distribution statistics
        """
        self.logger.info("Analyzing class distribution...")

        # Class distribution analysis
        train_class_counts = train_df['category'].value_counts()
        val_class_counts = val_df['category'].value_counts()

        # Images per class (unique images containing each class)
        train_images_per_class = train_df.groupby(
            'category')['image_name'].nunique()
        val_images_per_class = val_df.groupby(
            'category')['image_name'].nunique()

        distribution_stats = {
            'train_class_counts': train_class_counts.to_dict(),
            'val_class_counts': val_class_counts.to_dict(),
            'train_images_per_class': train_images_per_class.to_dict(),
            'val_images_per_class': val_images_per_class.to_dict(),
            'train_total_objects': len(train_df[train_df['category'].notna()]),
            'val_total_objects': len(val_df[val_df['category'].notna()]),
            'train_total_images': train_df['image_name'].nunique(),
            'val_total_images': val_df['image_name'].nunique()
        }

        return distribution_stats

    def analyze_bbox_statistics(self, train_df, val_df):
        """
        Analyze bounding box statistics across classes.

        Returns:
            Dict[str, Any]: Bounding box statistics
        """
        self.logger.info("Analyzing bounding box statistics...")

        # Filter out rows with no bounding boxes
        train_bbox_df = train_df[train_df['category'].notna()].copy()
        val_bbox_df = val_df[val_df['category'].notna()].copy()

        bbox_stats: Dict[str, Dict[str, Any]] = {}

        for split_name, df in [('train', train_bbox_df), ('val', val_bbox_df)]:
            bbox_stats[split_name] = {}

            for category in self.detection_classes:
                cat_data = df[df['category'] == category]
                if len(cat_data) > 0:
                    ar = (cat_data['bbox_width'] /
                          cat_data['bbox_height']).replace(
                        [np.inf, -np.inf], np.nan)
                    bbox_stats[split_name][category] = {
                        'count': int(len(cat_data)),
                        'area_mean': float(cat_data['bbox_area'].mean()),
                        'area_std': float(cat_data['bbox_area'].std()),
                        'area_median': float(cat_data['bbox_area'].median()),
                        'width_mean': float(cat_data['bbox_width'].mean()),
                        'height_mean': float(cat_data['bbox_height'].mean()),
                        'aspect_ratio_mean': float(ar.mean()),
                        'area_min': float(cat_data['bbox_area'].min()),
                        'area_max': float(cat_data['bbox_area'].max())
                    }

        return bbox_stats

    def analyze_environmental_conditions(self, train_df, val_df):
        """
        Analyze object distribution under different environmental conditions.

        Returns:
            Dict[str, Any]: Environmental condition analysis
        """
        self.logger.info("Analyzing environmental conditions...")

        env_stats: Dict[str, Any] = {}

        for split_name, df in [('train', train_df), ('val', val_df)]:
            # Weather distribution
            weather_dist = df.groupby(
                ['weather', 'category']).size().unstack(fill_value=0)

            # Scene distribution
            scene_dist = df.groupby(
                ['scene', 'category']).size().unstack(fill_value=0)

            # Time of day distribution
            time_dist = df.groupby(
                ['timeofday', 'category']).size().unstack(fill_value=0)

            env_stats[split_name] = {
                'weather_distribution': weather_dist.to_dict(),
                'scene_distribution': scene_dist.to_dict(),
                'timeofday_distribution': time_dist.to_dict()
            }

        return env_stats

    def identify_anomalies(self, train_df, val_df):
        """
        Identify anomalies and interesting patterns in the dataset.

        Returns:
            Dict[str, Any]: Anomaly detection results
        """
        self.logger.info("Identifying anomalies and patterns...")

        anomalies: Dict[str, Any] = {}

        for split_name, df in [('train', train_df), ('val', val_df)]:
            bbox_df = df[df['category'].notna()].copy()

            anomalies[split_name] = {}

            for category in self.detection_classes:
                cat_data = bbox_df[bbox_df['category'] == category]
                if len(cat_data) > 0:
                    # Detect outliers using IQR method
                    q1 = cat_data['bbox_area'].quantile(0.25)
                    q3 = cat_data['bbox_area'].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = cat_data[
                        (cat_data['bbox_area'] < lower_bound) |
                        (cat_data['bbox_area'] > upper_bound)
                    ]

                    # Extremely small/large objects
                    very_small = cat_data[cat_data
                                          ['bbox_area'] < 100]
                    # < 10x10 pixels
                    very_large = cat_data[cat_data
                                          ['bbox_area'] > 50000]
                    # > ~224x224 pixels

                    # Unusual aspect ratios
                    aspect_ratios = cat_data['bbox_width']\
                                    / cat_data['bbox_height']
                    unusual_aspect = cat_data[
                        (aspect_ratios < 0.1) |
                        (aspect_ratios > 10)
                    ]

                    anomalies[split_name][category] = {
                        'outlier_count': int(len(outliers)),
                        'outlier_percentage': float(
                            len(outliers) / len(cat_data) * 100),
                        'very_small_objects': int(len(very_small)),
                        'very_large_objects': int(len(very_large)),
                        'unusual_aspect_ratio': int(len(
                            unusual_aspect)),
                        'outlier_samples': outliers['image_name']
                                               .tolist()[:5]
                        # Top 5 examples
                    }

        return anomalies

    # Data quality & attributes
    def validate_annotations(self) -> pd.DataFrame:
        """Return rows with invalid/missing geometry or out-of-bounds boxes."""
        df = self.parse_annotations(self.train_data + self.val_data)
        problems: List[Dict[str, Any]] = []
        for _, r in df[df['category'].notna()].iterrows():
            bad_geom = (
                r.bbox_x1 is None or r.bbox_y1 is None or
                r.bbox_x2 is None or r.bbox_y2 is None or
                r.bbox_width is None or r.bbox_height is None or
                r.bbox_width <= 0 or r.bbox_height <= 0
            )
            if bad_geom:
                problems.append(r.to_dict())
                continue
            # Simple bounds check against
            # default size (use real dims if available)
            width, height = self.DEFAULT_W, self.DEFAULT_H
            out_of_bounds = (r.bbox_x1 < 0) or (r.bbox_y1 < 0)\
                            or (r.bbox_x2 > width) or\
                            (r.bbox_y2 > height)
            if out_of_bounds:
                problems.append(r.to_dict())
        issues = pd.DataFrame(problems)
        out_path = self.output_dir / "data_quality_issues.csv"
        issues.to_csv(out_path, index=False)
        self.logger.info(f"Data quality issues saved to"
                         f" {out_path} (n={len(issues)})")
        return issues

    def analyze_object_attributes(self) -> Dict[str, Any]:
        """Quantify occluded/truncated/crowd flags by class and split."""
        result: Dict[str, Any] = {}
        for split_name, data in [('train', self.train_data),
                                 ('val', self.val_data)]:
            df = self.parse_annotations(data)
            df = df[df['category'].notna()].copy()
            for attr in ['occluded', 'truncated', 'crowd']:
                ct = df.groupby(['category', attr]
                                ).size().unstack(fill_value=0)
                result.setdefault(split_name, {})[attr] = ct.to_dict()
        with open(self.output_dir / "object_attributes.json",
                  "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # Image quality & paths
    def _resolve_split_dir(self, split: str) -> Path:
        assert self.images_dir is not None, "images_dir is not set"
        return self.images_dir / split

    def analyze_image_quality(self, split: str = "train",
                              sample: int = 1000) -> pd.DataFrame:
        """Compute simple image quality metrics (requires
        cv2 & images)."""
        if cv2 is None or self.images_dir is None:
            self.logger.warning("Image quality skipped "
                                "(cv2 or images directory missing).")
            return pd.DataFrame()

        data = self.train_data if split == "train"\
            else self.val_data
        names = [d.get("name") for d in data][:sample]
        split_dir = self._resolve_split_dir(split)

        rows: List[Dict[str, Any]] = []
        for name in names:
            img_path = split_dir / name
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray,
                                       cv2.CV_64F).var())
            brightness = float(gray.mean())
            contrast = float(gray.std())
            dyn_range = float(gray.max() - gray.min())
            rows.append({
                "image_name": name,
                "blur_var": blur,
                "brightness": brightness,
                "contrast": contrast,
                "dynamic_range": dyn_range,
            })
        df = pd.DataFrame(rows)
        out = self.output_dir / f"image_quality_{split}.csv"
        df.to_csv(out, index=False)
        self.logger.info(f"Image quality ({split}) "
                         f"saved to {out} (n={len(df)})")
        return df

    # New: Coverage & negatives
    def scenario_coverage_report(self, min_count: int = 10) -> pd.DataFrame:
        """Create a (class × weather × time ×
        scene) coverage table (counts)."""
        df = self.parse_annotations(self.train_data)
        df = df[df['category'].notna()].copy()
        pivot = df.pivot_table(
            index=['category'],
            columns=['weather', 'timeofday', 'scene'],
            values='image_name',
            aggfunc='count',
            fill_value=0
        )
        (self.output_dir / "scenario_coverage_counts.csv"
         ).write_text(pivot.to_csv())
        low = (pivot < min_count)
        (self.output_dir / "scenario_coverage_lowcells.csv"
         ).write_text(low.to_csv())
        self.logger.info("Scenario coverage reports "
                         "saved (counts & low-cells)")
        return pivot

    def analyze_negative_frames(self) -> pd.DataFrame:
        """List images with zero detection
         classes by split."""
        frames: List[Dict[str, Any]] = []
        for split, data in [('train', self.train_data),
                            ('val', self.val_data)]:
            df = self.parse_annotations(data)
            per_img = df.groupby('image_name')['category'
            ].apply(lambda s: s.dropna().shape[0])
            negatives = per_img[per_img == 0].index.tolist()
            for n in negatives:
                frames.append({'split': split, 'image_name': n})
        out = pd.DataFrame(frames)
        out.to_csv(self.output_dir / "negative_frames.csv",
                   index=False)
        self.logger.info(f"Negative frames saved: {len(out)}")
        return out

    # Visualizations (existing + spatial density)
    def create_visualizations(self, train_df, val_df) -> None:
        """Create comprehensive visualizations of
        the dataset analysis."""
        self.logger.info("Creating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Class Distribution Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        train_counts = train_df['category'].value_counts()
        val_counts = val_df['category'].value_counts()

        train_counts.plot(kind='bar', ax=ax1,
                          title='Training Set - Class Distribution')
        ax1.set_ylabel('Number of Objects')
        ax1.tick_params(axis='x', rotation=45)

        val_counts.plot(kind='bar', ax=ax2,
                        title='Validation Set - Class Distribution')
        ax2.set_ylabel('Number of Objects')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Bounding Box Area Distribution
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()

        bbox_train = train_df[train_df['category'].notna()]

        for i, category in enumerate(self.detection_classes):
            cat_data = bbox_train[bbox_train['category'] == category]
            if len(cat_data) > 0:
                axes[i].hist(cat_data['bbox_area'],
                             bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{category}\n(n={len(cat_data)})')
                axes[i].set_xlabel('Bounding Box Area (pixels²)')
                axes[i].set_ylabel('Frequency')
                axes[i].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_area_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Environmental Conditions Heatmap
        pivot_weather = train_df.groupby(['weather', 'category']
                                         ).size().unstack(fill_value=0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_weather, annot=True, fmt='d',
                    cmap='YlOrRd')
        plt.title('Object Distribution by Weather '
                  'Conditions (Training Set)')
        plt.ylabel('Weather')
        plt.xlabel('Object Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir /
                    'weather_distribution_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Objects per Image Distribution
        objects_per_image_train = train_df.groupby(
            'image_name')['category'].count()
        objects_per_image_val = val_df.groupby(
            'image_name')['category'].count()

        plt.figure(figsize=(12, 6))
        plt.hist(objects_per_image_train, bins=50,
                 alpha=0.7, label='Training', density=True)
        plt.hist(objects_per_image_val, bins=50,
                 alpha=0.7, label='Validation', density=True)
        plt.xlabel('Number of Objects per Image')
        plt.ylabel('Density')
        plt.title('Distribution of Objects per Image')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir /
                    'objects_per_image_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Static visualizations saved")

    @staticmethod
    def _annotate_bars_with_pct(ax, total: int,
                                fmt: str = "{:,} ({:.1f}%)",
                                dy: float = 0.01):
        """
        Put 'count (pct%)' labels above each
        bar in a bar chart.
        Args:
            ax: matplotlib Axes with bars already drawn
            total: denominator for percentage
            fmt: label format
            dy: vertical offset as fraction of y-range
        """
        total = max(int(total), 1)
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin if ymax > ymin else 1.0

        for p in ax.patches:
            if p.get_height() is None:
                continue
            h = float(p.get_height())
            x = p.get_x() + p.get_width() / 2.0
            ax.text(
                x, h + dy * yr,
                fmt.format(int(round(h)),
                           100.0 * h / total),
                ha="center", va="bottom",
                fontsize=9, rotation=0
            )

    def visualize_statistics(self, train_df,
                             val_df) -> None:
        """Create figures for each dashboard
        panel with counts & percents on bars."""
        self.logger.info("Creating dashboard "
                         "images with counts & percentages...")

        bbox_train = train_df[train_df[
            "category"].notna()].copy()
        bbox_val = val_df[val_df[
            "category"].notna()].copy()

        # =========================
        # 1) Class Distribution Comparison
        # (bars with labels)
        # =========================
        train_counts = train_df["category"].value_counts()
        val_counts = val_df["category"].value_counts()

        fig, axes = plt.subplots(1, 2,
                                 figsize=(16, 6),
                                 constrained_layout=True)

        # Train
        train_counts.plot(kind="bar", ax=axes[0],
                          color="#4C78A8", edgecolor="black")
        axes[0].set_title("Training Set - Class Distribution")
        axes[0].set_ylabel("Number of Objects")
        axes[0].tick_params(axis="x", rotation=45)
        self._annotate_bars_with_pct(axes[0],
                                     total=train_counts.sum(),
                                     dy=0.02)

        # Val
        val_counts.plot(kind="bar", ax=axes[1],
                        color="#F58518", edgecolor="black")
        axes[1].set_title("Validation Set - Class Distribution")
        axes[1].set_ylabel("Number of Objects")
        axes[1].tick_params(axis="x", rotation=45)
        self._annotate_bars_with_pct(axes[1],
                                     total=val_counts.sum(),
                                     dy=0.02)

        out_path = self.output_dir / \
                   "class_distribution_comparison.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        # =========================
        # 2) Bounding Box Area by Class (Top-5)
        # — keep as boxplot (no labels)
        # =========================
        top5 = bbox_train["category"].value_counts()\
            .head(5).index.tolist()
        top5_df = bbox_train[bbox_train["category"].isin(top5)].copy()
        fig, ax = plt.subplots(figsize=(10, 6),
                               constrained_layout=True)
        sns.boxplot(data=top5_df, x="category",
                    y="bbox_area", ax=ax, showfliers=False)
        ax.set_yscale("log")
        ax.set_title("Bounding Box Area by Class "
                     "(Top 5, Training)")
        ax.set_xlabel("Class")
        ax.set_ylabel("Box Area (pixels², log)")
        ax.tick_params(axis="x", rotation=30)
        (self.output_dir / "bbox_area_by_class_top5_boxplot.png")\
            .write_bytes(
            fig.canvas.tostring_png()
            if hasattr(fig.canvas, "tostring_png") else b"")
        fig.savefig(self.output_dir /
                    "bbox_area_by_class_top5_boxplot.png", dpi=300)
        plt.close(fig)

        # =========================
        # 3) Objects per Environmental Condition
        # (Weather) — bars with labels
        # =========================
        weather_counts = train_df.groupby(
            "weather")["category"].count().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6),
                               constrained_layout=True)
        weather_counts.plot(kind="bar", ax=ax,
                            color="#54A24B", edgecolor="black")
        ax.set_title("Object Counts by Weather (Training)")
        ax.set_xlabel("Weather")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        self._annotate_bars_with_pct(ax, total=weather_counts.sum(), dy=0.02)

        fig.savefig(self.output_dir /
                    "weather_distribution_training.png", dpi=300)
        plt.close(fig)

        # =========================
        # 4) Aspect Ratio Distribution — add secondary percentage axis
        # =========================
        aspect = (bbox_train["bbox_width"] /
                  bbox_train["bbox_height"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        total_aspect = max(len(aspect), 1)

        fig, ax = plt.subplots(figsize=(10, 6),
                               constrained_layout=True)
        counts, bins, patches = ax.hist(aspect, bins=50,
                                        edgecolor="black", alpha=0.85)
        ax.set_title("Aspect Ratio Distribution (Training)")
        ax.set_xlabel("Width / Height")
        ax.set_ylabel("Count")

        # Right y-axis as percentage of total
        def count2pct(y):
            return 100.0 * y / total_aspect

        def pct2count(p):
            return total_aspect * p / 100.0
        secax = ax.secondary_yaxis("right", functions=(count2pct, pct2count))
        secax.set_ylabel("Percentage (%)")

        fig.savefig(self.output_dir / "aspect_ratio_distribution_training.png",
                    dpi=300)
        plt.close(fig)

        # =========================
        # 5) Train vs Val Split Analysis — annotate counts only
        # =========================
        split_comparison = pd.DataFrame({
            "Training": [len(train_df), train_df["image_name"].nunique()],
            "Validation": [len(val_df), val_df["image_name"].nunique()]
        }, index=["Total Annotations", "Unique Images"])

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        split_comparison.T.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Train vs Val Split Analysis")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Metric")

        # Annotate each bar with its count
        for p in ax.patches:
            h = p.get_height()
            if h is None:
                continue
            x = p.get_x() + p.get_width() / 2.0
            ax.text(x, h + 0.01 * ax.get_ylim()[1],
                    f"{int(round(h)):,}", ha="center", va="bottom", fontsize=9)

        fig.savefig(self.output_dir / "train_val_split_analysis.png", dpi=300)
        plt.close(fig)

        # =========================
        # 6) Dataset Summary table — unchanged
        # =========================
        summary_rows = [
            ["Total Training Images", train_df["image_name"].nunique()],
            ["Total Validation Images", val_df["image_name"].nunique()],
            ["Total Object Classes", len(self.detection_classes)],
            ["Training Objects", int(bbox_train.shape[0])],
            ["Validation Objects", int(bbox_val.shape[0])],
        ]
        fig, ax = plt.subplots(figsize=(8, 2 + 0.3 * len(summary_rows)))
        ax.axis("off")
        tbl = ax.table(
            cellText=[[str(k), str(v)] for k, v in summary_rows],
            colLabels=["Metric", "Value"],
            cellLoc="left", colLoc="left", loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
        fig.savefig(self.output_dir / "dataset_summary_table.png",
                     dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info("Static dashboard images "\
                         "(with counts/percentages) "\
                            "saved to %s", self.output_dir)

    def find_interesting_samples(self, train_df,
                                 num_samples: int = 5) -> Dict[str, List[str]]:
        """
        Identify interesting/unique samples in different classes.

        Args:
            train_df (List): data
            num_samples (int): Number of samples to find per category

        Returns:
            Dict[str, List[str]]: Interesting samples by category
        """
        self.logger.info("Finding interesting samples...")

        interesting_samples: Dict[str, List[str]] = {}

        for category in self.detection_classes:
            cat_data = train_df[train_df['category'] == category]
            if len(cat_data) > 0:
                samples: List[str] = []

                # Largest objects
                largest = cat_data.nlargest(
                    2, 'bbox_area')['image_name'].tolist()

                # Smallest objects
                smallest = cat_data.nsmallest(
                    2, 'bbox_area')['image_name'].tolist()

                # Most crowded images (images with
                #  many objects of this class)
                crowded = cat_data.groupby('image_name'
                                           ).size().nlargest(1).index.tolist()

                samples.extend(largest + smallest + crowded)
                interesting_samples[category] =\
                      list(dict.fromkeys(samples))[:num_samples]

        return interesting_samples

    # Report (existing + extended)
    def complete_analysis(self) -> None:
        """Generate a comprehensive analysis report."""
        self.logger.info("Generating comprehensive report...")

        # Perform all analyses
        train_df = self.parse_annotations(self.train_data)
        val_df = self.parse_annotations(self.val_data)

        class_dist = self.analyze_class_distribution(train_df, val_df)
        bbox_stats = self.analyze_bbox_statistics(train_df, val_df)
        env_stats = self.analyze_environmental_conditions(train_df, val_df)
        anomalies = self.identify_anomalies(train_df, val_df)
        interesting = self.find_interesting_samples(train_df)

        issues = self.validate_annotations()
        attrs = self.analyze_object_attributes()
        coverage = self.scenario_coverage_report()
        negatives = self.analyze_negative_frames()

        # Save detailed statistics
        with open(self.output_dir / 'detailed_statistics.json',
                  'w', encoding="utf-8") as f:
            json.dump({
                'class_distribution': class_dist,
                'bbox_statistics': bbox_stats,
                'environmental_stats': env_stats,
                'anomalies': anomalies,
                'interesting_samples': interesting,
                'data_quality_issues_file': str(self.output_dir
                                                / 'data_quality_issues.csv'),
                'object_attributes_file': str(self.output_dir
                                              / 'object_attributes.json'),
                'scenario_coverage_counts_file':
                    str(self.output_dir / 'scenario_coverage_counts.csv'),
                'scenario_coverage_lowcells_file':
                    str(self.output_dir / 'scenario_coverage_lowcells.csv'),
                'negative_frames_file': str(self.output_dir
                                            / 'negative_frames.csv'),
            }, f, indent=2, default=str)

        # Generate markdown report
        report_content = self._generate_markdown_report(
            class_dist, bbox_stats, env_stats, anomalies, interesting,
            issues, attrs
        )

        with open(self.output_dir / 'BDD_Dataset_Analysis_Report.md',
                  'w', encoding="utf-8") as f:
            f.write(report_content)

        self.logger.info("Report generation completed")

    @staticmethod
    def _generate_markdown_report(class_dist: Dict, bbox_stats: Dict,
                                  env_stats: Dict, anomalies: Dict,
                                  interesting: Dict,
                                  issues: pd.DataFrame,
                                  attrs: Dict[str, Any],
                                  ) -> str:
        """Generate a markdown report with all analysis results."""
        report = f"""# BDD Dataset Analysis Report — Enhanceds

## Executive Summary

This report provides a comprehensive analysis of the Berkeley
 DeepDrive (BDD) dataset for object detection tasks, extended
 with data quality checks, drift analysis, spatial/context
 insights, and model-ready priors.

### Dataset Overview
- **Training Images**: {class_dist['train_total_images']:,}
- **Validation Images**: {class_dist['val_total_images']:,}
- **Total Training Objects**: {class_dist['train_total_objects']:,}
- **Total Validation Objects**: {class_dist['val_total_objects']:,}

## Class Distribution Analysis

### Training Set Distribution
"""
        for class_name, count in class_dist['train_class_counts'].items():
            if class_name is not None:
                percentage = (count / max(
                    class_dist['train_total_objects'], 1)) * 100
                report += f"- **{class_name}**: " \
                          f"{count:,} objects ({percentage:.2f}%)\n"

        report += "\n### Validation Set Distribution\n"
        for class_name, count in \
                class_dist['val_class_counts'].items():
            if class_name is not None:
                percentage = (count / max(
                    class_dist['val_total_objects'], 1)) * 100
                report += f"- **{class_name}**: "\
                    f"{count:,} objects ({percentage:.2f}%)\n"

        report += "\n## Key Findings\n\n"

        # Add key insights
        train_counts = class_dist['train_class_counts']
        most_common = max(train_counts.items(),
                           key=lambda x: x[1]
                           if x[0] else 0)
        least_common = min(train_counts.items(),
                           key=lambda x: x[1]
                           if x[0] else float('inf'))

        report += f"""### Class Imbalance
- Most frequent class: **{most_common[0]}**
 with {most_common[1]:,} instances
- Least frequent class: **{least_common[0]}**
with {least_common[1]:,} instances
- Imbalance ratio: {most_common[1] /
                    max(least_common[1], 1):.1f}:1

### Anomalies Detected
"""
        for split in ['train', 'val']:
            report += f"\n#### {split.title()} Set Anomalies\n"
            for category, anom_data in anomalies[split].items():
                report += f"- **{category}**: {anom_data['outlier_count']}" \
                          f" outliers "\
                            f"({anom_data['outlier_percentage']:.2f}%)\n"
                report += f"  - Very small objects:"\
                    f"{anom_data['very_small_objects']}\n"
                report += "  - Very large objects:"\
                    f" {anom_data['very_large_objects']}\n"

        report += "\n## Data Quality & Attributes\n"
        report += "- Invalid/Out-of-bounds/Empty "\
            f"boxes detected: **{len(issues)}** "\
                "(see `data_quality_issues.csv`).\n"
        report += "- Occluded/Truncated/Crowd distributions "\
            "saved to `object_attributes.json`.\n"

        report += "\n## Scenario Coverage & Hard Negatives\n"
        report += "- Scenario coverage table: `scenario_coverage_counts.csv" \
                  "` (low-count cells in `scenario_coverage_lowcells.csv`).\n"
        report += "- Images with zero detection "\
            "classes logged to `negative_frames.csv`.\n"

        report += "\n## Recommendations\n\n"
        report += """1. **Addressing Class Imbalance**: Consider data
        augmentation or weighted sampling for underrepresented classes
2. **Quality Control**: Review outlier samples and flagged
quality issues for potential label errors
3. **Environmental Bias**: Ensure model training accounts
for weather/time-of-day/scene distributions
4. **Scale Handling**: Implement multi-scale training
and small-object strategies given S/M/L mix
6. **Crowding**: Prefer Soft-NMS / cluster-NMS when same-class IoU is high
"""

        return report

    # Pipeline
    def run_complete_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        self.logger.info("Starting complete BDD dataset analysis...")

        try:
            self.load_dataset()

            # Perform all analyses
            train_df = self.parse_annotations(self.train_data)
            val_df = self.parse_annotations(self.val_data)

            self.create_visualizations(train_df, val_df)
            self.visualize_statistics(train_df, val_df)
            self.complete_analysis()

            # image quality
            try:
                self.analyze_image_quality("train", sample=800)
                self.analyze_image_quality("val", sample=400)
            except Exception as e:
                self.logger.warning(f"Image quality skipped: {e}")

            self.logger.info("Analysis completed successfully!")
            self.logger.info(f"Results saved to: {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the BDD100K dataset (object detection) and generate reports/visualizations."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the BDD dataset root directory containing labels/ (and optionally images/).",
    )
    parser.add_argument(
        "--output-dir",
        default="bdd_analysis_results",
        help="Directory to write analysis outputs (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    analyzer = BDDDatasetAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        images_dir="",
    )

    # Run the full pipeline step-by-step (mirrors run_complete_analysis,
    # but lets us control image-quality sampling via CLI).
    analyzer.logger.info("Starting complete BDD dataset analysis...")
    try:
        analyzer.load_dataset()

        train_df = analyzer.parse_annotations(analyzer.train_data)
        val_df = analyzer.parse_annotations(analyzer.val_data)

        analyzer.create_visualizations(train_df, val_df)
        analyzer.visualize_statistics(train_df, val_df)
        analyzer.complete_analysis()

        analyzer.logger.info("Analysis completed successfully!")
        analyzer.logger.info(f"Results saved to: {analyzer.output_dir}")

        print("BDD Dataset Analysis Complete!")
        print(f"Check the '{analyzer.output_dir}' directory for results:")
        print("- BDD_Dataset_Analysis_Report.md")
        print("- Various visualization PNG files")
        print("- detailed_statistics.json and other CSV/JSON artifacts")

    except Exception as e:
        analyzer.logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()