"""
Final Test-Set Threshold Comparison Entry Point for AlveoSage.

Compares benchmark test performance between:
- 0.50 (original baseline)
- 0.80 (or validation-selected threshold loaded from outputs/evaluation/validation_threshold_analysis.json)

Usage:
    python compare_test_thresholds.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.test_threshold_comparison import run_test_threshold_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("compare_test_thresholds")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AlveoSage test-set performance at baseline (0.50) vs validation-selected threshold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to best model checkpoint",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="processed_dataset",
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--val-analysis",
        type=str,
        default="outputs/evaluation/validation_threshold_analysis.json",
        help="Path to validation threshold analysis JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save comparison metrics and plots",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    args = parser.parse_args()

    results = run_test_threshold_comparison(
        checkpoint_path=Path(args.checkpoint),
        dataset_root=Path(args.dataset_root),
        val_analysis_path=Path(args.val_analysis),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )

    t_val = results["validation_selected_threshold_metadata"]["threshold_value"]
    m_base = results["baseline_threshold_0_50"]
    m_sel = results["validation_selected_threshold"]
    delta = results["comparison_summary"]

    print("\n" + "=" * 72)
    print("ALVEOSAGE TEST-SET THRESHOLD COMPARISON RESULTS")
    print("=" * 72)
    print(f"Total Test Samples:   {results['sample_count']} (NORMAL: {results['class_distribution']['NORMAL']}, PNEUMONIA: {results['class_distribution']['PNEUMONIA']})")
    print(f"ROC-AUC (Global):     {results['roc_auc']:.4f}")
    print(f"Val-Selected Thresh:  {t_val:.2f} ({results['validation_selected_threshold_metadata']['selection_criterion']})")
    print("-" * 72)
    print(f"{'Metric':<22} | {'Baseline (0.50)':<16} | {f'Val-Selected ({t_val:.2f})':<18} | {'Delta':<10}")
    print("-" * 72)
    print(f"{'Accuracy':<22} | {m_base['accuracy']*100:>14.2f}% | {m_sel['accuracy']*100:>16.2f}% | {delta['accuracy_delta']*100:>+8.2f}%")
    print(f"{'Sensitivity (Recall)':<22} | {m_base['sensitivity']*100:>14.2f}% | {m_sel['sensitivity']*100:>16.2f}% | {delta['sensitivity_delta']*100:>+8.2f}%")
    print(f"{'Specificity':<22} | {m_base['specificity']*100:>14.2f}% | {m_sel['specificity']*100:>16.2f}% | {delta['specificity_delta']*100:>+8.2f}%")
    print(f"{'Precision':<22} | {m_base['precision']*100:>14.2f}% | {m_sel['precision']*100:>16.2f}% | {delta['precision_delta']*100:>+8.2f}%")
    print(f"{'F1-Score':<22} | {m_base['f1_score']*100:>14.2f}% | {m_sel['f1_score']*100:>16.2f}% | {delta['f1_score_delta']*100:>+8.2f}%")
    print("-" * 72)
    print(f"False Positives:      {m_base['confusion_matrix']['fp']} -> {m_sel['confusion_matrix']['fp']} (FP Reduction: {delta['false_positive_reduction']})")
    print(f"False Negatives:      {m_base['confusion_matrix']['fn']} -> {m_sel['confusion_matrix']['fn']} (FN Change: {delta['false_negative_change']:+d})")
    print("=" * 72)


if __name__ == "__main__":
    main()
