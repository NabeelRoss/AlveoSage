"""
Final Test-Set Evaluation Entry Point for AlveoSage.

Evaluates the trained DenseNet121 checkpoint (outputs/checkpoints/best_model.pth)
exclusively on the untouched benchmark test split (processed_dataset/manifests/test.csv).

Saves:
- outputs/evaluation/test_metrics.json
- outputs/evaluation/test_confusion_matrix.png
- outputs/evaluation/test_roc_curve.png
- outputs/evaluation/test_precision_recall.png
- outputs/evaluation/test_calibration_diagram.png
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import evaluate_test_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate_test")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlveoSage on the isolated test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to best model checkpoint (defaults to outputs/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="processed_dataset",
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Path to save evaluation artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    metrics = evaluate_test_set(
        checkpoint_path=Path(args.checkpoint),
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )
    print("\n" + "=" * 60)
    print("ALVEOSAGE FINAL TEST-SET BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Test Samples:  {metrics.get('sample_count', 624)}")
    print(f"Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f"Sensitivity (Recall):{metrics['sensitivity'] * 100:.2f}%")
    print(f"Specificity:         {metrics['specificity'] * 100:.2f}%")
    print(f"Precision:           {metrics['precision'] * 100:.2f}%")
    print(f"F1-Score:            {metrics['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"Brier Score:         {metrics['brier_score']:.4f}")
    print(f"ECE (Calibration):   {metrics['expected_calibration_error']:.4f}")
    print("\nClass Breakdown:")
    for cls_name, cls_m in metrics.get("class_breakdown", {}).items():
        print(f"  {cls_name:15s} | Support: {cls_m['support']:3d} | Prec: {cls_m['precision']:.4f} | Rec: {cls_m['recall']:.4f} | F1: {cls_m['f1_score']:.4f}")
    print(f"\nArtifacts saved under: {args.output_dir}/")


if __name__ == "__main__":
    main()
