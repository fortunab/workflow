from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES = [
    "segmentation_metrics.py",
    "inference_efficiency.py",
    "vqa_evaluation_small.py",
    "vqa_evaluation_large.py",
    "rougel_comparison.py",
    "bleu_comparison.py",
    "pipeline_metrics.py",
    "latency_efficiency.py",
    "token_ablation.py",
    "perception_analysis.py",
    "transfer_analysis.py",
    "cross_dataset_transfer.py",
    "pipeline_stage_latency.py",
    "domain_token_ablation.py",
    "uncertainty_behavior.py",
    "segmentation_ablation.py",
]