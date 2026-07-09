"""
TDQE: Text Data Quality Evaluation
===================================

Paper: "TDQE: 一种面向深度学习的文本数据质量评估方法"
       Luo Chunxu, Xiong Haixu, Ye Yazhen, Ding Yan, Zong Shize, Xiong Yun, Zhu Yangyong
       Fudan University & National University of Defense Technology

Reference implementation of the TDQE framework consisting of:
    - Semantic Consistency Detection (Section 2.1)  →  robustness.py
    - Matching Degree Detection       (Section 2.2)  →  accuracy.py
    - Quality Score Aggregation       (Section 2.3)  →  quality.py
    - 20NG Dataset Loader             (Section 3.1)  →  data.py
    - Classifier Validation           (Section 3)    →  experiment.py
"""

from .config import (
    NUM_DROPOUT_PASSES,
    MAX_INPUT_LENGTH,
    HIDDEN_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ACTIVATION,
    TRAIN_SPLIT,
    RANDOM_SEED,
    NUM_CLASSES,
)

from .data import load_20ng, split_dataset, TDQEDataset
from .robustness import robustness_score
from .accuracy import accuracy_score
from .quality import quality_score, compute_all_scores
from .experiment import (
    train_classifier,
    evaluate_classifier,
    run_data_removal_experiment,
    run_ablation_experiment,
)

__all__ = [
    # config
    "NUM_DROPOUT_PASSES", "MAX_INPUT_LENGTH", "HIDDEN_SIZE",
    "BATCH_SIZE", "LEARNING_RATE", "NUM_EPOCHS", "ACTIVATION",
    "TRAIN_SPLIT", "RANDOM_SEED", "NUM_CLASSES",
    # data
    "load_20ng", "split_dataset", "TDQEDataset",
    # quality
    "robustness_score", "accuracy_score", "quality_score", "compute_all_scores",
    # experiment
    "train_classifier", "evaluate_classifier",
    "run_data_removal_experiment", "run_ablation_experiment",
]
