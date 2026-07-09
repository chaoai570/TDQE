"""
TDQE: 文本数据质量评估 —— 论文完整复现。

论文: "TDQE: 一种面向深度学习的文本数据质量评估方法"
      罗春旭, 熊海旭, 叶雅珍, 丁滟, 宗世泽, 熊贇, 朱扬勇
      复旦大学 & 国防科技大学

包含以下模块:
    - 语义一致性检测 (Section 2.1)  →  robustness.py
    - 匹配度检测       (Section 2.2)  →  accuracy.py
    - 质量分数聚合     (Section 2.3)  →  quality.py
    - 20NG 数据集加载  (Section 3.1)  →  data.py
    - 分类器验证实验   (Section 3)    →  experiment.py
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
