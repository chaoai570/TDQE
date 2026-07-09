"""
TDQE 配置常量。

所有参数均取自论文:
    - Table 1 (参数设置)
    - Section 2.1 (m = 随机前向传播次数)
    - Section 3.1 (数据集划分比例、输入最大长度)
"""

# ── 语义一致性检测 (Section 2.1) ────────────────────────
NUM_DROPOUT_PASSES = 3  # m = 公式(1)中随机子网络个数

# ── 模型 / 分词器 (Section 3.1, Table 1) ────────────────
MAX_INPUT_LENGTH = 512      # 每条样本最大分词数
HIDDEN_SIZE = 768           # 输出特征维度
BATCH_SIZE = 8              # 分类器训练的批大小
LEARNING_RATE = 0.01        # 学习率
NUM_EPOCHS = 10             # 训练轮次
ACTIVATION = "leaky_relu"   # Table 1 中的 Leaky-ReLU

# ── 数据集 (Section 3.1) ────────────────────────────────
TRAIN_SPLIT = 0.8           # 8 : 2 训练/测试划分
RANDOM_SEED = 42
NUM_CLASSES = 20            # 20 Newsgroups 类别数

# ── 摘要生成 ────────────────────────────────────────────
SUMMARY_MAX_NEW_TOKENS = 80
SUMMARY_NUM_BEAMS = 2
