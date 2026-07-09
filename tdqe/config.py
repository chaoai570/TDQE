"""
TDQE: Text Data Quality Evaluation — configuration and constants.

All values are taken directly from the paper:

    - Table 1 (parameter settings)
    - Section 2.1 (m = number of stochastic forward passes)
    - Section 3.1 (dataset split ratio, input max length)
"""

# ── Semantic Consistency Detection (Section 2.1) ──────────────
NUM_DROPOUT_PASSES = 3  # m in Formula (1); number of stochastic sub-networks

# ── Model / Tokenizer (Section 3.1, Table 1) ──────────────────
MAX_INPUT_LENGTH = 512      # max token length per sample
HIDDEN_SIZE = 768           # output feature dimension
BATCH_SIZE = 8              # batch size for classifier training
LEARNING_RATE = 0.01        # learning rate
NUM_EPOCHS = 10             # training epochs
ACTIVATION = "leaky_relu"   # Leaky-ReLU as per Table 1

# ── Dataset (Section 3.1) ─────────────────────────────────────
TRAIN_SPLIT = 0.8           # 8 : 2 train-test split
RANDOM_SEED = 42
NUM_CLASSES = 20            # 20 Newsgroups categories

# ── Summary Generation ─────────────────────────────────────────
SUMMARY_MAX_NEW_TOKENS = 80
SUMMARY_NUM_BEAMS = 2
