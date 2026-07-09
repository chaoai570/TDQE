# TDQE: 文本数据质量评估方法

论文 *TDQE: 一种面向深度学习的文本数据质量评估方法* 的 PyTorch 复现。

罗春旭, 熊海旭, 叶雅珍, 丁滟, 宗世泽, 熊贇, 朱扬勇 — 复旦大学 & 国防科技大学, 发表于《大数据》2025.

---

## 方法概述

TDQE 通过两个维度评估文本数据被语言模型理解的程度。

### 语义一致性检测 — 鲁棒性（论文 Section 2.1）

利用文本摘要生成模型的 Dropout 生成 $m = 3$ 个随机子网络，提取多个嵌入表示，通过嵌入向量间的平均欧氏距离衡量鲁棒性。鲁棒性指标 $R(V(T))$ 定义为（论文公式 1）：

$$ R(V(T)) = \sigma \left( -\frac{2}{m(m-1)} \sum_{i \lt j} d(v\_i, v\_j) \right) $$

其中 $d(\cdot,\cdot)$ 为欧氏距离， $\sigma(\cdot)$ 为 Sigmoid 函数，确保 $R \in [0, 0.5]$。

### 匹配度检测 — 准确性（论文 Section 2.2）

利用文本相似匹配模型计算数据样本 $T$ 与其模型生成的摘要 $S$ 之间的匹配度 $M(T,S)$，以此评估模型理解的准确程度。 $M(T,S) \in \[0, 0.5\]$ 。

### 质量评估（论文 Section 2.3）

质量分数 $Q(T)$ 为鲁棒性与准确性之和（论文公式 2）：

$$Q(T) = R(V(T)) + M(T, S)$$

$Q(T) \in [0, 1.0]$，分数越高表示文本越适合用于语言模型训练。

---

## 实验验证（论文 Section 3）

- **数据集:** 20 Newsgroups（20 类，约 19,000 篇文章）
- **划分:** 8:2 训练/测试
- **分类器参数（Table 1）:** epochs=10, batch=8, lr=0.01, Leaky-ReLU, max_len=512
- **实验类型:** 删除低质量数据、删除高质量数据、消融实验

---

## 项目结构

```
├── tdqe/                     # 核心 Python 包
│   ├── __init__.py           # 公共接口
│   ├── config.py             # 论文参数常量（Table 1）
│   ├── robustness.py         # 语义一致性检测（Section 2.1）
│   ├── accuracy.py           # 匹配度检测（Section 2.2）
│   ├── quality.py            # 质量分数聚合（Section 2.3）
│   ├── data.py               # 20NG 数据集加载（Section 3.1）
│   └── experiment.py         # 分类器验证实验（Section 3.3–3.5）
├── TDQE_All_In_One.ipynb     # 完整流水线 Notebook
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

本项目使用 **20 Newsgroups** 数据集（20 个类别，约 19,000 篇新闻文本）。

**方法一：通过 scikit-learn 下载（推荐）**

在项目根目录运行以下 Python 脚本，自动下载并提取为 20 个 `.txt` 文件放入 `archive/` 目录：

```python
import os
from sklearn.datasets import fetch_20newsgroups

os.makedirs("archive", exist_ok=True)

for cat in fetch_20newsgroups(subset="all").target_names:
    data = fetch_20newsgroups(subset="all", categories=[cat], remove=())
    with open(f"archive/{cat}.txt", "w", encoding="utf-8") as f:
        for i, text in enumerate(data.data):
            f.write(f"Newsgroup: {cat}\n")
            f.write(f"document_id: {i}\n")
            f.write(text + "\n\n")
    print(f"  Done: {cat}")
```

**方法二：从 Kaggle 手动下载**

1. 访问 [20 Newsgroups on Kaggle](https://www.kaggle.com/datasets/crawford/20-newsgroups)
2. 下载 `20news-bydate.tar.gz` 并解压
3. 将解压后所有 `.txt` 文件放入 `archive/` 目录

最终 `archive/` 目录应包含 20 个 `.txt` 文件（如 `sci.space.txt`、`rec.autos.txt` 等）。

### 3. 运行

本项目默认使用 **distilgpt2** 作为基础模型。该模型满足 TDQE 框架的三个必要条件：(1) 含有 Dropout 层（embd/attn/resid dropout 均为 0.1），支持生成随机子网络以计算鲁棒性；(2) 输出隐藏状态用于提取嵌入向量；(3) 具备文本生成能力用于产生摘要计算准确性。论文指出 TDQE 框架是模型无关的（Section 2.3），不同结构的模型不会对质量评分排名产生明显影响。

模型会在首次运行时从 Hugging Face Hub 自动下载（约 330 MB），无需手动准备。

```bash
jupyter notebook TDQE_All_In_One.ipynb
```

按顺序执行所有 Cell 即可完成：

1. 自动下载并加载模型
2. 加载数据集
3. 计算 TDQE 质量分数
4. 训练分类器并运行数据删除/消融实验

---

## 与论文的对应关系

| 代码模块 | 论文章节 | 说明 |
|---|---|---|
| `tdqe/robustness.py` | Section 2.1 | 鲁棒性：$m=3$ 次 Dropout 前向传播，嵌入向量欧氏距离均值 |
| `tdqe/accuracy.py` | Section 2.2 | 准确性：文本与摘要嵌入的余弦相似度，缩放到 $[0, 0.5]$ |
| `tdqe/quality.py` | Section 2.3 | $Q(T) = R(V(T)) + M(T,S)$ |
| `tdqe/data.py` | Section 3.1 | 20NG 数据集加载，$8:2$ 训练/测试划分 |
| `tdqe/experiment.py` | Section 3.3–3.5 | 数据删除实验、消融实验 |
| `tdqe/config.py` | Table 1 | $m=3$, max_len=512, epochs=10, batch=8, lr=0.01 |

## 参考文献

1. Longpre S, et al. *A pretrainer's guide to training data.* NAACL, 2024.
2. Ferdowsi H, et al. *An online outlier identification and removal scheme.* IEEE TNNLS, 2014.
3. Frenay B, Verleysen M. *Classification in the presence of label noise.* IEEE TNNLS, 2014.
4. Schoch S, et al. *CS-SHAPLEY.* NeurIPS, 2022.
5. Ghorbani A, Zou J. *Data Shapley.* ICML, 2019.
6. Yoon J, et al. *DVRL.* ICML, 2020.
7. Srivastava N, et al. *Dropout.* JMLR, 2014.
8. Devlin J, et al. *BERT.* NAACL, 2019.
