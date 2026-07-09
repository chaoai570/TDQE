# TDQE: 文本数据质量评估方法

论文 "[TDQE: 一种面向深度学习的文本数据质量评估方法](https://www.infocomm-journal.com/bdr/CN/10.11959/j.issn.2096-0271.2025073)" 的 PyTorch 复现。

**作者:** 罗春旭, 熊海旭, 叶雅珍, 丁滟, 宗世泽, 熊贇, 朱扬勇 (复旦大学 & 国防科技大学)

---

## 方法概述

TDQE 通过两个维度评估文本数据被语言模型理解的程度：

### 1. 语义一致性检测 — 鲁棒性 R(V(T)) [Section 2.1]

利用文本摘要生成模型的 Dropout 生成 m = 3 个随机子网络，提取多个嵌入表示 {v₁, v₂, ..., vₘ}，通过嵌入向量间的平均欧氏距离衡量鲁棒性：

$$\displaystyle R(V(T)) = \sigma\left(-\frac{2}{m(m-1)}\sum_{i<j} d(v_i, v_j)\right)$$

其中 d(·,·) 为欧氏距离，σ(·) 为 Sigmoid 函数，确保 R ∈ [0, 0.5]。

### 2. 匹配度检测 — 准确性 M(T,S) [Section 2.2]

利用文本相似匹配模型计算数据样本 T 与其模型生成的摘要 S 之间的匹配度，评估模型理解的准确程度。M(T,S) ∈ [0, 0.5]。

### 3. 质量评估 Q(T) [Section 2.3]

$$\displaystyle Q(T) = R(V(T)) + M(T, S)$$

Q(T) ∈ [0, 1.0]，分数越高表示文本越适合用于语言模型训练。

---

## 实验验证 [Section 3]

- **数据集:** 20 Newsgroups (20 类, ~19,000 篇文章)
- **划分:** 8:2 train/test
- **分类器参数 (Table 1):** epochs=10, batch=8, lr=0.01, Leaky-ReLU, max_len=512
- **实验类型:** 删除低质量数据、删除高质量数据、消融实验

---

## 项目结构

```
TDQE_Github/
├── tdqe/                     # 核心 Python 包
│   ├── __init__.py           # 公共接口
│   ├── config.py             # 论文参数常量
│   ├── robustness.py         # 语义一致性检测 (Section 2.1)
│   ├── accuracy.py           # 匹配度检测 (Section 2.2)
│   ├── quality.py            # 质量分数聚合 (Section 2.3)
│   ├── data.py               # 20NG 数据集加载 (Section 3.1)
│   └── experiment.py         # 分类器验证实验 (Section 3)
├── TDQE_All_In_One.ipynb     # 完整流水线 Notebook
├── requirements.txt
├── my_model/                 # 【需自行放置】离线模型权重
├── archive/                  # 【需自行放置】20NG 数据集
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型和数据

- 将预训练模型（如 distilgpt2）放入 `my_model/` 目录
- 将 20 Newsgroups 数据集的 20 个 `.txt` 文件放入 `archive/` 目录

### 3. 运行

```bash
jupyter notebook TDQE_All_In_One.ipynb
```

按顺序执行所有 Cell 即可完成：
1. 加载模型和数据集
2. 计算 TDQE 质量分数
3. 训练分类器并运行数据删除/消融实验

---

## 与论文的对应关系

| 代码模块 | 论文章节 | 说明 |
|---|---|---|
| `tdqe/robustness.py` | Section 2.1 | Formula (1): Dropout m=3 次前向传播，欧氏距离均值 |
| `tdqe/accuracy.py` | Section 2.2 | 文本-摘要嵌入余弦相似度，缩放到 [0, 0.5] |
| `tdqe/quality.py` | Section 2.3 | Formula (2): Q(T) = R + M |
| `tdqe/data.py` | Section 3.1 | 20NG 数据集, 8:2 划分 |
| `tdqe/experiment.py` | Section 3.3–3.5 | 数据删除实验、消融实验 |
| `tdqe/config.py` | Table 1 | m=3, max_len=512, epochs=10, batch=8, lr=0.01 |

## 参考文献

[1] Longpre S, et al. A pretrainer's guide to training data. NAACL, 2024.
[2] Ferdowsi H, et al. An online outlier identification and removal scheme. IEEE TNNLS, 2014.
[3] Frénay B, Verleysen M. Classification in the presence of label noise. IEEE TNNLS, 2014.
[5] Schoch S, et al. CS-SHAPLEY. NeurIPS, 2022.
[6] Ghorbani A, Zou J. Data Shapley. ICML, 2019.
[7] Yoon J, et al. DVRL. ICML, 2020.
[9] Srivastava N, et al. Dropout. JMLR, 2014.
[16] Devlin J, et al. BERT. NAACL, 2019.
[19] 20 Newsgroups dataset.
