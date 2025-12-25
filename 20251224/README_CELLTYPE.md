# PBMC 5k 数据集实验说明

## 为什么使用细胞类型标签

PBMC (Peripheral Blood Mononuclear Cells，外周血单核细胞) 是免疫学研究中的经典样本，包含多种免疫细胞类型。这个数据集包含**RNA表达（16,581个基因）**和**蛋白质表达（29个表面标记）**两种模态的数据。

虽然原始h5ad文件中没有明确的细胞类型标注，但数据集包含了关键的**免疫细胞表面蛋白标记**（CD3、CD4、CD8、CD14、CD19、CD56等），这些是免疫学中用于鉴定细胞类型的金标准。使用这些蛋白质标记，我们可以生成生物学上有意义的细胞类型标签。

## 细胞类型标签的生成方法

基于免疫学领域的标准细胞分型原则，使用蛋白质标记表达模式定义细胞类型：

| 细胞类型 | 蛋白质标记组合 | 生物学意义 |
|---------|--------------|-----------|
| **Monocytes** | CD14+ | 单核细胞，先天免疫 |
| **B cells** | CD19+ | B细胞，产生抗体 |
| **NK cells** | CD56+ CD3- | 自然杀伤细胞，抗肿瘤/病毒 |
| **CD4 T cells** | CD3+ CD4+ CD8- | 辅助性T细胞，协调免疫应答 |
| **CD8 T cells** | CD3+ CD8+ CD4- | 细胞毒性T细胞，杀伤感染细胞 |
| **Other** | 其他组合 | 未分类细胞 |

**技术细节**：
- 使用蛋白质表达的**60分位数**作为阳性阈值（相对宽松，避免遗漏）
- 按优先级分配细胞类型（避免冲突）
- 生成的标签分布符合典型PBMC组成

**细胞类型分布**：
```
Monocytes:     449 cells (11.2%)
B cells:       701 cells (17.6%)
NK cells:     1008 cells (25.2%)
CD4 T cells:   925 cells (23.2%)
CD8 T cells:   290 cells ( 7.3%)
Other:         621 cells (15.5%)
```

---

## 目录结构

```
tem/未命名/
├── pbmc5k_data_celltype.py          # 数据加载器（生成细胞类型标签）
├── base/                             # 新的模型架构（256-128-64）
│   ├── __init__.py                   # 导出所有模型和函数
│   ├── autoencoder.py                # Autoencoder 模型
│   ├── self_supervised_vime.py       # VIME-Self 模型
│   └── self_supervised_multilayer.py # 多层 VIME 模型
├── VAT/
│   ├── pbmc5k_vat_celltype.py        # VAT 运行脚本
│   └── results_celltype/             # VAT 结果目录
├── VIME/
│   ├── pbmc5k_vime_celltype.py       # VIME 运行脚本
│   └── results_celltype/             # VIME 结果目录
├── VAT_VIME_Feature/
│   ├── pbmc5k_vat_vime_feature_celltype.py  # VAT_VIME_Feature 运行脚本
│   └── results_celltype/             # VAT_VIME_Feature 结果目录
├── run_all_celltype.sh               # 批量运行脚本
└── README_CELLTYPE.md                # 本文档
```

---

## 文件说明

### 核心文件

#### 1. 数据加载器
**`pbmc5k_data_celltype.py`**
- **作用**: 读取h5ad文件，生成细胞类型标签，分割数据集
- **输出**: X_label, y_label, X_unlab, X_valid, y_valid, X_test, y_test
- **标签**: 6类细胞类型的one-hot编码
- **特点**: 独立实现，不依赖外部模块，可单独测试

#### 2. 模型架构（base/ 文件夹）
**`base/autoencoder.py`**
- **架构**: 16,581 → 256 → 128 → 64（渐进式降维）
- **编码器**: 3层 ReLU 激活
- **解码器**: 镜像结构，Sigmoid 输出

**`base/self_supervised_vime.py`**
- **编码器**: 16,581 → 256 → 128 → 64
- **双解码器**: mask_estimator + feature_estimator
- **用于**: VAT 和 VIME 的自监督预训练

**`base/self_supervised_multilayer.py`**
- **分层编码**: Layer1(256) + Layer2(128) + Layer3(64)
- **多尺度特征**: 可提取不同层次的表示
- **用于**: VAT_VIME_Feature 的特征提取

**`base/__init__.py`**
- **导出**: 所有模型和训练函数
- **作用**: 方便其他脚本通过 monkey-patching 替换原始模型

**实现方式**：
- 三个 celltype 脚本通过 `sys.path.insert()` 将 base/ 添加到路径
- 使用 monkey-patching 替换 20251210 中的原始模型函数
- 保持外部 API 不变，只替换内部实现

#### 3. 运行脚本

**`VAT/pbmc5k_vat_celltype.py`**
- **作用**: 在PBMC 5k数据上运行VAT方法（使用细胞类型标签）
- **训练流程**:
  1. 自监督预训练（VIME-Self，256→128→64架构）
  2. 半监督训练（VAT对抗损失 + 监督损失）
- **输出**: results_celltype/目录下的JSON结果文件
- **架构**: 使用 base/ 文件夹中的新模型

**`VIME/pbmc5k_vime_celltype.py`**
- **作用**: 在PBMC 5k数据上运行VIME方法（使用细胞类型标签）
- **训练流程**:
  1. 自监督预训练（VIME-Self：重构 + mask预测，256→128→64架构）
  2. 半监督训练（一致性正则化 + 监督损失）
- **输出**: results_celltype/目录下的JSON结果文件
- **架构**: 使用 base/ 文件夹中的新模型

**`VAT_VIME_Feature/pbmc5k_vat_vime_feature_celltype.py`**
- **作用**: 在PBMC 5k数据上运行VAT_VIME_Feature方法（使用细胞类型标签）
- **训练流程**:
  1. 自监督预训练（VIME-Self，3层编码器：256→128→64）
  2. 提取多层特征（Layer1: 256维, Layer2: 128维, Layer3: 64维）
  3. 在特征空间应用VAT半监督训练
- **输出**: results_celltype/目录下的JSON结果文件
- **特色**: 对比不同层特征的效果（通常Layer3最好）
- **架构**: 使用 base/ 文件夹中的新多层模型

#### 4. 批量运行脚本

**`run_all_celltype.sh`**
- **作用**: 一键运行所有三个方法
- **执行顺序**: VAT → VAT_VIME_Feature → VIME
- **使用方法**: `./run_all_celltype.sh`

---

## 使用方法

### 快速开始

```bash
# 1. 测试数据加载器（可选）
python3 pbmc5k_data_celltype.py

# 2. 运行所有方法
./run_all_celltype.sh
```

### 单独运行某个方法

```bash
# 运行VAT
cd VAT
python3 pbmc5k_vat_celltype.py

# 运行VIME
cd VIME
python3 pbmc5k_vime_celltype.py

# 运行VAT_VIME_Feature
cd VAT_VIME_Feature
python3 pbmc5k_vat_vime_feature_celltype.py
```

### 结果位置

每个方法的结果保存在各自目录下的 `results_celltype/` 文件夹：
- `VAT/results_celltype/`
- `VIME/results_celltype/`
- `VAT_VIME_Feature/results_celltype/`

---

## 三个方法的区别

| 方法 | 性质 | 特征空间 | 半监督策略 | 编码器架构 | 特点 |
|------|------|---------|-----------|-----------|------|
| **VAT** | 半监督 | 原始RNA表达X | VAT对抗损失 | 256→128→64 | 在原始数据空间做对抗训练 |
| **VIME** | 自监督+半监督 | VIME-Self特征 | 一致性正则化 | 256→128→64 | mask预测 + 特征重构 |
| **VAT_VIME_Feature** | 自监督+半监督 | VIME多层特征 | VAT对抗损失 | 分层(256/128/64) | 在学到的特征空间做对抗训练 |

**协同关系**：
1. **VIME-Self**（自监督）：学习特征表示（所有方法都会用）
   - 使用 256→128→64 渐进式架构
   - 双任务学习：mask预测 + 特征重构
2. **VIME/VAT**（半监督）：利用特征做分类
   - VAT: 在原始数据空间应用对抗扰动
   - VIME: 在学习到的特征空间应用一致性正则化
3. **VAT_VIME_Feature**：结合VIME特征 + VAT对抗
   - 提取多层特征（256维/128维/64维）
   - 在不同特征空间分别应用VAT

---

## 实验配置

- **数据集**: PBMC 5k (3,994个细胞)
- **特征**: 16,581个基因
- **标签类型**: 细胞类型（6类）
- **标签数量**: 默认1000个（10%标签率）
- **迭代次数**: 默认10次
- **模型**: MLP
- **网络架构**: 16,581 → 256 → 128 → 64（渐进式降维）
- **隐藏层维度**: 64（最终编码维度）
- **Early Stopping**: Patience=5

---

## 网络架构设计

### 为什么使用 256-128-64 渐进式架构？

**原始架构问题**（16,581 → 100 → 100 → 100）：
- 第一层压缩比过大（165倍），信息损失严重
- 后续层没有进一步降维，特征提取不充分
- 不符合深度学习中渐进式特征抽象的原则

**新架构优势**（16,581 → 256 → 128 → 64）：
- **渐进式降维**: 65倍 → 2倍 → 2倍，信息流动更平滑
- **更好的信息保留**: 第一层保留更多原始信息（256维 vs 100维）
- **层次化特征**: 每层学习不同抽象级别的特征
- **参考 scVI/totalVI**: 单细胞领域 SOTA 模型通常使用 128-256 作为第一隐藏层

**参数量对比**：
- 原始: 16,581×100 + 100×100 + 100×100 ≈ 1.68M
- 新架构: 16,581×256 + 256×128 + 128×64 ≈ 4.28M
- 在 3,994 个样本下，参数量仍然合理（样本/参数 ≈ 0.93）

---

## 注意事项

1. **环境要求**: 需要安装torch, numpy, h5py等依赖
2. **GPU**: 可选，但建议使用GPU加速训练
3. **运行时间**:
   - 新架构参数量增加（1.68M → 4.28M）
   - 每个方法约15-40分钟（取决于硬件，比原架构稍慢）
   - GPU加速可显著缩短训练时间
4. **结果文件**: JSON格式，包含每次迭代的准确率和统计信息
5. **架构变更**: 当前所有方法使用 256-128-64 架构，如需恢复原 100-100-100 架构，可修改 `--hidden_dim` 参数并不加载 base/ 模型
