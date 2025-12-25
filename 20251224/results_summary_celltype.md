# PBMC 5k 细胞类型分类实验结果汇总

## 实验配置

- **数据集**: PBMC_5k_celltype
- **标签类型**: Cell type (6 classes)
- **细胞类型**: Monocytes, B cells, NK cells, CD4 T cells, CD8 T cells, Other
- **迭代次数**: 50
- **标签数据比例**: 0.1
- **模型**: mlp
- **隐藏层维度**: 64

## 方法对比

### VIME

| 方法 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| Supervised | 48.86% | 0.97% | 47.06% | 51.69% |
| Autoencoder | 50.58% | 2.99% | 34.29% | 53.69% |
| VIME-Self | 50.53% | 1.28% | 44.81% | 52.44% |
| VIME | 49.96% | 0.97% | 47.06% | 51.69% |

### VAT

| 方法 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| Supervised | 48.86% | 0.97% | 47.06% | 51.69% |
| Autoencoder | 50.58% | 2.99% | 34.29% | 53.69% |
| VIME-Self | 50.53% | 1.28% | 44.81% | 52.44% |
| VAT | 49.67% | 0.84% | 47.06% | 51.44% |

### VAT_VIME_Feature

| 方法 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| Supervised | 48.86% | 0.97% | 47.06% | 51.69% |
| Autoencoder | 49.33% | 1.42% | 42.30% | 51.06% |
| VIME-Layer3 | 46.73% | 2.84% | 42.55% | 50.94% |
| VAT-on-X | 48.01% | 2.31% | 42.68% | 50.81% |
| RandomNoise-on-X | 47.85% | 2.46% | 42.05% | 51.31% |
| VAT-on-Layer1 | 49.73% | 1.16% | 46.06% | 51.94% |
| VAT-on-Layer2 | 49.56% | 1.45% | 45.56% | 52.07% |
| VAT-on-Layer3 | 49.08% | 1.82% | 44.43% | 52.07% |
| VAT-on-Concat | 49.34% | 0.65% | 48.31% | 51.31% |

## 综合对比（主要方法）

| 方法 | VIME | VAT | VAT_VIME_Feature |
|------|------|-----|------------------|
| Supervised | 48.86% | 48.86% | 48.86% |
| Autoencoder | 50.58% | 50.58% | 49.33% |
| VIME | 49.96% | - | - |
| VAT | - | 49.67% | - |
| VAT-on-Layer1 (最佳) | - | - | 49.73% |

## 数据来源

| 方法 | 结果文件位置 |
|------|-------------|
| VIME | `VIME/results_celltype/summary_celltype_labels1000_iters50.json` |
| VAT | `VAT/results/summary_mnist_noise0_labels1000_iters50.json` |
| VAT_VIME_Feature | `VAT_VIME_Feature/results/summary_mnist_noise0_labels1000_VAT_iters50.json` |

---

*结果基于 50 次独立实验的平均值*