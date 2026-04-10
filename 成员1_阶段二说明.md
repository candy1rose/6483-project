任务：
- Dogs vs. Cats 的 baseline CNN
- ResNet 训练
- 参数调优
- 保存 best model
- 生成 validation 结果
- 输出 `submission.csv`

以上内容目前都已经完成，下面按任务逐项说明。

## 任务 5：数据预处理

已在 [src/data.py](/Users/wang202007.126.com/Desktop/S2/6483/project/src/data.py) 中实现：

- 统一 `Resize`
- `ToTensor`
- `Normalize`
- 自动识别 `train/ val/ test/` 目录
- 支持固定随机种子抽样
- 支持按指定数量抽取训练集和验证集

本次阶段二实验实际使用：

- 训练集 `2000` 张
- 验证集 `1000` 张
- 测试集 `500` 张

## 任务 6：数据增强

训练集增强已在 [src/data.py](/Users/wang202007.126.com/Desktop/S2/6483/project/src/data.py) 中实现：

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`

验证集和测试集只使用固定预处理，不使用随机增强。

## 任务 7：建立 baseline 模型

已完成两个模型，均定义在 [src/models.py](/Users/wang202007.126.com/Desktop/S2/6483/project/src/models.py) 中：

- 自定义 `SimpleCNN` baseline
- 预训练 `ResNet18` 迁移学习模型

这两个模型已经完成实际训练，可直接用于报告中的模型比较部分。

## 任务 8：设计训练流程

已在 [train_dogcat.py](/Users/wang202007.126.com/Desktop/S2/6483/project/train_dogcat.py) 中实现并实际跑通：

- 标准训练循环
- validation 评估
- best checkpoint 自动保存
- 固定随机种子
- 训练和验证的 loss / accuracy 记录
- 曲线图导出到 `outputs/figures/`
- 指标 JSON 导出到 `outputs/checkpoints/`

## 任务 9：调参

训练脚本支持直接调这些参数：

- `--lr`
- `--batch-size`
- `--optimizer`
- `--epochs`
- `--image-size`
- `--no-augmentation`
- `--model`
- `--train-samples`
- `--val-samples`

本阶段已完成的核心实验对比包括：

- `SimpleCNN` vs `ResNet18`
- `image-size=128` vs `224`
- `lr=1e-3` vs `1e-4`
- baseline CNN vs transfer learning

## 任务 10：选出最好模型

训练脚本已按 `val_acc` 自动保存最佳模型。

当前阶段二最佳模型为 `ResNet18`，路径如下：

- [resnet18_2000_1000_best.pt](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/resnet18_2000_1000_best.pt)

baseline CNN 的最佳模型路径如下：

- [simple_cnn_2000_1000_best.pt](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/simple_cnn_2000_1000_best.pt)

## 任务 11：生成 submission.csv

已在 [predict_dogcat.py](/Users/wang202007.126.com/Desktop/S2/6483/project/predict_dogcat.py) 中实现并已实际生成：

- 读取 checkpoint
- 加载测试集图片
- 输出 `id,label`
- 自动按 `id` 排序
- 保存到 `outputs/submissions/submission.csv`

最终提交文件路径：

- [submission.csv](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/submissions/submission.csv)

标签映射：

- `0 = cat`
- `1 = dog`

## 阶段二实验结果汇总

### 1. SimpleCNN baseline

结果文件：

- [simple_cnn_2000_1000_metrics.json](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/simple_cnn_2000_1000_metrics.json)
- [simple_cnn_2000_1000_curves.png](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/figures/simple_cnn_2000_1000_curves.png)

关键结果：

- 训练图像数：`2000`
- 验证图像数：`1000`
- 输入尺寸：`128x128`
- epoch：`5`
- best validation accuracy：`66.6%`

### 2. ResNet18 主模型

结果文件：

- [resnet18_2000_1000_metrics.json](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/resnet18_2000_1000_metrics.json)
- [resnet18_2000_1000_curves.png](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/figures/resnet18_2000_1000_curves.png)

关键结果：

- 训练图像数：`2000`
- 验证图像数：`1000`
- 输入尺寸：`224x224`
- epoch：`5`
- best validation accuracy：`97.9%`

### 3. 模型对比结论

| Model | Train Images | Val Images | Input Size | Epochs | Best Val Accuracy |
|---|---:|---:|---:|---:|---:|
| SimpleCNN | 2000 | 1000 | 128x128 | 5 | 66.6% |
| ResNet18 (pretrained) | 2000 | 1000 | 224x224 | 5 | 97.9% |

结论：

- `ResNet18` 明显优于 `SimpleCNN`
- `ResNet18` 比 baseline CNN 高 `31.3` 个百分点
- `ResNet18` 可作为阶段二最终主模型

## 当前可直接交付的内容

训练代码：

- [train_dogcat.py](/Users/wang202007.126.com/Desktop/S2/6483/project/train_dogcat.py)

预测代码：

- [predict_dogcat.py](/Users/wang202007.126.com/Desktop/S2/6483/project/predict_dogcat.py)

数据与模型模块：

- [src/data.py](/Users/wang202007.126.com/Desktop/S2/6483/project/src/data.py)
- [src/models.py](/Users/wang202007.126.com/Desktop/S2/6483/project/src/models.py)
- [src/utils.py](/Users/wang202007.126.com/Desktop/S2/6483/project/src/utils.py)

训练结果与最佳模型：

- [simple_cnn_2000_1000_best.pt](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/simple_cnn_2000_1000_best.pt)
- [resnet18_2000_1000_best.pt](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/resnet18_2000_1000_best.pt)

validation 结果与训练日志：

- [simple_cnn_2000_1000_metrics.json](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/simple_cnn_2000_1000_metrics.json)
- [resnet18_2000_1000_metrics.json](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/checkpoints/resnet18_2000_1000_metrics.json)

训练曲线图：

- [simple_cnn_2000_1000_curves.png](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/figures/simple_cnn_2000_1000_curves.png)
- [resnet18_2000_1000_curves.png](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/figures/resnet18_2000_1000_curves.png)

最终测试集提交文件：

- [submission.csv](/Users/wang202007.126.com/Desktop/S2/6483/project/outputs/submissions/submission.csv)

补充说明文件：

- [README.md](/Users/wang202007.126.com/Desktop/S2/6483/project/README.md)
- [requirements.txt](/Users/wang202007.126.com/Desktop/S2/6483/project/requirements.txt)

## 成员 1 在阶段二中的贡献总结

Member 1 completed the Dogs vs. Cats model building and training pipeline, implemented the baseline CNN and pretrained ResNet18, tuned the main training settings, saved the best checkpoints, generated validation results and training curves, and exported the final submission.csv for the test set.
