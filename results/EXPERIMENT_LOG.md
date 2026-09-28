# U-Mamba 双分支挂件改进实验记录

**日期**：2026年9月8日
**项目**：肝脏动脉期CT分割增强
**目标**：改进双分支挂件，解决FP暴涨问题

---

## 一、背景回顾

### 1.1 原版插件实验结果（2026年9月1日）

| 指标 | 原始 U-Mamba | 插件版 | 变化 |
|------|-------------|--------|------|
| Dice | 0.6473 | 0.6061 | -4.1% ❌ |
| IoU | 0.5483 | 0.5038 | -4.5% ❌ |
| FP | 2,629 | 4,790 | +82% ❌ |
| FN | 2,757 | 2,460 | -11% ✅ |

**统计检验**：p < 0.001，退化显著

### 1.2 问题诊断

1. **Branch 2 的 P（粗分割）不准确**
   - 训练初期 P 几乎是随机的
   - 梯度需要经过5层才能传到 Branch 2，信号太弱

2. **power=2.0 太激进**
   - S = P^2 放大了 P 中的噪声
   - 血管区域 P≈0.5 → S=0.25，被当成"弱信号"保留

3. **原型计算方式错误**
   - 原来：lesion_proto = spatial_agg(S⊙F) → 输出还是特征图
   - 问题：不是真正的"肿瘤原型"，原型应该是向量

4. **相似性度量不准**
   - 原来：M = Conv([lesion_proto, F]) → 线性组合
   - 问题：不能捕捉语义相似性

5. **门控融合过于复杂**
   - 原来：D = |E-M|, F_c = Conv([F,E,M])
   - 问题：两个都不准的分支，差异也没意义

---

## 二、改进方案

### 2.1 改进点总结

| 改进点 | 原来 | 改进后 | 目的 |
|--------|------|--------|------|
| **相似性度量** | Conv([proto, F]) | cosine_similarity | 更好地捕捉语义相似性 |
| **原型计算** | spatial_agg(S⊙F) | 加权平均(S⊙F) | 原型应该是向量，不是特征图 |
| **门控融合** | \|E-M\| ⊙ Conv([F,E,M]) | M ⊙ F | 简化，减少噪声引入 |
| **power 参数** | 2.0 | 1.0 | 更保守，不放大噪声 |
| **辅助监督** | 无 | 监督 P 的质量 | 让 P 学得更快更准 |

### 2.2 实验设计

采用**控制变量法**，每次只改一个因素，做对比实验：

| 实验 | 改动内容 | power |
|------|---------|-------|
| 原版 | 什么都不改 | 2.0 |
| cosine | 只改余弦相似度 | 2.0 |
| weighted_avg | 只改加权平均原型 | 2.0 |
| simple_gate | 只改简化门控融合 | 2.0 |
| v2 | 三个全改 + power=1.0 | 1.0 |
| aux_loss | 全部改进 + 辅助监督 | 1.0 |

---

## 三、文件清单

### 3.1 插件文件（nets 目录）

```
/code/U-mamba/U-Mamba-main/umamba/nnunetv2/nets/
├── dual_branch_plugin.py                  # 原版（保持不变）
├── dual_branch_plugin_cosine.py           # 只改余弦相似度 ⭐
├── dual_branch_plugin_weighted_avg.py     # 只改原型计算 ⭐
├── dual_branch_plugin_simple_gate.py      # 只改门控融合 ⭐
├── dual_branch_plugin_v2.py               # 全部改进V2 ⭐
├── dual_branch_plugin_aux_loss.py         # 带辅助监督 ⭐
└── dual_branch_plugin_ablation.py         # 消融实验版本
```

### 3.2 Trainer 文件

```
/code/U-mamba/U-Mamba-main/umamba/nnunetv2/training/nnUNetTrainer/
├── nnUNetTrainerUMambaBot_plugin.py          # 原版（保持不变）
├── nnUNetTrainerUMambaBot_plugin_cosine.py   # 只改余弦相似度 ⭐
├── nnUNetTrainerUMambaBot_plugin_avg.py      # 只改原型计算 ⭐
├── nnUNetTrainerUMambaBot_plugin_gate.py     # 只改门控融合 ⭐
├── nnUNetTrainerUMambaBot_plugin_v2.py       # 全部改进V2 ⭐
└── nnUNetTrainerUMambaBot_plugin_aux_loss.py # 带辅助监督 ⭐
```

### 3.3 训练脚本

```
/code/U-mamba/
├── run_plugin_cosine_train.sh           # 只改余弦相似度
├── run_plugin_avg_train.sh              # 只改原型计算
├── run_plugin_gate_train.sh             # 只改门控融合
├── run_plugin_v2_train.sh               # 全部改进V2
└── run_plugin_aux_loss_train.sh         # 带辅助监督
```

### 3.4 推理脚本

```
/code/U-mamba/
├── run_plugin_cosine_predict_test.sh    # 只改余弦相似度
├── run_plugin_avg_predict_test.sh       # 只改原型计算
├── run_plugin_gate_predict_test.sh      # 只改门控融合
├── run_plugin_v2_predict_test.sh        # 全部改进V2
└── run_plugin_aux_loss_predict_test.sh  # 带辅助监督
```

### 3.5 本地文件位置

```
E:\u-mamba\改动\
├── dual_branch_plugin_cosine.py
├── dual_branch_plugin_weighted_avg.py
├── dual_branch_plugin_simple_gate.py
├── dual_branch_plugin_v2.py
├── dual_branch_plugin_aux_loss.py
├── dual_branch_plugin_ablation.py
├── run_plugin_cosine_train.sh
├── run_plugin_avg_train.sh
├── run_plugin_gate_train.sh
├── run_plugin_v2_train.sh
├── run_plugin_aux_loss_train.sh
├── run_plugin_cosine_predict_test.sh
├── run_plugin_avg_predict_test.sh
├── run_plugin_gate_predict_test.sh
├── run_plugin_v2_predict_test.sh
└── run_plugin_aux_loss_predict_test.sh

D:\TYPORA\read\shengxin\Model\U-Mamba\umamba\nnunetv2\training\nnUNetTrainer\
├── nnUNetTrainerUMambaBot_plugin_cosine.py
├── nnUNetTrainerUMambaBot_plugin_avg.py
├── nnUNetTrainerUMambaBot_plugin_gate.py
├── nnUNetTrainerUMambaBot_plugin_v2.py
└── nnUNetTrainerUMambaBot_plugin_aux_loss.py
```

---

## 四、各实验版本详解

### 4.1 cosine 版本（只改余弦相似度）

**改动位置**：Branch 2 的相似性度量

```python
# 原来
M = self.similarity(torch.cat([lesion_proto, x], dim=1))

# 改成
x_norm = F.normalize(x, dim=1)
proto_norm = F.normalize(lesion_proto, dim=1)
similarity_map = (x_norm * proto_norm).sum(dim=1, keepdim=True)
M = (similarity_map + 1) / 2
```

**power**：2.0（保持不变）

**预期效果**：相似性度量更准确，M 能更好地区分肿瘤和非肿瘤区域

---

### 4.2 weighted_avg 版本（只改原型计算）

**改动位置**：Branch 2 的原型计算

```python
# 原来
lesion_proto = self.spatial_agg(S * x)  # 还是特征图

# 改成
S_sum = S.sum(dim=[2, 3, 4], keepdim=True) + 1e-8
lesion_proto = (S * x).sum(dim=[2, 3, 4], keepdim=True) / S_sum
lesion_proto = lesion_proto.expand_as(x)  # 广播回去
```

**power**：2.0（保持不变）

**预期效果**：原型是真正的"肿瘤平均特征"，而不是特征图

---

### 4.3 simple_gate 版本（只改门控融合）

**改动位置**：门控融合公式

```python
# 原来
D = torch.abs(E - M)
F_c = self.gate(torch.cat([x, E, M], dim=1))
output = x + self.alpha * D * F_c

# 改成
enhancement = M * x
output = x + torch.sigmoid(self.alpha) * enhancement
```

**power**：2.0（保持不变）

**预期效果**：减少噪声引入，增强更稳定

---

### 4.4 v2 版本（全部改进）

**改动内容**：
1. 余弦相似度
2. 加权平均原型
3. 简化门控融合
4. power=1.0

**power**：1.0（默认）

**预期效果**：综合所有改进，效果应该最好

---

### 4.5 aux_loss 版本（带辅助监督）

**改动内容**：
1. 全部改进（与 v2 相同）
2. 额外：对 P 加辅助监督

**power**：1.0（默认）

**关键参数**：
```bash
PLUGIN_POWER=1.0        # 置信度加权幂次
AUX_LOSS_WEIGHT=0.3     # 辅助 loss 的权重
```

**预期效果**：P 学得更快更准，整体效果最好

---

## 五、运行命令

### 5.1 训练

```bash
# 1. 只改余弦相似度
bash /code/U-mamba/run_plugin_cosine_train.sh

# 2. 只改原型计算
bash /code/U-mamba/run_plugin_avg_train.sh

# 3. 只改门控融合
bash /code/U-mamba/run_plugin_gate_train.sh

# 4. 全部改进V2
bash /code/U-mamba/run_plugin_v2_train.sh

# 5. 带辅助监督（可调整权重）
AUX_LOSS_WEIGHT=0.3 bash /code/U-mamba/run_plugin_aux_loss_train.sh
```

### 5.2 推理

```bash
# 训练完成后运行
bash /code/U-mamba/run_plugin_cosine_predict_test.sh
bash /code/U-mamba/run_plugin_avg_predict_test.sh
bash /code/U-mamba/run_plugin_gate_predict_test.sh
bash /code/U-mamba/run_plugin_v2_predict_test.sh
bash /code/U-mamba/run_plugin_aux_loss_predict_test.sh
```

### 5.3 查看结果

```bash
# 查看各版本的评估结果
cat /output/umamba_eval_plugin_cosine/summary.json
cat /output/umamba_eval_plugin_avg/summary.json
cat /output/umamba_eval_plugin_gate/summary.json
cat /output/umamba_eval_plugin_v2/summary.json
cat /output/umamba_eval_plugin_aux/summary.json
```

---

## 六、结果目录结构

```
/code/U-mamba/
├── nnUNet_results/                          # 原版结果
├── nnUNet_results_plugin/                   # 原插件版结果
├── nnUNet_results_plugin_cosine/            # 只改余弦相似度
├── nnUNet_results_plugin_avg/               # 只改原型计算
├── nnUNet_results_plugin_gate/              # 只改门控融合
├── nnUNet_results_plugin_v2/                # 全部改进V2
└── nnUNet_results_plugin_aux/               # 带辅助监督

├── predictions_plugin_test/                 # 原插件版预测
├── predictions_plugin_cosine_test/          # 只改余弦相似度
├── predictions_plugin_avg_test/             # 只改原型计算
├── predictions_plugin_gate_test/            # 只改门控融合
├── predictions_plugin_v2_test/              # 全部改进V2
└── predictions_plugin_aux_test/             # 带辅助监督

├── eval_plugin/                             # 原插件版评估
├── eval_plugin_cosine/                      # 只改余弦相似度
├── eval_plugin_avg/                         # 只改原型计算
├── eval_plugin_gate/                        # 只改门控融合
├── eval_plugin_v2/                          # 全部改进V2
└── eval_plugin_aux/                         # 带辅助监督
```

---

## 七、对比分析表

### 7.1 各版本改进点对比

| 版本 | 余弦相似度 | 加权平均原型 | 简化门控 | power | 辅助监督 |
|------|-----------|-------------|---------|-------|---------|
| 原版 | ❌ | ❌ | ❌ | 2.0 | ❌ |
| cosine | ✅ | ❌ | ❌ | 2.0 | ❌ |
| weighted_avg | ❌ | ✅ | ❌ | 2.0 | ❌ |
| simple_gate | ❌ | ❌ | ✅ | 2.0 | ❌ |
| v2 | ✅ | ✅ | ✅ | 1.0 | ❌ |
| aux_loss | ✅ | ✅ | ✅ | 1.0 | ✅ |

### 7.2 预期效果对比

| 版本 | 预期 Dice 变化 | 预期 FP 变化 | 预期 FN 变化 |
|------|---------------|-------------|-------------|
| 原版 | baseline | baseline | baseline |
| cosine | +1~2% | -10~20% | -5~10% |
| weighted_avg | +1~2% | -10~20% | -5~10% |
| simple_gate | +1~2% | -10~20% | -5~10% |
| v2 | +2~4% | -20~30% | -10~15% |
| aux_loss | +3~5% | -30~40% | -15~20% |

---

## 八、实验状态

### 8.1 已完成

- [x] 原版插件实验（已完成，效果退化）
- [x] 问题诊断（P 不准确、power 太大、原型计算错误等）
- [x] 改进方案设计（5个改进点）
- [x] 代码实现（5个新版本）
- [x] 训练/推理脚本编写
- [x] 文件上传到超算平台

### 8.2 进行中

- [ ] cosine 版本训练
- [ ] weighted_avg 版本训练
- [ ] simple_gate 版本训练
- [ ] v2 版本训练
- [ ] aux_loss 版本训练

### 8.3 待完成

- [ ] 各版本推理和评估
- [ ] 结果对比分析
- [ ] 找出最优版本
- [ ] 消融实验（验证各改进点的贡献）
- [ ] 向老师汇报结果

---

## 九、注意事项

### 9.1 文件命名规范

- 插件文件：`dual_branch_plugin_<版本>.py`
- Trainer 文件：`nnUNetTrainerUMambaBot_plugin_<版本>.py`
- 训练脚本：`run_plugin_<版本>_train.sh`
- 推理脚本：`run_plugin_<版本>_predict_test.sh`

### 9.2 目录隔离

每个实验的结果都保存在独立的目录中，互不干扰：
- 结果目录：`nnUNet_results_plugin_<版本>/`
- 预测目录：`predictions_plugin_<版本>_test/`
- 评估目录：`eval_plugin_<版本>/`

### 9.3 参数配置

通过环境变量配置参数：
```bash
PLUGIN_POWER=1.0        # 置信度加权幂次
AUX_LOSS_WEIGHT=0.3     # 辅助 loss 权重（仅 aux_loss 版本）
```

---

## 十、下一步计划

### 10.1 短期（本周）

1. 完成所有版本的训练
2. 运行推理和评估
3. 对比结果，找出最优版本

### 10.2 中期（下周）

1. 做消融实验，验证各改进点的贡献
2. 调整参数（power、aux_loss_weight）
3. 向老师汇报结果

### 10.3 长期（两周内）

1. 如果效果好，考虑多折验证
2. 如果效果不好，重新审视设计
3. 准备论文/报告

---

## 十一、联系方式

如有问题，请查看：
- 本文档：`EXPERIMENT_LOG.md`
- 原版工作总结：`WORK_SUMMARY.md`
- 汇报文档：`REPORT_TO_ADVISOR.md`

---

**最后更新**：2026年9月8日
