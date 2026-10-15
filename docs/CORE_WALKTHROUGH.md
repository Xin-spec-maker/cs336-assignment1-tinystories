# HW1 核心链路复习与汇报提纲

这个项目实际从随机初始化训练了一个小型 decoder-only Transformer。模型结构、BPE、分词、注意力、损失和 AdamW 使用本项目实现；PyTorch 提供张量运算、自动微分和 MPS 设备执行。没有加载预训练语言模型权重，没有调用现成 Transformer、Attention 或 AdamW 替代核心实现。

## 1. 原始文本如何变成训练数据

来源是 README 指定的 TinyStoriesV2-GPT4 文本文件，下载时固定 Hugging Face dataset revision。为适配本次“核心跑通”的目标，仅选择官方 train 文件的前 20,000 篇完整故事、独立 valid 文件的前 1,000 篇完整故事，而非全量语料。清理每篇故事两端空白，保留内部文本，末尾添加 `<|endoftext|>` 和换行。

验证故事按清理后的完整文本哈希与训练故事比较，去除完全相同的故事；本次交集为零。该检查不代表做过语义近重复检测。分词器只使用训练故事，验证集不参与 BPE 学习。

**BPE training 与 encoding 是两件事：** 前者在训练文本中学习合并规则，后者用固定规则把任意输入变成整数 ID。前者没有神经网络优化器，也不是 LM 的梯度训练。

- 用训练集前 2,000 篇故事训练 4096 大小的 byte-level BPE 词表。
- 初始包含 256 个字节及特殊 token；预分词限制可合并的边界，特殊 token 不参与普通合并。
- 迭代选择高频相邻 pair，产生更长 token，记录有顺序的 merges。
- `BPETokenizer` 按这些 merges 编码训练集及验证集；不使用 tiktoken 生成训练数据。
- 得到 train 4,124,075 个 tokens，valid 202,722 个 tokens。
- ID 小于 4096，因此磁盘用 uint16；喂给模型前转为 torch.long。
- 保存 `.npy`，训练时 `mmap_mode='r'`，按批取片段，不把整个原始文本常驻 GPU。

原文、词表、merges、token 文件及 manifest 都保存在忽略的 `data/` 中；仓库只保存数据来源、选择规则与 SHA-256 清单。

## 2. 一次训练迭代的张量与计算

本次 B=8、T=256、D=192、H=6、每头宽度 d_k=32、FFN 宽度 F=512、词表 V=4096、层数 L=4。

| 阶段 | 张量形状 | 含义 |
|---|---|---|
| 随机采样窗口 | `[B,T]` | 输入 x 是连续 tokens；标签 y 是同一窗口向右移一位 |
| Embedding | `[B,T,D]` | 每个 token ID 查一行可训练向量 |
| Q/K/V 投影并拆头 | 各 `[B,H,T,d_k]` | 每个位置产生 query、key、value |
| Attention scores | `[B,H,T,T]` | Q 与 K 的相似度除以 sqrt(d_k) |
| 因果 mask + softmax | `[B,H,T,T]` | 位置 t 不能读取 t 之后的 token；最后一维归一化 |
| 加权 V、合并头、输出投影 | `[B,T,D]` | 汇集当前及历史 token 信息 |
| SwiGLU 两条上投影 | 各 `[B,T,F]` | SiLU 激活门控分支，再与另一分支逐元素相乘 |
| FFN 下投影 | `[B,T,D]` | 返回模型宽度，做逐位置的非线性变换 |
| 最终 RMSNorm + LM head | `[B,T,V]` | 每个位置对下一个 token 的所有候选产生 logits |
| Cross-entropy | 标量 | 2048 个下一 token 预测目标的平均负对数概率 |

每个 Transformer Block 使用 pre-norm：先 RMSNorm，再 attention，接残差；再 RMSNorm、SwiGLU、残差。RMSNorm 控制特征尺度；残差保留信息和梯度通路。RoPE 对 Q/K 的相邻坐标对进行与位置有关的旋转，将位置信息引入注意力分数；不对 V 做相同旋转。

4 个 Block 后，再做最终 RMSNorm 和词表投影。输入 Embedding 和输出 LM head 不共享参数。模型输出 logits，训练损失内部进行稳定的归一化计算。

## 3. 损失如何让参数学会预测

Cross-entropy 衡量模型给真实下一个 token 的概率；平均负 log 概率越小越好。训练采用 teacher forcing，一次前向可并行计算窗口内所有位置的下一 token 预测，但因果 mask 保证每个位置看不到答案。

`backward()` 由 PyTorch 自动微分计算所有可训练参数的梯度。梯度裁剪限制所有梯度合起来的 L2 范数为 1.0。自实现 AdamW 维护一阶矩与二阶矩，做偏差校正并使用解耦 weight decay 更新权重。

每步清空旧梯度。本次不使用梯度累积、不使用混合精度，也不使用预训练权重。学习率前 100 步线性 warmup 到 0.001，再按余弦下降到第 2000 步的 0.0001。AdamW beta=(0.9,0.95)，weight decay=0.1。

总参数量为 **3,344,064**：Embedding 786,432，4 个 Block 共 1,771,008，最终 RMSNorm 192，LM head 786,432。

## 4. Validation、日志与 checkpoint

Validation 不做反向传播、不更新权重。定期训练/验证评估各取固定且分散在对应数据中的 32,768 个目标，避免每次换评估样本导致曲线不可比，也不消耗训练采样 RNG。最终验证覆盖所选验证文件的全部 202,721 个下一 token 目标，包含不足一整个 context 的尾段，并按 token 数加权。

Perplexity = exp(平均 cross-entropy)，这里的 loss 使用自然对数。PPL 依赖数据范围与 tokenizer，不能直接把本项目 4096 词表、故事子集的数值当作 handout 全量 10K 词表的成绩。

`metrics.jsonl` 保存训练步数、训练批次 loss 均值、学习率、累计 tokens、验证指标和 elapsed seconds。`last.pt` 保存当前权重、AdamW 状态、已完成步数、模型/训练配置、数据与 tokenizer 指纹、Python/NumPy/Torch/MPS RNG。恢复时配置与数据指纹必须一致，学习率根据全局步数继续，不重新 warmup。

`best.pt` 按固定验证窗口的 loss 选择，`last.pt` 对应最后一步。报告与生成使用 `last.pt`，这样最终指标与被生成的模型对应。保存采用临时文件写入完成后原子替换。

## 5. 从 checkpoint 自回归生成

1. 读取配置构造同一模型，加载训练权重，并核对词表/merges 指纹。
2. 用训练时同一个 tokenizer 编码 prompt。
3. 前向仅取最后一个位置的 logits，因为它预测的是 prompt 后的下一个 token。
4. temperature > 0 时对 logits 做温度缩放：低温更集中，高温更分散；temperature=0 走 greedy argmax。
5. top-p 将概率排序，保留累计概率达到 p 的最小前缀，包括越过阈值的那个 token；重新归一化后采样。
6. 把采到的 token 追加到输入，重复前向；超过 context 时仅截取最后 256 tokens 给模型。
7. 遇到 EOS 或达到生成长度上限停止。EOS 是 token ID 0，不依赖字符串后处理猜测。
8. 拼接 token 对应的字节再 UTF-8 解码，得到文本。

模型训练是并行预测窗口内各位置；生成必须逐个产生新 token。当前实现没有 KV cache，每一步重新计算上下文，优先保持流程清楚。

## 6. 建议的 PPT 顺序

1. 目标与范围：从随机权重到真实 TinyStories 生成；不等同于完成全部作业实验。
2. 数据与 BPE：独立 train/valid、训练词表与编码的区别、EOS 和 memmap。
3. 模型结构：Embedding → 4×Block → RMSNorm → LM head；标注主要形状。
4. 训练迭代：shifted targets → logits → cross-entropy → backward → clip → AdamW。
5. 实测结果：展示 loss 曲线、最终 validation/PPL、设备、耗时及 token 预算。
6. Checkpoint 与 generation：续训恢复什么、temperature/top-p/EOS 如何起作用。
7. 展示真实生成并指出局限：可能出现人名不一致、代词和语法错误；未进行广泛调参或达到正式 benchmark。

结果及完整生成保存在 `reports/tinystories-core/`。本次未进行全量 10K/32K tokenizer 实验、完整 TinyStories 训练预算、大规模调参、架构消融、OWT、leaderboard 或全部书面题。
