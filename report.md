### 2.1.6 基于 GRPO 的强化学习微调

经过前述步数蒸馏和参数蒸馏，学生模型的推理效率得到显著提升，但与原始教师模型相比，蒸馏后的学生模型在图像质量、主体清晰度、语义合理性以及细节表现等方面仍存在一定差距。为进一步改善学生模型的生成效果，本阶段引入基于奖励反馈的强化学习方法，对生成模型进行后训练优化。近年来，强化学习在生成模型优化方面取得了显著进展，其中，群组相对策略优化（Group Relative Policy Optimization，GRPO）通过对同一输入下生成的一组样本进行相对奖励评估，估计各样本的优势函数，在无需额外训练价值函数模型的情况下实现策略更新，相比传统 PPO 方法具有更低的训练开销\(1\)。与此同时，DDPO 等研究已验证强化学习能够直接优化扩散模型的生成过程，改善图像美学质量以及文本与图像的匹配程度\(2\)。因此，本阶段选择 GRPO 作为生成模型强化学习微调的主要技术路线。

在具体算法选择方面，本阶段重点调研了 DanceGRPO\(3\)、Flow-GRPO\(4\) 以及 Neighbor GRPO\(5\) 等方法。DanceGRPO 和 Flow-GRPO 将 GRPO 引入图像生成任务，并针对 Flow Matching 模型采用确定性常微分方程（Ordinary Differential Equation，ODE）采样所带来的策略探索困难，提出将 ODE 采样过程转换为随机微分方程（Stochastic Differential Equation，SDE）采样，以构建随机策略并实现基于奖励的梯度优化。然而，这类方法引入的随机性可能影响原始 ODE 采样轨迹的性质，同时在训练效率以及高阶求解器适配方面存在一定限制。Neighbor GRPO 在此基础上提出进一步改进，通过扰动初始噪声构建相邻 ODE 采样轨迹，并利用基于轨迹距离的代理策略建立样本间的相对优化关系，使模型能够在保留确定性 ODE 采样过程的条件下完成 GRPO 训练。该方法避免了 ODE 到 SDE 的转换，并能够与高阶 ODE 求解器结合，因此更加适合 FLUX 等基于 Flow Matching 架构的图像生成模型，尤其有利于兼顾训练效率与低步数采样质量\(5\)。基于上述研究，本阶段以 DanceGRPO 的开源代码为基础，参考 Neighbor GRPO 的算法设计对采样过程及策略优化方式进行修改，实现适用于本项目生成模型的强化学习训练框架。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2

在奖励函数设计方面，本阶段首先选择 Human Preference Score v2（HPSv2）作为强化学习的奖励模型。HPSv2 基于大规模人类偏好标注数据训练，通过学习不同生成图像之间的人类偏好关系，对图像的整体视觉效果以及文本与图像的匹配程度进行综合评分。相关研究表明，HPSv2 在不同图像生成模型和数据分布下具有较好的泛化能力，能够较为准确地反映人类对文生图结果的主观偏好\(6\)。因此，本阶段首先以 HPSv2 作为单一奖励函数，对教师模型进行 GRPO 强化学习微调。为降低训练成本并便于后续将优化结果迁移至学生模型，训练过程中采用低秩适配（Low-Rank Adaptation，LoRA）技术\(8\)，冻结原始模型参数，仅更新少量低秩参数，最终得到以人类偏好优化为目标的 LoRA_A。初步实验表明，该方法能够改善生成图像的整体视觉表现，为进一步提升蒸馏学生模型的生成质量提供有效的参数增量。

然而，在对蒸馏模型进行初步主观评测时发现，部分提示词下学生模型生成的图像相较教师模型存在不必要的细节增加、局部纹理失真以及背景元素过于复杂等问题。这类现象虽然在部分情况下表现为图像细节更加丰富，但同时容易造成画面主体不突出、背景杂乱以及局部结构不合理，从而影响整体视觉质量。进一步分析发现，HPSv2 主要关注图像整体的人类偏好及图文匹配关系，对局部失真和冗余细节等问题的识别能力存在一定局限。在本项目的部分测试样例中，HPSv2 甚至表现出对细节较为繁杂的图像给予更高评分的倾向，导致单独使用 HPSv2 进行奖励优化时，难以有效消除上述问题，部分情况下还可能进一步强化不必要的细节。

针对这一问题，本阶段进一步调研和测试了其他图像质量评价模型，最终选择 MANIQA（Multi-dimension Attention Network for No-Reference Image Quality Assessment）作为补充奖励函数。MANIQA 是一种基于多维注意力机制的无参考图像质量评价模型，通过结合视觉 Transformer 特征、通道注意力和空间注意力，对图像的局部与整体感知质量进行综合评估。相关研究表明，MANIQA 在多个图像质量评价基准上取得了较好的性能，尤其能够有效处理生成式模型产生的图像失真问题\(7\)。与主要关注人类整体偏好的 HPSv2 不同，MANIQA 更侧重图像本身的感知质量，因此有望对不合理纹理、局部失真等问题提供更有针对性的优化信号。本阶段采用 MANIQA 作为单一奖励函数，沿用 Neighbor GRPO 训练框架对教师模型进行独立微调，得到以感知质量优化为目标的 LoRA_B。初步实验结果表明，采用 MANIQA 奖励训练能够在一定程度上改善图像清晰度，减少不必要的背景细节和局部失真，使整体画面更加自然、协调。因此，本阶段将 MANIQA 确定为与 HPSv2 互补的第二种奖励函数。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

考虑到 HPSv2 和 MANIQA 分别侧重整体人类偏好与图像感知质量，两种奖励函数的优化目标具有一定互补性，本阶段进一步采用多 LoRA 加权融合的方法，综合利用两种强化学习训练所获得的模型能力。已有研究表明，针对不同任务分别训练的 LoRA 模块可以通过加权组合实现能力融合。例如，LoraHub 研究了多个独立训练的 LoRA 模块之间的可组合性，证明通过调整各 LoRA 的组合权重，可以在不重新训练基础模型的条件下实现不同任务能力的迁移与整合\(9\)；ZipLoRA 针对文生图模型提出了独立 LoRA 模块的融合方法，验证了组合不同 LoRA 所学习的图像生成能力的可行性\(10\)；Multi-LoRA Composition for Image Generation 则系统研究了图像生成中的多 LoRA 组合方式，并将按权重线性叠加 LoRA 参数作为一种基础融合方案\(11\)。这些工作为本阶段采用双 LoRA 融合策略提供了技术依据，但同时也表明，不同 LoRA 之间可能存在参数干扰，需要通过合理的权重选择实现能力互补。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+2

具体而言，本阶段分别保留 HPSv2 奖励训练得到的 LoRA_A 和 MANIQA 奖励训练得到的 LoRA_B，并按照不同的权重比例将两组 LoRA 加载至蒸馏后的学生模型中。在适配层结构及参数维度兼容的条件下，融合后的模型参数可以表示为：

Wfinal=Wstudent+λAΔWA+λBΔWBW_{\mathrm{final}}=W_{\mathrm{student}}+\lambda_A\Delta W_A+\lambda_B\Delta W_BWfinal=Wstudent+λAΔWA+λBΔWB

其中，WstudentW_{\mathrm{student}}Wstudent 表示蒸馏后学生模型的原始参数，ΔWA\Delta W_AΔWA 和 ΔWB\Delta W_BΔWB 分别表示两个 LoRA 学习得到的参数增量，λA\lambda_AλA 和 λB\lambda_BλB 为对应的融合权重。通过调整两组权重，可以控制人类偏好优化与图像感知质量优化对最终模型的影响程度，从而在图像美观度、语义一致性、主体清晰度以及画面复杂度之间取得更好的平衡。在实际应用中，本阶段通过不同权重组合下的定量指标与主观图像评测，选择综合效果较好的融合配置，并将其分别应用于步数蒸馏和参数蒸馏得到的学生模型。最终，在不改变学生模型主体结构及原有采样步数的条件下，利用两组奖励微调获得的 LoRA 对学生模型进行质量增强，作为本阶段最终交付模型的组成部分。

### 参考文献

\(1\) Shao Z, Wang P, Zhu Q, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models\(EB/OL\). arXiv:2402.03300, 2024. arxiv.org .

\(2\) Black K, Janner M, Du Y, et al. Training Diffusion Models with Reinforcement Learning\(C\). ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems, 2023. arxiv.org .

\(3\) Xue Z, Wu J, Gao Y, et al. DanceGRPO: Unleashing GRPO on Visual Generation\(EB/OL\). arXiv:2505.07818, 2025. arxiv.org .

\(4\) Liu J, Liu G, Liang J, et al. Flow-GRPO: Training Flow Matching Models via Online RL\(C\). Advances in Neural Information Processing Systems (NeurIPS), 2025. arxiv.org .

\(5\) He D, Feng G, Ge X, et al. Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models\(EB/OL\). arXiv:2511.16955, 2025. arxiv.org .

\(6\) Wu X, Hao Y, Sun K, et al. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis\(EB/OL\). arXiv:2306.09341, 2023. arxiv.org .

\(7\) Yang S, Wu T, Shi S, et al. MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment\(C\). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022. arxiv.org .

\(8\) Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models\(C\). International Conference on Learning Representations (ICLR), 2022. arxiv.org .

\(9\) Huang C, Liu Q, Lin B Y, et al. LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition\(C\). Conference on Language Modeling (COLM), 2024. arxiv.org .

\(10\) Shah V, Ruiz N, Cole F, et al. ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs\(C\). European Conference on Computer Vision (ECCV), 2024. arxiv.org .

\(11\) Zhong M, Shen Y, Wang S, et al. Multi-LoRA Composition for Image Generation\(EB/OL\). arXiv:2402.16843, 2024. arxiv.org .
