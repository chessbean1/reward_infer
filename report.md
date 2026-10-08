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


## 2.6.5 两步模型与教师模型微调

在二阶段的研究中，本项目主要针对三步蒸馏模型开展基于 GRPO 的强化学习微调，通过直接对学生模型进行训练获得 LoRA 参数，并将其用于改善低步数模型的生成质量。该方案在三步模型上取得了一定的效果，因此，在三阶段初期，本项目尝试将相同的技术路线推广至两步蒸馏模型，即直接以两步学生模型作为强化学习的训练对象，通过奖励函数引导模型进一步优化图像质量。然而，初步实验发现，与微调前的两步学生模型相比，直接进行 GRPO 微调后的模型不仅未能稳定改善生成效果，反而在部分提示词下出现了图像质量劣化、主体结构异常以及细节失真等问题，表明原本适用于三步学生模型的微调方案难以直接迁移至两步模型。

针对上述现象，本项目分析认为，其原因可能与两步蒸馏模型的采样特点及优化稳定性有关。相较于三步模型，两步模型需要在更少的采样次数内完成从初始噪声到目标图像的映射，单次采样所承担的生成任务更加复杂，模型对采样轨迹和参数扰动也可能更加敏感。由于蒸馏过程已经使模型形成了适用于特定时间步的生成映射，直接使用基于奖励的强化学习方法更新其参数，可能改变原有的采样轨迹，使模型在追求更高奖励的同时偏离蒸馏阶段学习到的生成分布。此外，两步模型缺少足够的中间去噪步骤来逐步修正生成偏差，因此强化学习带来的局部参数变化可能更容易在最终图像中表现为结构失真或质量下降。已有研究指出，直接对时间步蒸馏后的扩散模型采用常规微调目标，可能导致生成结果模糊或质量下降，需要针对少步生成过程设计专门的优化策略\(1\)。这一结论与本项目观察到的现象具有一定一致性，但两步模型直接进行 GRPO 微调时的具体劣化机制仍需进一步研究验证。

为解决直接微调两步模型带来的质量下降问题，本项目进一步探索了将强化学习训练与学生模型部署相分离的方案，即不再直接更新两步学生模型的参数，而是首先在原始教师模型上进行 GRPO 微调，得到对应的 LoRA 参数，再将其加载至蒸馏后的两步学生模型中。LoRA 通过学习低秩参数增量实现对预训练模型的高效适配，具有训练开销低、参数量小以及便于独立加载的特点\(2\)。考虑到本项目的学生模型由教师模型蒸馏得到，二者在模型架构和参数空间上具有一定的继承关系，因此，教师模型上学习得到的参数增量有可能在学生模型中保留部分优化效果。

实验结果表明，在本项目的模型设置下，直接对教师模型进行 GRPO 微调得到的 LoRA 能够有效迁移至两步学生模型。相比于直接对两步模型进行强化学习微调，该方案表现出更好的生成稳定性，在提升图像清晰度、改善局部细节和优化整体视觉效果的同时，能够更好地保持学生模型原有的主体结构和生成能力。分析认为，教师模型具有更加充分的采样过程和相对完整的生成能力，基于教师模型学习得到的奖励优化方向可能具有更好的稳定性和泛化能力，从而避免了直接优化两步模型时对其低步数生成轨迹造成过大扰动。需要指出的是，教师模型 LoRA 向学生模型的迁移效果与两者的架构兼容性及参数分布差异有关，不能认为该方法适用于任意教师—学生模型组合。

综合上述实验，本阶段最终放弃了直接对两步学生模型进行 GRPO 微调的方案，转而采用**“教师模型强化学习微调—LoRA 参数迁移—学生模型推理”**的技术路线。在三阶段后续实验中，所有用于增强学生模型的奖励微调 LoRA 均通过对教师模型训练获得，再以适当权重加载至两步学生模型。这一方案既避免了直接微调两步模型时出现的质量劣化问题，又实现了强化学习优化能力向低步数学生模型的有效迁移，同时保留了学生模型原有的推理加速优势。

## 2.6.6 不同奖励函数的探索

在三阶段初期的模型评测中发现，尽管经过蒸馏和强化学习微调后的学生模型能够在部分客观评价指标上达到较好的结果，但其生成图像仍存在一些难以通过现有指标准确反映的质量问题，主要表现为主体清晰度不足、局部纹理失真以及背景细节过多等。尤其在部分复杂场景下，学生模型倾向于生成大量与主体语义关联较弱的背景纹理和装饰性细节，导致画面整体结构杂乱、主体不够突出，影响实际视觉效果。前期采用的 HPSv2 奖励模型主要基于大规模人类偏好数据学习图像的整体偏好关系，在文生图模型评价方面具有较好的泛化能力\(3\)。然而，本项目的初步测试表明，HPSv2 对上述局部质量问题的识别能力存在一定局限，部分情况下甚至可能对包含大量冗余细节的图像给出较高评分，因此仅依靠 HPSv2 难以有效解决此类问题。

为寻找更加适合当前模型质量问题的奖励函数，本阶段进一步调研了多种图像质量与人类偏好评价模型，包括 HPSv2\(3\)、MANIQA\(4\)、PickScore_v1\(5\)、Q-Align\(6\) 以及 MPS\(7\)。其中，HPSv2 和 PickScore 均基于人类偏好数据进行训练，主要用于评估生成图像与人类整体偏好之间的一致性；MANIQA 是一种基于多维注意力机制的无参考图像质量评价模型，通过建模图像不同区域之间的空间与通道特征关系，预测图像的感知质量，在图像失真评价任务中具有较好的表现；Q-Align 则利用大型多模态模型，通过离散文本定义的质量等级学习与人类主观评价相一致的视觉质量评分；MPS（Multi-dimensional Preference Score）进一步将人类偏好划分为美学质量、语义一致性、细节质量以及整体偏好等多个维度，尝试实现更细粒度的文生图评价。上述方法分别从整体偏好、感知质量和多维度视觉评价等角度提供不同的优化信号，具有作为强化学习奖励函数的潜力。([arxiv.org][1], [openaccess.thecvf.com][2], [papers.nips.cc][3], [proceedings.mlr.press][4], [openaccess.thecvf.com][5])

在具体实验中，本项目选取初期主观评测中出现明显问题的提示词及其对应的教师模型、学生模型生成图像，分别使用上述奖励模型进行评分，并比较各模型对教师与学生图像的质量排序是否符合人工评测结论。实验结果表明，不同奖励函数对当前模型质量问题的识别能力存在明显差异。其中，MANIQA 对问题图像表现出了较好的区分能力，在所测试的典型问题样例中，教师模型生成图像的 MANIQA 评分高于对应学生模型，与主观评测中教师模型画面更加清晰、背景更加自然、局部失真更少的判断基本一致。这说明 MANIQA 能够在一定程度上反映学生模型存在的感知质量问题，为后续针对性优化提供有效的奖励信号。

相比之下，PickScore_v1、Q-Align 和 MPS 在本项目测试样例中的评分结果与人工评测结论未能形成稳定一致的对应关系。对于存在明显背景冗杂或不合理细节的图像，这些奖励模型在部分样例中认为教师模型更优，而在另外一些样例中则给予学生模型更高的评分，未能稳定地将背景过度复杂的图像判定为较低质量。分析认为，这可能与不同奖励模型的训练目标以及对图像质量的关注维度有关。PickScore 主要学习用户对生成图像的整体偏好，未针对背景冗余等特定失真问题进行显式建模\(5\)；Q-Align 虽然能够进行视觉质量评价，但其评分目标是学习通用场景下的主观质量等级，未必能够准确反映本项目所关注的特定生成缺陷\(6\)；MPS 虽然引入了细节质量等评价维度，但其目标仍然是预测多维度人类偏好，而细节丰富程度与细节合理性之间并不总是具有一致关系\(7\)。因此，在当前测试样例中，上述奖励模型未能表现出足够稳定的针对性区分能力。需要说明的是，该结论仅反映不同奖励模型在本项目特定问题图像上的表现，并不意味着这些方法在一般图像质量评价任务中缺乏有效性。

综合上述测试结果，本阶段最终选择 MANIQA 作为针对图像清晰度不足、局部失真以及背景过于复杂等问题的补充奖励函数，并将其引入基于 GRPO 的强化学习微调流程。具体而言，本项目以 MANIQA 评分作为强化学习的奖励信号，对教师模型进行独立训练，得到面向感知质量优化的 LoRA 参数，再将其加载至蒸馏后的学生模型。后续实验表明，采用 MANIQA 奖励训练得到的 LoRA 能够在一定程度上改善学生模型的图像清晰度，减少不必要的背景细节，使生成图像在主体突出程度、局部纹理合理性和整体视觉协调性方面得到改善。与此同时，考虑到 MANIQA 主要关注图像感知质量，而 HPSv2 更侧重整体人类偏好，本阶段进一步将两种奖励函数独立训练得到的 LoRA 进行加权融合，使两者在不同质量维度上的优化效果形成互补。最终，本项目采用 HPSv2 与 MANIQA 双奖励函数独立训练、双 LoRA 加权融合的方案，在保留模型整体生成能力的同时，针对性改善低步数学生模型中存在的图像失真和背景冗杂问题。

## 参考文献

\(1\) Miao Z, Yang Z, Lin K, et al. Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization\(C\). International Conference on Learning Representations (ICLR), 2025. [https\://arxiv.org/abs/2410.03190](https://arxiv.org/abs/2410.03190).

\(2\) Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models\(C\). International Conference on Learning Representations (ICLR), 2022. [https\://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685).

\(3\) Wu X, Hao Y, Sun K, et al. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis\(EB/OL\). arXiv:2306.09341, 2023. [https\://arxiv.org/abs/2306.09341](https://arxiv.org/abs/2306.09341).

\(4\) Yang S, Wu T, Shi S, et al. MANIQA: Multi-Dimension Attention Network for No-Reference Image Quality Assessment\(C\). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022: 1191–1200. [https\://arxiv.org/abs/2204.08958](https://arxiv.org/abs/2204.08958).

\(5\) Kirstain Y, Polyak A, Singer U, et al. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation\(C\). Advances in Neural Information Processing Systems (NeurIPS), 2023, 36. [https\://arxiv.org/abs/2305.01569](https://arxiv.org/abs/2305.01569).

\(6\) Wu H, Zhang Z, Zhang W, et al. Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels\(C\). Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235, 2024: 54015–54029. [https\://arxiv.org/abs/2312.17090](https://arxiv.org/abs/2312.17090).

\(7\) Zhang S, Wang B, Wu J, et al. Learning Multi-Dimensional Human Preference for Text-to-Image Generation\(C\). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024: 8018–8027. [https\://arxiv.org/abs/2405.14705](https://arxiv.org/abs/2405.14705).

[1]: https://arxiv.org/abs/2306.09341 "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis"
[2]: https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html "CVPR 2022 Open Access Repository"
[3]: https://papers.nips.cc/paper_files/paper/2023/hash/73aacd8b3b05b4b503d58310b523553c-Abstract-Conference.html "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation"
[4]: https://proceedings.mlr.press/v235/wu24ah.html "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels"
[5]: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_Multi-Dimensional_Human_Preference_for_Text-to-Image_Generation_CVPR_2024_paper.html "CVPR 2024 Open Access Repository"
