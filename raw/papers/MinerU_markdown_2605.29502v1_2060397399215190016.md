# Source-Grounded Semantic Reinforcement Learning for Low-Resource Target-Language Generation

Zeli Su1,2 Ziyin Zhang3 Zewei Pan3 Zhou Liu4 Dingcheng Huang5 Dehan Li6 

Zhankai Xu2 Longfei Zheng2 Xiaolu Zhang2 Jun Zhou2,† Wentao Zhang4,† 

1 Minzu University of China 2 Ant Group 3 Shanghai Jiao Tong University 

4 Peking University 5 Harbin Institute of Technology 6 South China University of Technology 

† Corresponding authors 

# Abstract

Low-resource target-language generation is often limited by scarce parallel data, while highresource source-language monolingual data is abundant but difficult to use with standard supervised fine-tuning. We propose Source-Grounded Semantic Reinforcement Learning (SG-SRL), a resource-utilization framework that converts source-language monolingual data into cross-lingual semantic supervision for target-language generation. SG-SRL performs reference-free reinforcement learning (RL) on source-language data using a crosslingual semantic reward model, instantiated by a cross-lingual reranker that scores the semantic relevance between the source input and the target-language generation. While this induces severe verbosity-based reward hacking, a lightweight recovery stage using a small parallel corpus restores fluency, conciseness, and task format while preserving the semantic gains. Experiments on Chinese-to-Thai generation show that SG-SRL improves semantic grounding and factual coverage over cold-start SFT. Additional analyses on long-form transfer and Tibetan embedding-based rewards clarify the generalization behavior of SG-SRL and show that an encoder-based semantic reward can substitute for an LLM-based reranker in a realistic low-resource language setting. 

# 1 Introduction

Large language models (LLMs) have achieved strong general-purpose generation abilities through large-scale pretraining and post-training (DeepSeek-AI, 2024; DeepSeek-AI et al., 2025; Yang et al., 2025), but their performance remains highly uneven across languages. High-resource languages benefit from abundant pretraining text, instruction data, and task-specific supervision, whereas low-resource languages often suffer from unstable generation, hallucinated content, poor factual grounding, and weak task adaptation (Üstün et al., 2024; Qin et al., 2025). Continual pretraining on target-language corpora followed by supervised fine-tuning (SFT) on downstream tasks has been employed for extending LLMs to new languages (Wang et al., 2020; Joshi et al., 2025), but it still relies on substantial target-language text or target-language task supervision, which is often unavailable for genuinely low-resource languages. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/26133579-0178-423b-85e5-0622f6c4a21b/21d5f41ca46474ecf9e3b8ae327c76b14e2ebc4ca6bbaf93e54ecbbff7ebd4fa.jpg)



Figure 1: Reranker-based source-grounded semantic reward. A source-language input and a generated targetlanguage output are treated as a cross-lingual query– candidate pair. The reranker estimates their semantic match and provides a scalar reward for RL without requiring a target-language reference.


This data bottleneck becomes even more severe in cross-lingual generation. For many emerging or weakly supported target languages, highquality parallel data is expensive to collect, and task-specific target-language annotations are even scarcer. In contrast, high-resource source-language data, such as Chinese or English news articles, is often abundant and reliable. Standard SFT cannot directly use such source-language monolingual data, because it requires a target-language reference for each training instance. This leads to the central question of this work: Can abundant sourcelanguage monolingual data be converted into useful supervision for low-resource target-language generation, without requiring target-language references during training? 

To this end, we need a supervision signal that can compare a source-language input with a targetlanguage generation without a gold target-language reference. Multilingual rerankers provide a practical approximation: they score the relevance between a query and a candidate text (Nogueira and Cho, 2019), and recent LLM-based variants can distinguish semantically aligned cross-lingual pairs from unrelated ones (Sun et al., 2023; Zhang et al., 2024, 2025). As illustrated in Figure 1, we treat the source input as the query and the generated targetlanguage output as the candidate, and use a reranker as a source-grounded semantic reward. This converts source-language monolingual data into scalable RL supervision without target-language references. 

Empirically, direct optimization of this reward exposes a semantic–form trade-off. The relevanceoriented reward encourages the policy to generate longer outputs that cover more source-side concepts, leading to verbosity-based reward hacking: the intermediate RL checkpoint improves semantic coverage but becomes verbose, poorly formatted, and less fluent (Amodei et al., 2016). Stronger length or format constraints can reduce verbosity, but may also suppress useful source-totarget semantic learning. We therefore treat sourcegrounded semantic RL as semantic mid-training rather than final optimization. 

Based on this insight, we propose Source-Grounded Semantic Reinforcement Learning (SG-SRL), a train–reinforce–recover framework for low-resource target-language generation. SG-SRL first learns target-language form from a small parallel corpus, then reinforces sourcegrounded semantics on source-language monolingual data, and finally reuses the parallel corpus to recover fluent and concise target-language generation. On Chinese-to-Thai generation, starting from SmolLM3-3B (Bakouch et al., 2025), a model with weak Thai support, SG-SRL substantially improves over cold-start SFT, showing that semantic RL as mid-training can strengthen new-language semantic alignment. We further examine a Tibetan setting, where a strong LLM-based reranker is not available for the target language. By training an encoder-based Tibetan–Chinese embedding scorer, we demonstrate that SG-SRL can generalize to a realistic low-resource language scenario with a different form of relevance reward. 

Our contributions are summarized as follows: 

• We propose SG-SRL, a train–reinforce– recover framework that converts sourcelanguage monolingual data into semantic supervision to enable reference-free RL in the low-resource target-language generation setting. 

• We instantiate SG-SRL with a multilingual reranker reward, and identify a semantic– form trade-off: direct optimization induces verbosity-based reward hacking, but the intermediate checkpoint can still learn useful cross-lingual semantic alignment. 

• We show on Chinese-to-Thai generation with SmolLM3-3B that semantic RL as mid-training improves low-resource targetlanguage generation over cold-start SFT, and we further verify language generalization in a Tibetan setting where an encoder-based embedding reward replaces the LLM-based reranker. 

# 2 Related Work

Low-resource language expansion. Expanding language models to low-resource languages has commonly been approached through multilingual pretraining (Conneau et al., 2020; Xue et al., 2021), continued pretraining (Wang et al., 2020; Joshi et al., 2025), or supervised fine-tuning on targetlanguage data. However, these approaches still depend on the availability of target-language text or task-specific supervision. In genuinely lowresource settings, both pretraining corpora and high-quality downstream parallel data may be limited, motivating methods that can exploit abundant source-language data without requiring targetlanguage references for every instance. 

Semantic rewards for low-resource language learning. Recent work has explored reinforcement learning with semantic rewards as an alternative to token-level likelihood optimization for low-resource language expansion. Su et al. (2026) propose using embedding-level semantic rewards with GRPO to improve low-resource language capabilities while reducing alignment tax, showing that semantic-space optimization can preserve general capabilities better than conventional SFT. Our work shares the same motivation, and further extends the horizon to a reference-free, cross-lingual generation setting. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/26133579-0178-423b-85e5-0622f6c4a21b/186c44b5eb9437804977f0ed66a4f9aab5ac77907b5d45635ab3d5066dd643c0.jpg)



Figure 2: Overview of SG-SRL. The framework uses a small parallel corpus $\mathcal { D } _ { \mathrm { p a r a } }$ for target-form initialization and target-form recovery, and a large source-language monolingual corpus $\mathcal { D } _ { \mathrm { s r c } }$ for source-grounded semantic RL. The RL stage uses a multilingual reranker as a weak semantic reward model, with language, length, and repetition safeguards. A final recovery stage reuses $\mathcal { D } _ { \mathrm { p a r a } }$ to restore fluency, conciseness, and task format.


Cross-lingual semantic matching and LLMbased reranking. Dense multilingual representations and cross-lingual retrieval models provide a basis for measuring semantic correspondence across languages (Feng et al., 2022; Bonifacio et al., 2021). Recent LLM-based rerankers further formulate relevance estimation as an instructionfollowing judgment over a query–candidate pair, often deriving a score from the model’s preference for an affirmative answer (Sun et al., 2023; Zhang et al., 2024, 2025). We adopt this formulation by treating the source-language input as the query and the generated target-language output as the candidate. Unlike standard retrieval, however, we use the reranker’s yes/no relevance probability as an online reward for policy optimization. This turns reranker from an offline ranking module into an optimization target, which exposes relevance-oriented reward exploitation such as verbosity-based reward hacking. 

Reinforcement learning and reward hacking. Reinforcement learning has become a central tool for adapting language models to human preferences and task-specific objectives (Ouyang et al., 2022; Shao et al., 2024). However, optimizing imperfect proxy rewards can lead to reward hacking, where models satisfy the literal reward function while violating the intended behavior (Amodei et al., 2016). In our setting, the reranker reward is semantically informative but relevance-oriented: longer generations can cover more source-side information and thus receive higher scores, even when they degrade fluency, conciseness, or task format. Rather than treating this as a failure of RL alone, we show that the hacked intermediate policy can still acquire useful cross-lingual semantic alignment. 

# 3 Method

SG-SRL uses two data sources for different pure poses: a small parallel corpus $\mathcal { D } _ { \mathrm { p a r a } }$ for targetform initialization and recovery, and a large sourcelanguage monolingual corpus $\mathcal { D } _ { \mathrm { s r c } }$ for semantic mid-training. As shown in Figure 2, SG-SRL follows a train–reinforce–recover procedure: train target-language form from $\mathcal { D } _ { \mathrm { p a r a } }$ , reinforce sourcegrounded semantics from $\mathcal { D } _ { \mathrm { s r c } } .$ , and recover targetlanguage form again with $\mathcal { D } _ { \mathrm { p a r a } } .$ . 

# 3.1 Problem Formulation

We consider a high-resource source language X and a low-resource or weakly supported target language Y. We are given a small parallel corpus 

$$
\mathcal {D} _ {\text { para }} = \{(x _ {i}, y _ {i}) \} _ {i = 1} ^ {N}, \tag {1}
$$

where $x _ { i } ~ \in ~ { \mathcal { X } }$ and $y _ { i } ~ \in ~ \mathcal { V }$ , and a much larger source-language monolingual corpus 

$$
\mathcal {D} _ {\mathrm{src}} = \left\{x _ {j} \right\} _ {j = 1} ^ {M}, \quad M \gg N, \tag {2}
$$

which contains no target-language references. 

The goal is to train a policy model $\pi _ { \theta }$ that generates a target-language output $\hat { y } \in \mathcal { V }$ conditioned on a source-language input $x \in \mathcal { X }$ . Standard supervised fine-tuning only uses ${ \mathcal { D } } _ { \mathrm { p a r a } }$ . SG-SRL converts $\mathcal { D } _ { \mathrm { s r c } }$ into RL training data by using a cross-lingual semantic scorer as a weak reward model. 

# 3.2 SG-SRL Overview

SG-SRL has three stages: 

1. Target-form initialization: fine-tune the base model on $\mathcal { D } _ { \mathrm { p a r a } }$ for 3 epochs to obtain a coldstart target-language generator. 

2. Source-grounded semantic RL: optimize the initialized model on $\mathcal { D } _ { \mathrm { s r c } }$ for 2 epochs using a reranker-based semantic reward. 

3. Target-form recovery: fine-tune the RL checkpoint again on $\mathcal { D } _ { \mathrm { { p a r a } } }$ for 1 epoch to restore fluency, length control, and task format. 

The design separates target-language form learning from source-grounded semantic learning. The small parallel corpus gives reliable but limited form supervision, while the source-language monolingual corpus provides broader semantic coverage without target-language references. 

# 3.3 Target-form Initialization

We first perform supervised fine-tuning on $\mathcal { D } _ { \mathrm { p a r a } } \mathrm { : }$ 

$$
\theta_ {\mathrm{sft}} = \arg \min _ {\theta} - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {\text { para }}} \log \pi_ {\theta} (y \mid x). \tag {3}
$$

This stage teaches the model basic target-language form, task format, and local generation style. However, because $\mathcal { D } _ { \mathrm { p a r a } }$ is small, the resulting cold-start model may still miss source-side entities, events, or factual relations. The next stage therefore uses $\mathcal { D } _ { \mathrm { s r c } }$ to inject additional semantic grounding. 

# 3.4 Source-grounded Semantic RL

Starting from $\pi _ { \theta _ { \mathrm { s f t } } } ,$ we perform RL on $\mathcal { D } _ { \mathrm { s r c } }$ . For each source input $x \sim \mathcal { D } _ { \mathrm { s r c } }$ , the policy samples 

$$
\hat {y} \sim \pi_ {\theta} (\cdot | x). \tag {4}
$$

Since no target-language reference is available, we use the reranker format in Figure 1: the source input is treated as the query, and the generated target-language output is treated as the candidate document. 

Reranker reward. In our implementation, the reranker is a generative yes/no judge. Given an instruction I, source input x, and generated output $\hat { y } ,$ we define the semantic reward as the normalized probability of the affirmative answer: 

$$
z _ {\phi} ^ {+} = z _ {\phi} (\mathrm{yes} \mid I, x, \hat {y}),
$$

$$
z _ {\phi} ^ {-} = z _ {\phi} (\mathrm{no} \mid I, x, \hat {y}), \tag {5}
$$

$$
r _ {\mathrm{rank}} (x, \hat {y}) = \frac {\exp z _ {\phi} ^ {+}}{\exp z _ {\phi} ^ {+} + \exp z _ {\phi} ^ {-}}.
$$

This score lies in [0, 1] and provides the main source-grounded semantic signal. 

Reward safeguards. To reduce obvious degeneration, we combine the reranker score with the three safeguards shown in Figure 2: a hard language gate, a batch-relative length penalty, and a repetition penalty. For a candidate $\hat { y } _ { i }$ in reward batch B, the final reward is 

$$
s _ {i} = r _ {i} ^ {\text {rank}} - \lambda_ {\text {len}} p _ {i} ^ {\text {len}} - \lambda_ {\text {rep}} p _ {i} ^ {\text {rep}}, \tag {6}
$$

$$
r _ {i} = g _ {i} \max (s _ {i}, 0).
$$

Here, $g _ { i }$ is a hard target-language gate. In the Chinese–Thai setting, it requires the output to be predominantly Thai, contain no Chinese characters, and contain little Latin-script text. If the gate fails, the reward is set to zero. 

The length penalty $p _ { i } ^ { \mathrm { l e n } }$ is computed with the reranker tokenizer and is relative to the current reward batch. It starts only when an output exceeds 2.5 times the batch median length and increases linearly until 5.0 times the median length. The repetition penalty $p _ { i } ^ { \mathrm { r e p } }$ is based on repeated 4-grams and is activated when the repeated 4-gram ratio exceeds 0.15. We set $\lambda _ { \mathrm { l e n } } ~ = ~ 0 . 2 0$ and $\lambda _ { \mathrm { { r e p } } } ~ =$ 0.05 in the main configuration. Thus, semantic alignment is the only graded positive signal, while language validity, length control, and repetition control act as safeguards. 

Policy optimization. We optimize the policy with GRPO (Shao et al., 2024). For each source input, we sample a group of candidate outputs, compute their rewards, normalize rewards within the group, and update the policy according to relative advantages. The reference policy is initialized from $\pi _ { \theta _ { \mathrm { s f t } } }$ to reduce drift from the cold-start targetlanguage behavior. 

We use this RL stage as semantic mid-training rather than final optimization. Directly optimizing the semantic reward can improve source-grounded coverage, but may also produce verbose or poorly formatted outputs. The purpose of this stage is therefore to learn source-to-target semantic alignment from $\mathcal { D } _ { \mathrm { s r c } }$ , not to produce the final deployable generator. 

# 3.5 Target-form Recovery

After semantic RL, we obtain an intermediate policy $\pi _ { \theta _ { \mathrm { r l } } }$ that has absorbed source-side semantic supervision but may have degraded surface form. We then reuse $\mathcal { D } _ { \mathrm { { p a r a } } }$ for 1 epoch of lightweight supervised recovery: 

$$
\theta_ {\mathrm{sg-srl}} = \arg \min _ {\theta} \mathcal {L} _ {\text {rec}} (\theta), \quad \theta \leftarrow \theta_ {\mathrm{rl}}, \tag {7}
$$

$$
\mathcal {L} _ {\mathrm{rec}} (\theta) = - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {\mathrm{para}}} \log \pi_ {\theta} (y \mid x).
$$

Although initialization and recovery use the same parallel corpus, they serve different roles. Initialization teaches basic target-language generation from the base model, while recovery regularizes a semantically enhanced but form-degraded RL checkpoint. The final model keeps the semantic gains from source-grounded RL while restoring fluency, conciseness, and task format. 

# 4 Experiments

We conduct a series of experiments to evaluate whether SG-SRL can use source-language monolingual data to improve low-resource target-language generation beyond cold-start SFT. The experiments are designed to answer four questions: (1) whether the full train–reinforce–recover pipeline improves target-language generation over SFT on a small parallel corpus, (2) why the intermediate semantic RL checkpoint should be treated as mid-training rather than the final model, (3) whether the learned sourcegrounded semantics transfer beyond the original title-generation format, and (4) whether the framework can generalize to a realistic low-resource language setting by replacing the LLM-based reranker with an encoder-based embedding reward when no strong target-language reranker is available. 

# 4.1 Experiment 1: Effectiveness of SG-SRL

We first evaluate whether the full train–reinforce– recover pipeline improves target-language generation beyond cold-start SFT, testing the central claim that source-language monolingual data can provide useful semantic supervision when it is converted into a reference-free cross-lingual reward. 

Task and data. The main task is Chinese-to-Thai news-title generation. We use CNewSum, a largescale Chinese summarization dataset with humanannotated adequacy and deducibility levels (Wang et al., 2021), to construct a low-resource Chineseto-Thai setting. We translate 15k Chinese titles into Thai. Among them, 10k parallel examples are used for target-form initialization and targetform recovery, and 5k examples are held out as the development set. We additionally sample 100k Chinese-only examples from CNewSum as sourcelanguage monolingual data for source-grounded semantic RL. The three splits are each deduplicated and mutually disjoint. This setting matches the target scenario of SG-SRL: a small amount of targetlanguage supervision is available for learning output form, while substantially more source-language data can be used for semantic mid-training. 

Base model and training stages. All main Chinese-to-Thai experiments use SmolLM3- 3B (Bakouch et al., 2025) as the base model. The SG-SRL pipeline has three stages as described in Section 3. First, we train a cold-start supervised model on the 10k Chinese–Thai parallel examples. Second, we perform source-grounded semantic RL on the 100k Chinese-only examples. Third, we apply target-form recovery by fine-tuning the RL checkpoint for one epoch on the same 10k parallel examples. We refer to the supervised model after the first stage as COLD-START SFT, and to the final recovered model as SG-SRL. 

Evaluation protocol. For the main Chinese-to-Thai generation task, we use an LLM-judge protocol with deepseek-v4-flash. This choice reflects the metric-sensitivity issue of semanticreward RL: source grounding, hallucination avoidance, and meaning-preserving rephrasings can be under-measured by surface-overlap or embeddingsimilarity metrics (Su et al., 2026); Appendix A.1 provides further discussion and explains why the reranker reward is not used as the main metric. The judge compares the gold Thai reference, SG-SRL, and COLD-START SFT in terms of semantic adequacy, factual faithfulness, Thai fluency, and title-style conciseness. We report both three-way ranking statistics and direct pairwise win rates. For intermediate RL checkpoints, we additionally evaluate entity alignment, event alignment, factual consistency, fluency, and conciseness/format control, together with output length statistics to quantify verbosity. 

Results. Table 1a reports the three-way LLMjudge ranking among the gold Thai reference, SG-SRL, and COLD-START SFT on the 5k CNewSum development examples. The total score assigns +2 points to Rank 1, +1 point to Rank 2, and - 1 point to Rank 3. The gold reference remains the strongest candidate, showing that the task is still challenging. However, SG-SRL substantially improves over COLD-START SFT: it is ranked first in 995 examples and second in 2602 examples, while COLD-START SFT is ranked last in 3185 examples. 

The direct pairwise comparison in Table 1b gives a more intuitive model-to-model comparison: SG-SRL wins against COLD-START SFT in 72.3% of examples, with very few ties. Together, these results indicate that SFT on a small parallel corpus can teach the model to produce Thai-form outputs, but it does not provide sufficient coverage for robust source-grounded generation. By contrast, SG-SRL uses the larger Chinese-only corpus to learn additional semantic grounding and then restores target-language form through recovery. 

Analysis. The improvement is strongest in preference-based evaluation rather than in a narrow reference-matching setting, which is consistent with the goal of SG-SRL. The three-way comparison shows that the recovered RL model can sometimes compete even against the gold reference, while the pairwise comparison directly confirms its advantage over the supervised baseline. The method does not merely imitate a small set of Thai references; it uses Chinese-only inputs to strengthen source-grounded semantic coverage. The result supports the first part of our claim: semantic rewards can turn source-language monolingual data into effective supervision for lowresource target-language generation. 

# 4.2 Experiment 2: Semantic Learning versus Reward Hacking

We next analyze the intermediate RL checkpoints before target-form recovery. This experiment asks why the semantic RL stage should be viewed as mid-training rather than as the final deployable generator. All variants start from the same COLD-START SFT checkpoint and optimize on the same Chinese-only source corpus, but differ in reward design or RL control. 

We compare four variants at a high level here, leaving the exact reward definitions and ablation rationale to Appendix A.2. GATE+BATCHLEN is the main SG-SRL reward configuration, combining a reranker semantic score with Thai-language gating, batch-relative length control, and repetition control. GATE-ONLY removes the explicit length and repetition safeguards. REFLEN replaces batch-relative length control with source-conditioned absolute length control. Observing that recent analyses RL training suggest that improvements and failures can arise from the RL algorithm itself (Liu et al., 2025), we introduce another setting GRPO-CONTROL, where GRPO is replaced with Dr.GRPO (Liu et al., 2025) to examine whether verbosity and form degradation are mainly artifacts of the GRPO-style optimization, or the relevance-oriented reward. 

Results. Table 2 shows a clear semantic–form trade-off. GATE+BATCHLEN achieves the best overall score and the strongest entity, event, and factual alignment, indicating that source-grounded semantic RL can inject useful cross-lingual grounding into the model. However, its average output length is 531.7 characters, far longer than the gold Thai references, whose average length is 162.1 characters. Thus, even the best intermediate RL checkpoint is not directly deployable. 

Analysis. The comparison with GATE-ONLY shows why auxiliary regularization is necessary. When the reward contains only the reranker score and the language gate, the model exploits the relevance reward by generating much longer outputs (average length 866.6 characters). REFLEN shows the opposite failure mode. It reduces average output length to 179.7 characters, close to the gold reference length, but its entity, event, factual, and fluency scores drop substantially, suggesting that aggressive absolute length control can suppress verbosity but may also restrict the model’s ability to explore and express source-side semantics in the 

<table><tr><td>Model</td><td>Total Score</td><td>Avg. Score</td><td>Rank 1</td><td>Rank 2</td><td>Rank 3</td></tr><tr><td>Gold</td><td>7899</td><td>1.58</td><td>3723</td><td>865</td><td>412</td></tr><tr><td>SG-SRL</td><td>3189</td><td>0.64</td><td>995</td><td>2602</td><td>1403</td></tr><tr><td>COLD-START SFT</td><td>-1088</td><td>-0.22</td><td>282</td><td>1533</td><td>3185</td></tr></table>


(a) Three-way ranking.


<table><tr><td>Outcome</td><td>Count</td><td>Ratio</td></tr><tr><td>SG-SRL wins</td><td>3613</td><td>72.3%</td></tr><tr><td>COLD-START SFT wins</td><td>1378</td><td>27.6%</td></tr><tr><td>Tie</td><td>9</td><td>0.2%</td></tr></table>


(b) Pairwise win rate.



Table 1: Main effectiveness comparison on Chinese-to-Thai title generation.


Left: three-way LLM-judge ranking among the gold reference, SG-SRL, and COLD-START SFT; right: direct pairwise comparison between the two models. The exact judge prompts used for the three-way ranking and pairwise comparison are provided in Appendix A.3. 

<table><tr><td>Checkpoint</td><td>Entity</td><td>Event</td><td>Factual</td><td>Fluency</td><td>Conciseness</td><td>Avg.</td><td>Mean Length</td></tr><tr><td>GATE+BATCHLEN</td><td>2.331</td><td>2.887</td><td>2.363</td><td>2.194</td><td>1.738</td><td>2.303</td><td>531.7</td></tr><tr><td>GATE-ONLY</td><td>1.777</td><td>2.166</td><td>1.776</td><td>1.583</td><td>1.219</td><td>1.704</td><td>866.6</td></tr><tr><td>REFLEN</td><td>1.401</td><td>1.638</td><td>1.393</td><td>1.194</td><td>1.595</td><td>1.444</td><td>179.7</td></tr><tr><td>GRPO-CONTROL</td><td>1.441</td><td>1.549</td><td>1.428</td><td>1.000</td><td>0.976</td><td>1.279</td><td>708.9</td></tr></table>


Table 2: Semantic–form trade-off in intermediate RL checkpoints. GATE+BATCHLEN achieves the strongest semantic alignment before recovery, but its outputs remain much longer than the gold Thai references, whose mean length is 162.1 characters. GATE-ONLY suffers from severe verbosity, while REFLEN controls length but weakens semantic learning. A qualitative case study is provided in Appendix A.4, and the five-dimensional judge prompt is provided in Table 5.


target language. 

The standalone GRPO-CONTROL comparison further clarifies the cause of the failure. It obtains the lowest overall score while still producing long outputs, so the observed degeneration is not resolved by changing the RL control variant. The main issue is therefore not simply the GRPO-style optimizer, but the relevance-oriented reward structure itself. 

These findings motivate the train–reinforce– recover design. The RL checkpoint is valuable because it learns source-grounded semantic alignment, but it should not be used as the final generator. Target-form recovery is needed to convert the semantically enhanced but form-degraded checkpoint into a usable target-language model. 

# 4.3 Experiment 3: Transfer Beyond Title Generation

We further test whether the semantic gains from SG-SRL transfer beyond the original title-generation format. This experiment asks whether SG-SRL learns reusable source-grounded semantics or only improves the specific training format. 

Task and data. The transfer task is long-form Chinese-to-Thai translation, which requires preserving more entities, events, and factual relations over a longer context than title generation. We construct a 100-example long-form translation set with Chinese inputs longer than 500 characters. The data is constructed with GPT-5.5 assistance and manually checked. We also evaluate short-title translation, but report it in Appendix A.5 because the three models obtain very similar BLEU scores, making that setting less discriminative for semantic grounding. 

Results. Table 3 shows that SG-SRL improves over both the base model and COLD-START SFT in BLEU. More importantly, the LLM-judge results show a clearer advantage across semantic dimensions. SG-SRL achieves the highest entity alignment, event alignment, factual consistency, and completeness while maintaining Thai fluency. 

Analysis. The transfer results clarify the type of improvement learned by SG-SRL. The benefit is not a uniform gain on short translation instances; rather, it becomes more visible when the input requires longer-context semantic preservation and robust cross-lingual grounding. This supports the interpretation that source-grounded semantic RL improves reusable semantic alignment instead of only memorizing the title-generation format. 

# 4.4 Experiment 4: Language Generalization to Tibetan

Finally, we test whether SG-SRL can be instantiated in a target language that better matches a truly low-resource setting. Tibetan is a useful test case because it lacks a reliable LLM backbone that can be directly used as a cross-lingual reranker, unlike the Thai experiments where a stronger multilingual reranker is available. However, Tibetan is covered by CINO, a Chinese minority-language encoder model (Yang et al., 2022). We therefore replace the LLM-based reranker with a CINO-based Tibetan– 

<table><tr><td>Model</td><td>BLEU</td><td>Entity</td><td>Event</td><td>Factual</td><td>Thai Fluency</td><td>Completeness</td><td>Avg.</td></tr><tr><td>Base</td><td>0.11</td><td>2.600</td><td>2.800</td><td>2.720</td><td>2.420</td><td>2.680</td><td>2.644</td></tr><tr><td>COLD-START SFT</td><td>0.14</td><td>2.700</td><td>2.700</td><td>2.680</td><td>3.240</td><td>2.180</td><td>2.700</td></tr><tr><td>SG-SRL</td><td>0.17</td><td>3.420</td><td>3.540</td><td>3.520</td><td>3.280</td><td>3.260</td><td>3.404</td></tr></table>


Table 3: Transfer to long-form Chinese-to-Thai translation. BLEU (Papineni et al., 2002) provides a surface-level automatic metric, while the LLM-judge dimensions evaluate semantic preservation and Thai quality. SG-SRL improves over both the base model and COLD-START SFT, with the largest gains on entity alignment, event alignment, factual consistency, and completeness. The long-text translation judge prompt is provided in Table 6.


<table><tr><td>Setting</td><td>BLEU</td><td>Embedding Similarity</td></tr><tr><td>RL w/ Bo reference</td><td>0.4519</td><td>0.7164</td></tr><tr><td>RL w/ Ch reference</td><td>0.4523</td><td>0.7011</td></tr></table>


Table 4: Language generalization with embeddingbased Tibetan rewards. Using either Tibetan or Chinese as the semantic reference leads to comparable BLEU scores, suggesting that the trained encoder-based embedding model can provide an in-domain crosslingual relevance signal when an LLM-based reranker is not available.


Chinese embedding model and use its similarity score as the semantic reward. 

Task and data. The task is Chinese-to-Tibetan generation with an embedding-based semantic reward. The data comes from the VLM portion of FTibSuite, a resource suite for Tibetan vision– language modeling (Anonymous, 2026). This setting is fully in-domain: all data are drawn from a 100k-example Chinese–Tibetan parallel caption corpus, and the same corpus is used to train the embedding scorer and to construct the RL task. We split the corpus into 10k examples for cold-start supervised fine-tuning, 10k examples for development evaluation, and 80k examples for RL training. 

The embedding model is trained on the Tibetan– Chinese caption pairs with a contrastive objective, mapping matched pairs closer in representation space and pushing mismatched pairs apart. Unlike a cross-encoder reranker, this model returns a similarity score in a shared embedding space. We compare two RL reward settings: one uses Tibetan text as the semantic reference, and the other uses Chinese text as the semantic reference. The comparison tests whether the learned embedding space can provide a usable cross-lingual reward when the semantic anchor is placed on either side of the bilingual pair. 

Results. As shown in Table 4, the two reference choices lead to similar BLEU scores. This result shows that the SG-SRL paradigm can still work when the reward module is implemented by an encoder-based embedding model rather than an LLM-based reranker. The Chinese-reference reward is especially relevant to SG-SRL, because it shows that target-language generation can be guided by a semantic signal anchored in the source language even when the target language lacks a strong reranker backbone. 

Analysis. The Tibetan experiment should be interpreted together with the embedding-reward comparison summarized in Appendix A.6. That appendix replaces the reranker reward in the Thai setting with Qwen3-8B-Embedding similarity and shows a clear trade-off: the embedding reward produces much shorter outputs and is less prone to length hacking, but its semantic supervision is weaker. In particular, candidate scores concentrate in a narrow 55–70 range, whereas the reranker separates good and bad generations more clearly and yields stronger post-recovery performance. Thus, generic embedding rewards are not a drop-in replacement for rerankers. 

The Tibetan setting studies the complementary case where a strong reranker is unavailable, which is often the realistic constraint for genuinely lowresource languages. The CINO-based reward is effective because it is trained on the same Tibetan– Chinese caption domain used for RL, giving it sufficient in-domain resolution to distinguish matched from mismatched pairs. The comparable results with Tibetan and Chinese references further suggest that this learned cross-lingual space can support either target-anchored or source-anchored rewards. Therefore, Table 4 shows that SG-SRL can be adapted with an in-domain encoder-based reward when no strong reranker exists, rather than claiming that embedding rewards generally replace rerankers or generalize broadly out of domain. 

# 5 Conclusion

We presented SG-SRL, a novel train-reinforcerecover framework that enables reference-free reinforcement learning via cross-lingual semantic rewards to overcome the data bottleneck in lowresource target-language generation. Crucially, our decoupled approach isolates source-grounded semantic learning from target-form regularization, effectively neutralizing the verbosity-based reward hacking inherent in relevance optimization. Our evaluations across Chinese-to-Thai and Tibetan tasks, spanning both cross-lingual summarization and long-form translation, confirm that SG-SRL significantly enhances semantic grounding and factual consistency over standard SFT. This work establishes a robust pipeline for aligning weak target languages using high-resource source anchors, paving the way for more equitable multilingual model expansion. 

# Limitations

A main limitation of this work is that SG-SRL is not yet fully matched to the most ideal lowresource setting. In principle, the framework benefits from a strong cross-lingual reranker that can directly score the semantic match between a sourcelanguage input and a target-language generation. However, such rerankers are usually unavailable for genuinely low-resource target languages. Therefore, while our Chinese-to-Thai experiments use a reranker-based reward, the Tibetan experiment uses an encoder-based embedding reward as a practical substitute. This shows that the framework can be adapted to a more realistic low-resource scenario, but it also means that the low-resource experiment does not exactly replicate the full reranker-based SG-SRL setting. Future work can further improve this part by building stronger reward models for low-resource languages or by designing reward functions that rely less on high-quality multilingual rerankers. 

# References



Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Man’e. 2016. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565. 





Anonymous. 2026. FTibsuite: A comprehensive resource suite for tibetan vision–language modeling. In Submitted to ACL Rolling Review - January 2026. Under review. 





Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Patiño, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallouédec, Kashif Rasul, Nathan Habib, Clémentine Fourrier, Hynek Kydlicek, Guilherme Penedo, Hugo Larcher, Mathieu Morlon, Vaibhav Srivastav, Joshua Lochner, and 4 others. 2025. SmolLM3: smol, multilingual, long-context 





reasoner. https://huggingface.co/blog/ smollm3. 





Luiz Bonifacio, Vitor Jeronymo, Hugo Queiroz Abonizio, Israel Campiotti, Marzieh Fadaee, Roberto Lotufo, and Rodrigo Nogueira. 2021. mMARCO: A multilingual version of the MS MARCO passage ranking dataset. arXiv preprint arXiv:2108.13897. 





Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm’an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics. 





DeepSeek-AI. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437. 





DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Nature, 645:633– 638. 





Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Language-agnostic BERT sentence embedding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, pages 878–891, Dublin, Ireland. Association for Computational Linguistics. 





Raviraj Joshi, Kanishk Singla, Anusha Kamath, Raunak Kalani, Rakesh Paul, Utkarsh Vaidya, Sanjay Singh Chauhan, Niranjan Wartikar, and Eileen Long. 2025. Adapting multilingual LLMs to low-resource languages using continued pre-training and synthetic corpus: A case study for Hindi LLMs. In Proceedings of the First Workshop on Natural Language Processing for Indo-Aryan and Dravidian Languages, pages 50–57, Abu Dhabi. Association for Computational Linguistics. 





Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025. Understanding r1-zero-like training: A critical perspective. In Conference on Language Modeling (COLM). 





Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085. 





Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. 





Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318. 





Libo Qin and 1 others. 2025. A survey of multilingual large language models. Patterns. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. 





Zeli Su, Ziyin Zhang, Zhou Liu, Xuexian Song, Zhankai Xu, Longfei Zheng, Xiaolu Zhang, Rong Fu, Guixian Xu, and Wentao Zhang. 2026. Reinforcement learning with semantic rewards enables low-resource language expansion without alignment tax. arXiv preprint arXiv:2605.14366. 





Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT good at search? investigating large language models as re-ranking agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 





Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024. Aya model: An instruction finetuned open-access multilingual language model. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 





Danqing Wang, Jiaze Chen, Xianze Wu, Hao Zhou, and Lei Li. 2021. Cnewsum: A large-scale summarization dataset with human-annotated adequacy and deducibility level. In Natural Language Processing and Chinese Computing, pages 389–400, Cham. Springer International Publishing. 





Zihan Wang, Karthikeyan K, Stephen Mayhew, and Dan Roth. 2020. Extending multilingual BERT to lowresource languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2649–2656, Online. Association for Computational Linguistics. 





Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics. 





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, 





Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 2 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388. 





Ziqing Yang, Zihang Xu, Yiming Cui, Baoxin Wang, Min Lin, Dayong Wu, and Zhigang Chen. 2022. CINO: A Chinese minority pre-trained language model. In Proceedings of the 29th International Conference on Computational Linguistics, pages 3937– 3949, Gyeongju, Republic of Korea. International Committee on Computational Linguistics. 





Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. mGTE: Generalized longcontext text representation and reranking models for multilingual text retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. 





Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176. 



# A Additional Details

# A.1 Metric Sensitivity

Semantic-reward RL should not be evaluated only with the same reward model used for training. In our setting, a reranker reward is useful as a training signal because it can compare a Chinese source input with a Thai generation without a gold Thai reference. However, using that same reranker as the main evaluation metric would risk overestimating models that exploit the reward by producing overly long outputs. We therefore report LLM-judge preference results for the main Chinese-to-Thai task and use BLEU only as a supplementary surfaceoverlap metric in transfer experiments. 

# A.2 Reward Details for RL Diagnostics

The intermediate-checkpoint comparison in Section 4.2 uses four reward/control variants. GATE+BATCHLEN is the main SG-SRL configuration: it combines the reranker relevance score with a Thai-language gate, a batch-relative length penalty, and a repeated 4-gram penalty. GATE-ONLY keeps the reranker score and Thai-language gate but removes explicit length and repetition penalties. REFLEN replaces batch-relative length control with a source-conditioned absolute length constraint. GRPO-CONTROL keeps the main reward form but changes the RL control variant to test whether degeneration is mainly caused by the optimization recipe rather than the relevance-oriented reward. 

# A.3 Judge Prompts

For the main title-generation evaluation, the judge receives the Chinese source, the gold Thai title, and model outputs in randomized order. It ranks outputs by semantic adequacy, factual faithfulness, Thai fluency, and title-style conciseness. For pairwise evaluation, the judge compares SG-SRL and COLD-START SFT directly and may return a tie only when the two outputs are indistinguishable in quality. 

# A.4 Qualitative RL Diagnostic Case

The qualitative cases follow the same pattern as the aggregate results in Table 2. GATE-ONLY outputs often mention more source-side content but become excessively long and sometimes drift away from title format. REFLEN better controls length but frequently drops entities or event details. GATE+BATCHLEN provides the best compromise among the intermediate RL checkpoints, although it still requires target-form recovery before deployment. 

# A.5 Short-Title Translation

We also evaluated a short-title translation setting. The task is less discriminative than long-form transfer because the source inputs contain fewer entities and event relations, and the compared models obtain similar surface-overlap scores. We therefore report the more informative long-form transfer results in the main text, where semantic preservation over longer contexts better reveals the effect of source-grounded semantic RL. 

# A.6 Embedding Reward Comparison

As an alternative to the reranker reward, we tested a generic Qwen3-8B-Embedding similarity reward in the Thai setting. This reward is less prone to severe length hacking because longer outputs do not necessarily increase cosine similarity to a fixed semantic reference. However, it provides weaker score discrimination: candidate scores concentrate in a narrow range, whereas the reranker more clearly separates good and bad generations. This motivates the Tibetan experiment in Section 4.4, where an in-domain encoder-based reward is trained for the low-resource setting rather than directly substituting a generic embedding model for the reranker. 

<table><tr><td>Field</td><td>Prompt instruction</td></tr><tr><td>Input</td><td>Given a Chinese news input and a Thai candidate title, evaluate whether the Thai title preserves the source-side entities, events, and factual relations.</td></tr><tr><td>Entity</td><td>Score whether named entities, quantities, locations, organizations, and other salient participants are correctly preserved.</td></tr><tr><td>Event</td><td>Score whether the main action, event, or state described in the source is correctly expressed.</td></tr><tr><td>Factual</td><td>Score whether the output avoids unsupported claims, contradictions, and hallucinated details.</td></tr><tr><td>Fluency</td><td>Score whether the Thai output is grammatical, natural, and readable.</td></tr><tr><td>Conciseness</td><td>Score whether the output follows title style and avoids unnecessary verbosity or formatting artifacts.</td></tr><tr><td>Output</td><td>Return integer scores for the five dimensions and a brief justification.</td></tr></table>


Table 5: Five-dimensional judge prompt used for intermediate RL-checkpoint diagnostics.


<table><tr><td>Field</td><td>Prompt instruction</td></tr><tr><td>Input</td><td>Given a long Chinese passage and a Thai translation, evaluate translation quality without relying only on word overlap.</td></tr><tr><td>Entity</td><td>Score whether important entities and numerical information are preserved.</td></tr><tr><td>Event</td><td>Score whether the major events, actions, and relations are translated correctly.</td></tr><tr><td>Factual</td><td>Score whether the translation is faithful to the source and avoids hallucination.</td></tr><tr><td>Thai fluency</td><td>Score whether the Thai text is fluent, grammatical, and coherent.</td></tr><tr><td>Completeness</td><td>Score whether the translation covers the important information in the source passage.</td></tr><tr><td>Output</td><td>Return dimension scores and an overall assessment.</td></tr></table>


Table 6: Judge prompt used for long-form Chinese-to-Thai translation transfer evaluation.
