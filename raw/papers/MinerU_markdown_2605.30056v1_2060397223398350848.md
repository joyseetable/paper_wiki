# Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance

Shutong Ding 1 * Zejia Zhong 1 * Zhongyi Wang 1 Ke Hu 1 Bikang Pan 1 Jingya Wang 1 Ye Shi 1 † 

# Abstract

Recent advances in reinforcement learning (RL) have achieved great successes by leveraging the multimodality and exploration capability of diffusion policies. Among these approaches, one representative branch focuses on the sampling-based policy optimization. This design enables better exploration capability of the diffusion model, particularly at the beginning of training, but suffer from low exploitation in Q-value information, resulting in a slow policy convergence. Another branch pays attention to gradient-based policy optimization, which sufficiently exploits the gradient of the Q function yet tends to collapse into a unimodal policy with low diversity. To address this issue, we propose CGPO, Critic-Guided diffusion Policy Optimization, which effectively balances exploration and exploitation with the training-free guidance technique integrated into the denoising process of diffusion policy. Concretely, CGPO steers action generation toward high-value regions defined by the critic network and uses the guided actions as regression objectives. In this manner, CGPO reduces the time required to obtain high-quality actions and improves final performance with better balance between the exploration-exploitation tradeoff. We validate the effectiveness of CGPO on 5 Mu-JoCo locomotion tasks, and CGPO achieves stateof-the-art performance compared with existing diffusion-based RL methods. Notably, CGPO is the first success to incorporate diffusion policy into real-world RL, with its superior performance on Franka robot arm grasping tasks. Our official page is released at https://dingsht. tech/cgpo-webpage. 

# 1. Introduction

Recently, diffusion models have emerged as a powerful class of generative models, demonstrating strong capability in modeling complex and multimodal distributions (Ho et al., 2020; Song et al., 2020b;a). Motivated by these advantages, diffusion-based reinforcement learning has attracted increasing attention, as diffusion policies can naturally capture multimodal action distributions and encourage diverse exploration (Yang et al., 2023; Psenka et al., 2023; Ding et al., 2024; Wang et al., 2024; Ding et al., 2025). Existing approaches can be broadly divided into two lines of work: sampling-based optimization and gradient-based optimization. In sampling-based methods, actions sampled from a diffusion policy are reweighted according to their estimated values or advantages (Ding et al., 2024; Ma et al., 2025). In gradient-based methods, the gradients of a Q-network are used to guide the diffusion policy toward high-quality actions by training its noise prediction network (Yang et al., 2023; Psenka et al., 2023). Although these methods often exhibit reasonably high sample efficiency, their optimization mechanisms still inherently suffer from the explorationexploitation trade-off. 

In parallel, diffusion policies have also been actively explored in real-world robotic control, particularly in the context of imitation learning (Chi et al., 2023; Wu et al., 2025). By learning from large collections of human demonstrations, these methods have shown strong performance and robustness in manipulation tasks. Nevertheless, imitationlearning-based diffusion policies suffer from two fundamental limitations. First, collecting high-quality demonstrations in real-world settings is expensive and labor-intensive. Second, imitation learning is inherently bounded by the quality of human experts and therefore cannot systematically exceed demonstrated performance. These limitations motivate diffusion-based RL methods that learn directly from realworld interaction and surpass human demonstrations, yet this remains largely underexplored in existing works. 

To bridge this gap, we first systematically analyze the limitations of existing works on diffusion-based RL and propose CGPO (Critic-Guided Diffusion Policy Optimization), a diffusion-based reinforcement learning framework that enables efficient training in real-world robotic systems without demonstrations. Unlike prior methods that rely on sampling a large set of candidate actions followed by value-based reweighting, CGPO integrates classifier guidance into the diffusion denoising process, directly steering action generation toward high-value regions defined by a learned critic (Dhariwal & Nichol, 2021; Ho & Salimans, 2022). The resulting guided actions serve as optimization targets for updating the diffusion policy, eliminating the need for expensive candidate sampling while enabling more precise, gradient-informed policy improvement. By providing a continuous and directional optimization signal, CGPO effectively overcomes the refinement bottleneck of sampling diffusion RL methods. To summarize, our contributions are fourfold: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/bfc78f9bfcc21fdb8239041eb8c46bbffa607d0bb4c44078a0bd542d0da23b0d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/97b1cb2a566a50f6fd3d0206556cfb194e2945b3d052df2df85e950ebef5f9c8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/ec1208b2105270ebd4cb3686719f964f11d9a5763960820b700c37d7ecbf7b57.jpg)



Figure 1. Overview of CGPO. Compared with Q-guided and sampling-based diffusion RL that suffers from low action diversity and slow improvement, CGPO performs critic-guided action generation during training to improve policy fitting and downstream task performance.


• We systematically analyze existing diffusion-based RL methods and interpret their limitations through the lens of exploration-exploitation trade-offs, providing a unified explanation for their suboptimal action discovery behavior and slow policy convergence. 

• We propose CGPO, a novel diffusion-based reinforcement learning framework that integrates critic guidance directly into the denoising process of diffusion policies. By steering action generation toward high-value regions while preserving sufficient diversity, CGPO achieves a principled trade-off between exploration and exploitation. 

• We also introduce several training recipe for critic-guided diffusion RL, including (i) a value-calibrated network that mitigates the impact of value drift on state-conditional weighting, and (ii) an overestimation-reduced critic target 

construction to improve the reliability of the Q signal used for both weighting and guidance. 

• We conduct extensive evaluations in both simulation and real-world environments, including MuJoCo continuous control benchmarks and real-world manipulation tasks on a Franka robotic arm. The results show that CGPO consistently outperforms existing diffusion-based reinforcement learning methods as well as strong conventional baselines, demonstrating its effectiveness and practical relevance. 

# 2. Related Work

Diffusion Models in Reinforcement Learning. Diffusion models have recently been applied to reinforcement learning across offline, off-policy, and on-policy settings. In offline reinforcement learning (Levine et al., 2020; Park et al., 2025), diffusion models are primarily used to policy networks to adhere to the behavioral cloning framework, enabling policy optimization by sampling actions or trajectories that remain within the data support. In off-policy reinforcement learning, DIPO (Yang et al., 2023) optimizes diffusion policies by augmenting a behavior-cloning objective with Q-function gradients, and relies on the diffusion models inherent stochasticity for exploration rather than explicitly regulating exploratory behavior. Notably, its extensions to multi-modal policy learning do not resolve this lack of explicit exploration control. QSM (Psenka et al., 2023) avoids backpropagation through the diffusion chain by matching the policy score to $\nabla _ { a } Q ( s , a )$ , but disregards policy entropy and therefore requires heuristic noise injection for exploration. DACER (Wang et al., 2024) backpropagates gradients through the diffusion process but optimizes only expected Q-values and does not model a backward process; exploration is induced via additional Gaussian noise with entropy estimated approximately using a Gaussian mixture model. Similarly, QVPO (Ding et al., 2024) reweights diffusion losses using transformed Q-values. Soft Diffusion Actor-Critic (SDAC) (Ma et al., 2025) demonstrates improved sample efficiency but still requires careful energy-based modeling and incurs computational costs. In on-policy reinforcement learning, DPPO (Ren et al., 2024) proposes practical recipes for fine-tuning expressive diffusion policies using policy-gradient updates, enabling structured on-manifold exploration and stable long-horizon training. FPO (McAllister et al., 2025) estimates policy importance ratios via an ELBO objective, providing a scalable approximation but inducing asymmetric estimation bias. The estimates tend to be more reliable when the importance ratio increases than when it decreases, which can amplify variance and compromise training stability. GenPO (Ding et al., 2025) leverages exact diffusion inversion to construct invertible action mappings and introduces a doubled dummy action mechanism that achieves invertibility through alternating updates, thereby yielding tractable log-likelihoods. 

Diffusion Policy in Real-World Tasks. Diffusion Policy (DP) (Chi et al., 2023) establishes diffusion models as a strong visuomotor policy class for real-world manipulation by generating short-horizon action sequences, or action chunks, conditioned on visual observations and robot proprioception, and training the denoiser via behavior cloning on demonstrations. Building on DP, subsequent work improves generalization by enriching the policy input and representation, for example DP3 (Ze et al., 2024) leverages simple 3D scene representations to provide geometry-aware conditioning for diffusion-based visuomotor control. Another line of work scales diffusion policies through large-scale pretraining, for example RDT-1B (Liu et al.) pretrains a large diffusion-transformer policy on broad multi-task robot experience and then adapts it to downstream tasks. A remaining challenge is how to use reinforcement learning to improve diffusion policies beyond demonstration-limited behavior. Although DSRL (Wagenmaker et al., 2025) introduces RL signals, it mainly fine-tunes the noise distribution (or sampling behavior) around a demonstration-anchored diffusion model, which still limits how far the policy can move away from demonstrations. 

Despite the progress made by prior work, existing approaches either rely on diffusion models primarily for imitation learning or struggle to achieve strong exploration and efficient utilization in reinforcement learning settings. In contrast, our method incorporate a critic guidance into the denoising process of diffusion: it preserves the expressive generative representations of diffusion policies while leveraging value-based feedback and resulting in effective policy improvement. 

# 3. Preliminaries

# 3.1. Reinforcement Learning

Reinforcement learning (RL) studies sequential decisionmaking under uncertainty, where an agent interacts with an environment to maximize long-term return. A discounted Markov decision process (MDP) is defined as $\boldsymbol { \mathcal { M } } = ( \boldsymbol { \mathcal { S } } , \boldsymbol { \mathcal { A } } , \boldsymbol { P } , \boldsymbol { r } , \boldsymbol { \gamma } )$ , where S and $\mathcal { A } \subseteq \mathbb { R } ^ { d }$ denote the state and continuous action spaces, $\textstyle P ( \cdot \mid s , a )$ is the transition kernel, $r : \mathcal { S } \times \mathcal { A } $ R is the reward function, and $\gamma \in ( 0 , 1 )$ is the discount factor. A stochastic policy $\pi ( a \mid s )$ specifies the agent’s action distribution given state s. The standard RL objective is to maximize the expected discounted return: 

$$
J (\pi) = \mathbb {E} _ {\pi , P} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r (s _ {t}, a _ {t}) \right]. \tag {1}
$$

To evaluate a policy, the action-value function $Q ^ { \pi } ( s , a )$ is defined as the expected return after taking action a in state s and following policy π thereafter: 

$$
Q ^ {\pi} (s, a) := \mathbb {E} _ {\pi , P} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r \left(s _ {t}, a _ {t}\right) \mid s _ {0} = s, a _ {0} = a \right]. \tag {2}
$$

The action-value function satisfies the Bellman expectation equation, which decomposes the return into the immediate reward and the discounted value of successor states: 

$$
Q ^ {\pi} (s, a) = r (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim P (\cdot | s, a), a ^ {\prime} \sim \pi (\cdot | s ^ {\prime})} [ Q ^ {\pi} (s ^ {\prime}, a ^ {\prime}) ]. \tag {3}
$$

In off-policy RL, transitions $( s , a , r , s ^ { \prime } )$ are collected by a behavior policy and stored in a replay buffer D. Actor– critic methods learn a parameterized critic $Q _ { \omega } ( s , a )$ from Bellman-style regression targets and update a parameterized policy using critic-based policy improvement signals. 

# 3.2. Diffusion Policies

A diffusion policy models the action distribution $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ as a conditional DDPM in action space, where the clean variable $x _ { 0 } \in \mathbb { R } ^ { d }$ corresponds to the action a and the state s serves as the conditioning input (Chi et al., 2023; 2020). Let T be the number of diffusion steps and $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ be a variance schedule, with $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ 

Noising and denoising. The forward process gradually corrupts $x _ { 0 }$ into Gaussian noise, and the reverse process is parameterized by a denoiser $\epsilon _ { \theta } ( x _ { t } , s , t )$ that predicts the injected noise at each step. Under the standard noiseprediction parameterization, the forward marginal and the induced clean-action prediction are 

$$
x _ {t} = \sqrt {\bar {\alpha} _ {t}} x _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \epsilon , \qquad \epsilon \sim \mathcal {N} (0, I),
$$

$$
\hat {x} _ {0} (x _ {t}, s, t) = \frac {1}{\sqrt {\bar {\alpha} _ {t}}} \left(x _ {t} - \sqrt {1 - \bar {\alpha} _ {t}} \epsilon_ {\theta} (x _ {t}, s, t)\right). \tag {4}
$$

Sampling starts from $x _ { T } \sim \mathcal { N } ( 0 , I )$ and applies the learned reverse transitions for $t = T , \dots , 1$ , returning the final action $a = x _ { 0 }$ (or equivalently xˆ0 at the last step). 

Training. Given state–action pairs $( s , x _ { 0 } )$ , the denoiser is trained by the standard noise-prediction objective, which draws a random timestep and regresses the predicted noise to the injected noise: 

$$
\mathbb {E} _ {s, x _ {0}, t, \epsilon} \left[ \| \epsilon - \epsilon_ {\theta} (x _ {t}, s, t) \| _ {2} ^ {2} \right], t \sim \text { Uniform } \{1, \dots , T \}. \tag {5}
$$

This objective yields a conditional generative model whose sampling-time trajectory can later be modified by external objectives (e.g., value guidance) without changing the diffusion parameterization. 

# 3.3. Training-free Guidance

Training-free guidance steers the reverse diffusion trajectory using gradients of an external differentiable objective, without updating diffusion parameters or training an additional time-conditioned guidance model (Dhariwal & Nichol, 2021; Ho & Salimans, 2022; Chung et al., 2023; Song et al., 2023; Yu et al., 2023). Let $\ell ( \cdot ; y )$ be an objective defined on the clean prediction xˆ0, where y denotes the desired condition. 

Generic loss-guided update. Given the mapping $x _ { t } \mapsto$ $\hat { x } _ { 0 } ( x _ { t } , t )$ induced by the diffusion parameterization, guidance forms 

$$
g _ {t} := \nabla_ {x _ {t}} \ell (\hat {x} _ {0} (x _ {t}, t); y). \tag {6}
$$

A common plug-in correction modifies one reverse step by shifting the mean along $- g _ { t } \mathbf { \cdot }$ : 

$$
x _ {t - 1} = \mu_ {\theta} (x _ {t}, t) - \eta_ {t}   g _ {t} + \sigma_ {t}   \epsilon_ {t};   \epsilon_ {t} \sim \mathcal {N} (0, I), \tag {7}
$$

where $\left( \mu _ { \theta } , \sigma _ { t } \right)$ follow the chosen reverse parameterization and $\eta _ { t } > 0$ is a stepsize (Chung et al., 2023; Song et al., 2023). 

Other training-free variants. Beyond the additive meanshift in Equation (7), related training-free rules include score/noise mixing (e.g., classifier(-free) guidance) that alters the effective score used in the reverse step (Dhariwal & Nichol, 2021; Ho & Salimans, 2022), and projection/proximal updates that enforce constraints by projecting guided proposals back to a feasible set (He et al., 2024). 

# 4. Methods

In this section, we present Critic-Guided Diffusion Policy Optimization (CGPO). Our motivation is that samplingbased diffusion-policy improvement becomes less effective as training proceeds: under a fixed sampling budget, action candidates drawn from a progressively concentrated policy tend to be similar, making critic scores less discriminative and weakening the practical update signal (Ding et al., 2024). 

CGPO replaces multi-candidate selection with a single guided target per state. Specifically, we generate $a ^ { g } ( s )$ by training-free loss-guided reverse diffusion under an actionvalue critic (Song et al., 2023; Yu et al., 2023), apply DSG only in the last G denoising steps to focus guidance where clean predictions are more reliable (Yang et al., 2024), and train the actor with reweighted denoising regression on these guided targets (Ding et al., 2024). To keep the training signal well-conditioned, CGPO additionally (i) learns a lightweight base value network to stabilize the scale of critic-derived weights, and (ii) adopts overestimation-aware critic targets and aggregation so that the Q signal used for weighting and guidance remains reliable throughout training (Van Hasselt et al., 2016; Kuznetsov et al., 2020). 

# 4.1. Limitation in Sampling-based Optimization

Many online diffusion-policy RL methods improve the actor by drawing a finite set of candidate actions from the current policy and then using a learned critic to choose among these candidates (Ding et al., 2024). For each state s, the improvement step starts from a finite candidate set 

$$
a _ {i} \sim \pi_ {\theta} (\cdot \mid s), \quad \mathcal {A} _ {K} (s) = \left\{a _ {i} \right\} _ {i = 1} ^ {K}. \tag {8}
$$

The update signal is therefore sampling-driven: it is limited by the diversity of $\boldsymbol { \mathcal { A } } _ { K } ( \boldsymbol { s } )$ in Equation (8). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/854c9bb8cf5effeca7a90d386437f6c4cbccda7e8e07c727891704ce1cf69b3a.jpg)



Figure 2. t-SNE visualization of sampled candidate actions on HalfCheetah-v3 at different training stages (left to right). Points are colored by critic-based labels (good vs. bad). As the policy concentrates, candidate diversity shrinks, and within-set separability decreases, consistent with a reduced critic contrast under finite candidate sampling.


A key limitation is that, as training proceeds, $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s )$ becomes more concentrated, so the candidates in Equation (8) 

cover a narrower neighborhood under a fixed budget K. Consequently, the critic values $\{ \hat { Q } _ { \psi } ( s , a _ { i } ) \} _ { i = \cdot } ^ { K }$ 1 become less discriminative within $\boldsymbol { \mathcal { A } } _ { K } ( \boldsymbol { s } )$ . A simple evaluation metric for this problem is the within-set critic contrast $\Delta _ { Q } ( s )$ , which is defined as (9). This metric shrinks as candidates become similar; Figure 2 provides a toy example consistent with this effect. When $\Delta _ { Q } ( s )$ is small, selecting the best candidate yields only a marginal gain over a typical sample. Although we can alleviate it by increasing K or the number of diffusion steps, the increased computation and control latency are undesirable for real-robot learning. 

$$
\Delta_ {Q} (s) = \max _ {a \in \mathcal {A} _ {K} (s)} \hat {Q} _ {\psi} (s, a) - \frac {1}{K} \sum_ {i = 1} ^ {K} \hat {Q} _ {\psi} (s, a _ {i}), \tag {9}
$$

# 4.2. Critic-Guided Actions for Policy Improvement

Sampling-based improvement with a finite candidate set can weaken as training progresses(Section 4.1), motivating a more direct way to search for better actions. In conditional generation, classifier guidance steers diffusion trajectories by following gradients of a conditioning objective, often yielding substantially stronger controllability than pure sampling (Dhariwal & Nichol, 2021).This suggests a natural idea for diffusion-policy RL: use gradient-based guidance to synthesize higher-value actions and then train the policy toward these improved targets. 

Classic classifier guidance requires a time-conditioned guidance model; in RL this would amount to learning a critic such as $\hat { Q } ( s , x _ { t } , t )$ that is accurate across diffusion noise levels, increasing training cost and introducing an undesirable bilevel coupling in online/robot learning. Training-free guidance instead backpropagates an objective on ${ \hat { x } } _ { 0 }$ through $x _ { t } \mapsto \hat { x } _ { 0 } ( x _ { t } , s , t )$ without an extra network (Chung et al., 2023; Song et al., 2023; Yu et al., 2023), but naive guidance can suffer from manifold deviation and thus needs conservative steps for stability (Yang et al., 2024; He et al., 2024). 

Motivated by DSG (Yang et al., 2024), CGPO adopts a typicality-preserving constraint for guided reverse diffusion steps, which incorporate guidance while remaining close to the high-probability region of the unconditional reverse transition. 

CGPO instantiates training-free guidance using a standard action-value critic on clean actions. For each replay state s in an actor update, CGPO synthesizes a single refined target action $a ^ { g } ( s )$ by running a guided reverse diffusion chain. The guidance signal is defined by the gradient of the critic objective with respect to the predicted clean action $\hat { x } _ { 0 } ( x _ { t } , s , t )$ : 

$$
\mathcal {L} _ {\text { guide }} (x _ {t}; s) = - \hat {Q} (s, \hat {x} _ {0} (x _ {t}, s, t)), \tag {10}
$$

$$
g _ {t} = \nabla_ {x _ {t}} \mathcal {L} _ {\mathrm{guide}} (x _ {t}; s),
$$

where $\hat { Q }$ denotes a conservative scalar value estimate, the proof can be referred to Appendix A.5. 

Standard unconditional reverse steps, $x _ { t - 1 } = \mu _ { \theta } ( x _ { t } , s , t ) +$ $\sigma _ { t } \epsilon _ { t }$ , rely on the property that Gaussian perturbations in d dimensions concentrate around a spherical shell with a typical radius $r _ { t } = \sqrt { d } \sigma _ { t }$ . To incorporate guidance without violating this manifold structure, CGPO employs DSG. First, it maps the gradient $g _ { t }$ to a constrained descent direction $d ^ { \star }$ located on this sphere: 

$$
d ^ {\star} = - r _ {t} \frac {g _ {t}}{\left\| g _ {t} \right\| _ {2} + \varepsilon}, \tag {11}
$$

where ε is a constant for numerical stability. To preserve the stochastic nature of diffusion, DSG does not use $d ^ { \star }$ directly. Instead, it mixes this deterministic direction with the standard Gaussian noise $\epsilon _ { t } \sim \mathcal { N } ( 0 , I )$ using a guidance rate $\rho \in [ 0 , 1 ]$ : 

$$
d _ {m} = \sigma_ {t} \epsilon_ {t} + \rho \left(d ^ {\star} - \sigma_ {t} \epsilon_ {t}\right). \tag {12}
$$

Finally, the mixed direction $d _ { m }$ is projected back onto the typical-radius shell to ensure the update magnitude matches the diffusion noise schedule, yielding the final update step: 

$$
x _ {t - 1} = \mu_ {\theta} (x _ {t}, s, t) + r _ {t} \frac {d _ {m}}{\| d _ {m} \| _ {2} + \varepsilon}. \tag {13}
$$

A key difference from conditional generation is that, in RL, the critic can be highly inaccurate and noisy early in training. Consequently, using raw critic gradients for guidance can be misleading and may destabilize refinement, especially when $\hat { x } _ { 0 } ( x _ { t } , s , t )$ is still far from a plausible action at large noise levels. 

To stabilize guidance, CGPO uses a conservative value estimate based on truncated quantiles (Kuznetsov et al., 2020). Specifically, with a distributional critic that outputs M quantile estimates per critic in an N -member ensemble, let $Z _ { ( 1 ) } ^ { ( n ) } ( s , a ) \leq \cdots \leq Z _ { ( M ) } ^ { ( n ) } ( s , a )$ ) denote the sorted quantiles. CGPO discards the top k quantiles and aggregates the remainder as 

$$
\hat {Q} (s, a) = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\frac {1}{M - k} \sum_ {m = 1} ^ {M - k} Z _ {(m)} ^ {(n)} (s, a)\right), \tag {14}
$$

which yields a more conservative and stable guidance signal.In Equation (10), Qˆ is instantiated by Equation (14), improving the reliability of the guidance direction. 

In addition, CGPO applies DSG guidance only in the last G denoising steps, while earlier steps follow the unconditional reverse transition. Restricting guidance to late steps reduces out-of-distribution critic queries and concentrates refinement in the regime where action-level changes become semantically meaningful. The guided sampler is used only during training to produce $a ^ { g } ( s ) ;$ ; environment interaction and deployment use the unguided sampler to preserve runtime efficiency under real-robot control-loop latency constraints. 

# 4.3. State Reweighting via Calibrated Value Signals

This section specifies the training objectives and update rules used in our implementation of CGPO. As discussed in Section 4.2, each actor update is supervised by a single training-time guided target $a ^ { g } ( s )$ , while environment interaction and deployment use the unguided sampler to preserve runtime efficiency. 

Our implementation couples state reweighting with valuesignal calibration to make critic-weighted diffusion updates reliable. Concretely, CGPO uses three tightly connected components: (i) weighted denoising regression on guided targets, together with a matched diffusion-entropy regularization (Ding et al., 2024); (ii) a value-calibrated network $V _ { \phi } ( s )$ that anchors the weight scale and reduces sensitivity to value drift; and (iii) an overestimation-aware critic target/aggregation scheme (DDQN targets (Van Hasselt et al., 2016) and truncated-quantile aggregation (Kuznetsov et al., 2020)) that improves the reliability of the Q signal used by both weighting and guidance. We next describe these components in the same order. 

Value-calibrated network. A recurring issue in criticweighted updates is that the scale of $\hat { Q } ( s , a )$ can drift during training, making weights unstable and leading to overly aggressive or vanishing actor updates (Ding et al., 2024). CGPO introduces a lightweight value-calibrated network $V _ { \phi } ( s )$ (a small MLP) and forms weights from a rectified advantage: 

$$
w (s) = \max \left(\hat {Q} (s, a ^ {g} (s)) - V _ {\phi} (s), 0\right). \tag {15}
$$

Importantly, $V _ { \phi }$ is trained to track the critic value of unguided actor samples, which anchors the value-calibrated network to the current policy distribution and avoids coupling it to the guided target generator. Concretely, let $\tilde { a } ( s ) \sim \pi _ { \theta } ( \cdot \mid s )$ be an unguided policy sample and define 

$$
y _ {V} (s) = \operatorname{sg} \bigl (\bar {Q} (s, \tilde {a} (s)) \bigr), \tag {16}
$$

where $\operatorname { s g } ( \cdot )$ stops gradients and Q¯ uses the same critic aggregation rule as in Section 4.2. The base network is optimized by 

$$
\mathcal {L} _ {V} = \mathbb {E} _ {s \sim \mathcal {D}} \left[ (V _ {\phi} (s) - y _ {V} (s)) ^ {2} \right]. \tag {17}
$$

This value-calibrated network is used only to stabilize the weighting in the actor regression loss; it is not used in the guide objective. This design controls the scale of state reweighting, while the critic-side calibration in Section 4.2 improves the reliability of the underlying Q signal. 

Weighted denoising and diversity regularization. Given guided targets $a ^ { g } ( s )$ , the diffusion actor is trained by weighted denoising regression, where $w ( s )$ directly implements state reweighting in the actor loss: 

$$
\ell_ {\mathrm{diff}} (s, t, \epsilon) = w (s) \| \epsilon - \epsilon_ {\theta} (x _ {t} ^ {g}, s, t) \| _ {2} ^ {2},
$$

$$
\mathcal {L} _ {\text { diff }} = \mathbb {E} _ {s \sim \mathcal {D}, t, \epsilon} \left[ \ell_ {\text { diff }} (s, t, \epsilon) \right], \tag {18}
$$

$$
x _ {t} ^ {g} = \sqrt {\bar {\alpha} _ {t}} a ^ {g} (s) + \sqrt {1 - \bar {\alpha} _ {t}} \epsilon ,
$$

where t ∼ Uniform $\{ 0 , \ldots , T - 1 \}$ and $\epsilon \sim \mathcal { N } ( 0 , I )$ . To preserve action diversity, we additionally optimize a diffusion entropy regularization (Ding et al., 2024) using matched weighting: 

$$
\ell_ {\mathrm{ent}} (s, t, \epsilon) = w _ {\mathrm{ent}} (s) \left\| \epsilon - \epsilon_ {\theta} \bigl (x _ {t} ^ {U}, s, t \bigr) \right\| _ {2} ^ {2},
$$

$$
\mathcal {L} _ {\text { ent }} = \mathbb {E} _ {s \sim \mathcal {D}, a \sim U (\underline {{a}}, \overline {{a}}), t, \epsilon} \left[ \ell_ {\text { ent }} (s, t, \epsilon) \right], \tag {19}
$$

$$
x _ {t} ^ {U} = \sqrt {\bar {\alpha} _ {t}} a + \sqrt {1 - \bar {\alpha} _ {t}} \epsilon ,
$$

where $w _ { \mathrm { e n t } } ( s ) = \lambda _ { \mathrm { e n t } } w ( s )$ and $\lambda _ { \mathrm { e n t } } > 0$ is a tunable hyperparameter. 

Q-signal calibration. Since the same $Q$ signal is used to assign weights in Equation (15) and provide guidance gradients for synthesizing $a ^ { g } ( s )$ , optimistic value targets can distort both the update emphasis and the direction of improvement. CGPO therefore adopts an overestimationaware target construction and aggregation scheme. Specifically, critic targets follow a DDQN update, decoupling action selection from target evaluation (Van Hasselt et al., 2016). When using a distributional critic, we form the scalar $\hat { Q }$ by truncated-quantile aggregation, discarding the highest quantiles before computing targets (Kuznetsov et al., 2020). Together with the base value network $V _ { \phi } .$ this calibration yields a $Q$ signal that is stable in scale and reliable in ranking, which is crucial for critic-weighted diffusion updates. 

Overall procedure. The overall training procedure follows a standard off-policy actor–critic loop: a mini-batch of states is sampled from the replay buffer, guided targets $a ^ { g } ( s )$ are synthesized for actor learning, the diffusion actor is updated by Equations (18) and (19), and the critic is updated by Bellman regression with the target construction above. Algorithm 1 summarizes the procedure. Overall, (iii) calibrates the Q signal, (ii) anchors its scale for stable reweighting, and (i) turns the resulting state weights into effective diffusion-actor updates. 

# 5. Experiments

We evaluate CGPO in simulation and on a real robot to assess both learning performance and practical deployability. Our experiments are designed to answer three questions: (i) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/0d52269e14cdde7718711567b47234028ccaf6dae9a9e9a5d33c632a7350400a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/f782dd8818d70ed85598eb07b14b1f9276336f008d6805eef7590db110652df2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/0525fab004bc0c7fe739b23769a5f3c40ba40349df1a4ee15ce8a6e1d870e6f2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/f73ab526048b10717b9cfe0bf8eb66f6249fb582c0cbe7a4919918ecb603ac8e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/accc1f469cd63d4236c0defcff7689b17ec7989dfcf791b26b5d25948c09e8b8.jpg)



SPO SAC TD3 PPO QVPO DIPO SDAC DACER QSM CGPO(*)



Figure 3. Learning curves on five MuJoCo v3 locomotion tasks over ${ 1 0 } ^ { 6 }$ environment steps. Curves show the mean episodic return across five random seeds; shaded bands indicate ±1 standard deviation.



Table 1. Best episodic return achieved during training (mean ± standard deviation over five seeds). TD3 uses a deterministic unimodal actor; SAC/PPO/SPO use Gaussian stochastic policies; the remaining methods use diffusion policies. N/A indicates the method does not produce valid results under our setting.


<table><tr><td rowspan="2">Task</td><td>Unimodal</td><td colspan="3">Gaussian</td><td colspan="6">Diffusion</td></tr><tr><td>TD3</td><td>SAC</td><td>PPO</td><td>SPO</td><td>DIPO</td><td>QSM</td><td>DACER</td><td>QVPO</td><td>SDAC</td><td>CGPO(*)</td></tr><tr><td>Ant-v3</td><td>4583.8 (69.5)</td><td>5030.9 (1000.3)</td><td>2781.9 (74.1)</td><td>2100.2 (302.4)</td><td>6100.0 (177.3)</td><td>N/A</td><td>5910.8 (82.3)</td><td>6425.1 (67.6)</td><td>1404.6 (105.3)</td><td>7272.1 (74.6)</td></tr><tr><td>HalfCheetah-v3</td><td>10388.6 (80.4)</td><td>10616.9 (72.8)</td><td>4773.5 (53.4)</td><td>4008.2 (246.8)</td><td>9555.0 (1654.3)</td><td>3888.2 (632.6)</td><td>10232.2 (272.5)</td><td>11385.6 (164.5)</td><td>11179.4 (325.1)</td><td>14368.6 (310.6)</td></tr><tr><td>Hopper-v3</td><td>3267.5 (8.5)</td><td>2996.6 (111.9)</td><td>3154.3 (426.2)</td><td>2212.8 (988.4)</td><td>3724.1 (240.7)</td><td>2154.7 (998.2)</td><td>3436.3 (33.2)</td><td>3728.5 (13.8)</td><td>2957.8 (323.7)</td><td>4170.5 (165.6)</td></tr><tr><td>Humanoid-v3</td><td>5353.5 (53.7)</td><td>5159.7 (475.3)</td><td>713.7 (85.9)</td><td>797.4 (262.1)</td><td>5280.1 (54.6)</td><td>4793.1 (229.5)</td><td>5291.3 (135.4)</td><td>5306.6 (14.5)</td><td>4429.6 (174.2)</td><td>7131.4 (243.8)</td></tr><tr><td>Walker2d-v3</td><td>3513.9 (40.7)</td><td>4888.1 (80.0)</td><td>3751.5 (609.1)</td><td>3321.8 (1328.9)</td><td>4917.1 (181.9)</td><td>3613.4 (1443.5)</td><td>4381.7 (226.2)</td><td>5191.8 (60.2)</td><td>4250.8 (411.9)</td><td>6482.6 (170.9)</td></tr></table>

does CGPO improve online diffusion-policy reinforcement learning under a fixed interaction budget; (ii) does trainingtime critic-guided target synthesis yield more reliable policy improvement than sampling-based candidate selection when the policy distribution becomes concentrated; and (iii) can these gains be obtained without increasing deployment-time computation, by keeping rollout and evaluation unguided. 

We first benchmark CGPO on five MuJoCo v3 locomotion tasks against strong model-free baselines and recent diffusion-policy methods, reporting learning curves and best achieved performance under identical training budgets. We then validate CGPO on a Franka Emika Panda system under a standardized intervention protocol, highlighting that the training-time guidance mechanism translates to robust real-world learning while respecting control-loop latency constraints. 

# 5.1. Comparative Evaluation

We evaluate CGPO on five MuJoCo v3 locomotion benchmarks (Todorov et al., 2012) against representative online model-free RL baselines. The compared methods include off-policy algorithms (TD3 (Fujimoto et al., 2018), SAC (Haarnoja et al., 2018)), on-policy algorithms (PPO (Schulman et al., 2017), SPO (Chen et al., 2024)), and diffusion-policy / diffusion-actor–critic approaches (DIPO (Yang et al., 2023), DACER (Wang et al., 2024), QSM (Psenka et al., 2023), QVPO (Ding et al., 2024)), SDAC (Ma et al., 2025). All methods are trained for 106 environment steps with the same interaction budget and evaluation protocol, and results are reported over five random seeds. 

Figure 3 presents learning curves (mean return with ± one standard deviation across seeds). Table 1 reports, for each algorithm, the best episodic return attained during training, aggregated over the same five seeds; this metric summarizes the strongest performance reached within the fixed data budget. 


Algorithm 1 Critic-Guided Diffusion Policy Optimization (CGPO)


Input: Diffusion policy $\pi_{\theta}$ , critic $\hat{Q}$ , replay buffer $\mathcal{D}$ , diffusion steps $T$ , guidance step $G$ , learning rates $\eta_{\theta}, \eta_{\psi}$ . Initialize actor parameters $\theta$ and critic parameters $\psi$ .

for iteration = 1, 2, ... do
    Collect transitions using the unguided sampler of $\pi_{\theta}$ and store $(s, a, r, s')$ in $\mathcal{D}$ .
    Sample a mini-batch of states $\{s_i\}_{i=1}^B$ from $\mathcal{D}$ .
    for each state $s$ in the mini-batch do
    Compute the refined target action $a^g(s)$ via critico-guided refinement
    (guidance objective in Equation (10); target synthesis procedure in Alg. 2).

end for
Compute $\mathcal{L}_{\text{diff}}$ and $\mathcal{L}_{\text{ent}}$ using Equations (18) and (19). Update actor: $\theta \leftarrow \theta - \eta_{\theta} \nabla_{\theta} (\mathcal{L}_{\text{diff}} + \alpha_{\text{ent}} \mathcal{L}_{\text{ent}})$ .
    Update critic parameters $\psi$ by minimizing a conservative Bellman regression objective.

end for 

Across all five tasks, CGPO achieves the best overall performance and exhibits consistently stable learning dynamics. Beyond final scores, the learning curves indicate that CGPO maintains effective optimization progress in regimes where sampling-based diffusion policy improvement tends to slow down. This behavior is consistent with CGPO’s trainingtime critic-guided target generation: instead of relying on finite candidate sets to indirectly shape the actor update, CGPO constructs a single guided target per state and trains the actor by denoising regression on these targets. As a result, the policy update remains informative even when the diffusion policy becomes concentrated and the critic values of nearby samples become less separable. 

Finally, we emphasize that critic guidance is used only during training to generate supervised targets for actor updates. Environment interaction and evaluation use the standard unguided sampler, keeping test-time inference cost and control-loop latency comparable to vanilla diffusion-policy execution, which is important for real-world deployment. 

# 5.2. Ablation Study

To isolate the contribution of each component, we conduct ablations on Ant-v3, a representative high-dimensional locomotion task. As shown in Figure 5, we compare full CGPO with variants that remove critic guidance, replace DSG with naive guidance, remove DDQN, remove truncated-quantile aggregation, or remove the base value network. 

Guidance design. Figure 5a shows that removing guidance substantially reduces performance, while naive guidance improves over the unguided variant but remains below CGPO. This indicates that critic gradients are useful, but the DSG-constrained update is important for injecting guidance in a way that remains compatible with the diffusion reverse process. 

Q-signal calibration. Figures 5b and 5c show that removing either DDQN or truncated-quantile aggregation degrades learning. This supports the need for a reliable Q signal, since the same critic is used for both guidance and actor weighting. 

Base value network. As shown in Figure 5d, removing $V _ { \phi }$ leads to lower and less stable performance. This confirms that anchoring the scale of critic-derived weights to unguided actor samples is important for stable state reweighting. Overall, these ablations show that CGPO benefits from both guided target synthesis and calibrated critic-weighted actor learning. 

# 5.3. Real-world Experiments

Tasks and Protocol. We evaluate our method on two challenging tasks: Cube Stacking and Cylindrical Peg-in-Hole (Fig. 4). Following SiLRI (Zhao et al., 2025), we employ a four-stage intervention protocol that balances initial expert guidance with autonomous exploration. Corrective interventions are only triggered by recurring deviations, with a fallback to full demonstrations only after persistent failures. Further details about the robot setup can be found in the appendix. 

Quantitative Results. As shown in Table 2, we compare our CGPO policy against the SAC baseline at 15,000 training steps. CGPO achieves a 80% success rate (16/20) in the cylindrical peg-in-hole task, outperforming SAC by 15%. Qualitatively, CGPO generates significantly smoother exploration trajectories, whereas the SAC baseline exhibits severe jittering during the insertion phase (Fig. 4, row 2). 


Table 2. Success rates comparison of the cylindrical peg-in-hole task on the HIL-SERL Framework. We conducted 20 evaluation trials for each policy to calculate the success rate.


<table><tr><td>Method</td><td>Success Rate (%)</td></tr><tr><td>HIL-SERL-SAC</td><td>65</td></tr><tr><td>HIL-SERL-CGPO (ours)</td><td>80</td></tr></table>

# 6. Conclusion

In this work, we introduced CGPO, a new framework for diffusion-based reinforcement learning that addresses the optimization limitations of existing weighted diffusion methods. By integrating critic guidance into the diffusion denoising process, CGPO generates high-quality action targets for diffusion policy improvement, avoiding extensive candidate sampling and enabling more precise policy refinement beyond early-stage learning. More importantly, CGPO demonstrates that diffusion-based reinforcement learning can be trained directly in real-world robotic systems from scratch, without reliance on large-scale human demonstrations. This work bridges the gap between diffusion policies and real-world reinforcement learning, and opens up a promising direction for combining generative models with critic-driven optimization in embodied intelligence. We believe CGPO provides a general and extensible foundation for future research on scalable, efficient, and real-worldready diffusion-based reinforcement learning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/80e8e239336e2f12a302cdf03c51531a458d23df05e7901d3f0dd4a97d163f2f.jpg)



Figure 4. Sequential video frames of the real-world evaluation tasks. The top row shows the cube stacking task, and the bottom row shows the cylindrical peg-in-hole task.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/8a9ba31ab1e0e08e00b505f3cb5e67e1ed733fee8205c54c7b3a598d0d291703.jpg)



(a) Guidance variants


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/3ba07311f2b845ee0d0f7419b4ed6fd4a7888891e9f050c7190b0c1e1d157574.jpg)



(b) w/o DDQN


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/a0a29a1240c89f18b8788c6d4422b9226c28344a4402e33c617878c32f3e00de.jpg)



(c) w/o TQC


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/7e3f4df771e5672acff3176419a33a29c820b046d5a9d62858510dec15b1a79c.jpg)



(d) w/o base value network



Figure 5. Ablation study on Ant-v3. From left to right, we evaluate the effect of DSG guidance, DDQN target construction, truncatedquantile aggregation, and the base value network.


# Acknowledgment

This work was supported by the National Natural Science Foundation of China (62303319, 62406195), HPC Platform of ShanghaiTech University, and MoE Key Laboratory of Intelligent Perception and Human-Machine Collaboration (ShanghaiTech University), Shanghai Engineering Research Center of Intelligent Vision and Imaging. This work was also supported in part by computational resources provided by Fcloud CO., LTD. 

# Impact Statement

This work advances diffusion-based reinforcement learning by introducing CGPO, which utilize critic guidance that improves the explorationexploitation balance and enables diffusion policies to be trained directly through interaction. By reducing reliance on large-scale human demonstrations in imitation learning, CGPO has the potential to facilitate more practical and scalable deployment of diffusion policies in real-world robotic systems. 

# References



Chen, C., Deng, F., Kawaguchi, K., Gulcehre, C., and Ahn, S. Simple hierarchical planning with diffusion. arXiv preprint arXiv:2401.02644, 2024. 





Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. arXiv preprint arXiv:2303.04137, 2023. 





Chung, H., Kim, J., McCann, M. T., Klasky, M. L., and Ye, J. C. Diffusion posterior sampling for general noisy inverse problems. In 11th International Conference on Learning Representations, ICLR 2023, 2023. 





Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 





Ding, S., Hu, K., Zhang, Z., Ren, K., Zhang, W., Yu, J., Wang, J., and Shi, Y. Diffusion-based reinforcement learning via q-weighted variational policy optimization. Advances in Neural Information Processing Systems, 37: 53945–53968, 2024. 





Ding, S., Hu, K., Zhong, S., Luo, H., Zhang, W., Wang, J., Wang, J., and Shi, Y. Genpo: Generative diffusion models meet on-policy reinforcement learning. arXiv preprint arXiv:2505.18763, 2025. 





Fujimoto, S., Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pp. 1587–1596. PMLR, 2018. 





Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. PMLR, 2018. 





He, Y., Murata, N., Lai, C.-H., Takida, Y., Uesaka, T., Kim, D., Liao, W.-H., Mitsufuji, Y., Kolter, J. Z., Salakhutdinov, R., et al. Manifold preserving guided diffusion. In The Twelfth International Conference on Learning Representations, 2024. 





Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 





Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 





Kuznetsov, A., Shvechikov, P., Grishin, A., and Vetrov, D. Controlling overestimation bias with truncated mixture of continuous distributional quantile critics. In International conference on machine learning, pp. 5556–5566. PMLR, 2020. 





Levine, S., Kumar, A., Tucker, G., and Fu, J. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020. 





Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., and Zhu, J. Rdt-1b: a diffusion foundation model for bimanual manipulation. In The Thirteenth International Conference on Learning Representations. 





Luo, J., Xu, C., Wu, J., and Levine, S. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning, 2024. 





Ma, H., Chen, T., Wang, K., Li, N., and Dai, B. Soft diffusion actor-critic: Efficient online reinforcement learning for diffusion policy. arXiv e-prints, pp. arXiv–2502, 2025. 





McAllister, D., Ge, S., Yi, B., Kim, C. M., Weber, E., Choi, H., Feng, H., and Kanazawa, A. Flow matching policy gradients. arXiv preprint arXiv:2507.21053, 2025. 





Park, S., Li, Q., and Levine, S. Flow q-learning. arXiv preprint arXiv:2502.02538, 2025. 





Psenka, M., Escontrela, A., Abbeel, P., and Ma, Y. Learning a diffusion model policy from rewards via q-score matching. arXiv preprint arXiv:2312.11752, 2023. 





Ren, A. Z., Lidard, J., Ankile, L. L., Simeonov, A., Agrawal, P., Majumdar, A., Burchfiel, B., Dai, H., and Simchowitz, M. Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588, 2024. 





Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a. 





Song, J., Zhang, Q., Yin, H., Mardani, M., Liu, M.-Y., Kautz, J., Chen, Y., and Vahdat, A. Loss-guided diffusion models for plug-and-play controllable generation. In International Conference on Machine Learning, pp. 32483–32498. PMLR, 2023. 





Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b. 





Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999. 





Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012. 6386109. 





Van Hasselt, H., Guez, A., and Silver, D. Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence, volume 30, 2016. 





Wagenmaker, A., Nakamoto, M., Zhang, Y., Park, S., Yagoub, W., Nagabandi, A., Gupta, A., and Levine, S. Steering your diffusion policy with latent space reinforcement learning. arXiv preprint arXiv:2506.15799, 2025. 





Wang, Y., Wang, L., Jiang, Y., Zou, W., Liu, T., Song, X., Wang, W., Xiao, L., Wu, J., Duan, J., et al. Diffusion actor-critic with entropy regulator. Advances in Neural Information Processing Systems, 37:54183–54204, 2024. 





Wu, S., Zhu, Y., Huang, Y., Zhu, K., Gu, J., Yu, J., Shi, Y., and Wang, J. Afforddp: Generalizable diffusion policy with transferable affordance. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6971–6980, 2025. 





Yang, L., Huang, Z., Lei, F., Zhong, Y., Yang, Y., Fang, C., Wen, S., Zhou, B., and Lin, Z. Policy representation via diffusion probability model for reinforcement learning. arXiv preprint arXiv:2305.13122, 2023. 





Yang, L., Ding, S., Cai, Y., Yu, J., Wang, J., and Shi, Y. Guidance with spherical gaussian constraint for conditional diffusion. In Forty-first International Conference on Machine Learning, 2024. 





Yu, J., Wang, Y., Zhao, C., Ghanem, B., and Zhang, J. Freedom: Training-free energy-guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23174–23184, 2023. 





Ze, Y., Zhang, G., Zhang, K., Hu, C., Wang, M., and Xu, H. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations, 2024. 





Zhao, Y., Jin, H., Jiang, L., Zhang, X., Wu, K., Ren, P., Xu, Z., Che, Z., Sun, L., Wu, D., et al. Real-world reinforcement learning from suboptimal interventions. arXiv preprint arXiv:2512.24288, 2025. 





Zhu, Y., Joshi, A., Stone, P., and Zhu, Y. Viola: Imitation learning for vision-based manipulation with object proposal priors. arXiv preprint arXiv:2210.11339, 2022. doi: 10.48550/arXiv.2210.11339. 



# A. Proofs

# A.1. Relative Entropy Policy Search

Relative Entropy Policy Search (REPS) derives a closed-form update for a new sampling distribution by maximizing expected return while constraining the KL divergence to a reference distribution: 

$$
\pi_ {k + 1} \in \arg \max _ {\pi} J (\pi) \tag {20}
$$

$$
\text { s.t. } \quad \int_ {s} d ^ {\pi_ {k}} (s)   \mathrm{KL} (\pi (\cdot \mid s) \| \pi_ {k} (\cdot \mid s))   d s \leq \varepsilon \tag {21}
$$

where $\begin{array} { r } { J ( \pi ) = \sum _ { s } { d ^ { \pi _ { k } } ( s ) \sum _ { a } { \pi ( a \mid s ) Q ^ { \pi _ { k } } ( s , a ) } } } \end{array}$ . This derivation begins by forming the Lagrangian of the constrained optimization problem presented above, 

$$
\mathcal {L} (\pi , \eta) = \int_ {s} d ^ {\pi_ {k}} (s) \int_ {a} \pi (a \mid s) Q ^ {\pi_ {k}} (s, a) - \eta \left(\int_ {s} d ^ {\pi_ {k}} (s) \int_ {a} \pi (a \mid s) \log \frac {\pi (a \mid s)}{\pi_ {k} (a \mid s)} - \varepsilon\right) \tag {22}
$$

where $\beta$ is a Lagrange multiplier.Differentiating $\mathcal { L } ( \pi , \eta )$ with respect to $\pi ( a | s )$ and solving for the optimal policy results in the following expression for the optimal policy 

$$
\pi_ {k + 1} (a | s) = \frac {\pi_ {k} (a \mid s) \exp \left(\frac {Q ^ {\pi_ {k}} (s , a)}{\eta}\right)}{Z _ {\eta} (s)}, \tag {23}
$$

with $Z _ { \eta } ( s )$ being the partition function. 

# A.2. Policy Evaluation

Given policy $\pi _ { k }$ , the update rule for the Q-function as: 

$$
\left(\mathcal {T} ^ {\pi_ {k}} Q\right) (s, a) := r (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim P (\cdot | s, a), a ^ {\prime} \sim \pi_ {k} (\cdot | s ^ {\prime})} \left[ Q \left(s ^ {\prime}, a ^ {\prime}\right) \right] \tag {24}
$$

This formulation enables the application of standard convergence results for policy evaluation (Sutton et al., 1999). 

# A.3. Policy Improvement

For the current value function $Q ^ { k } ( s , a )$ and $\pi _ { k } ( a | s )$ , it holds that: 

$$
\pi_ {k + 1} (a | s) = \arg \max _ {\pi (\cdot | s)} \left\{\int \pi (a | s) Q ^ {\pi_ {k}} (s, a) d a - \eta \mathrm{KL} \bigl (\pi (a | s) \| \pi_ {k} (a | s) \bigr) \right\}. \tag {25}
$$

There is: 

$$
\int \pi_ {k + 1} (a | s) Q ^ {\pi_ {k}} (s, a) d a - \eta \mathrm{KL} \left(\pi_ {k + 1} (a | s) \| \pi_ {k} (a | s)\right) \geq \int \pi_ {k} (a | s) Q ^ {\pi_ {k}} (s, a) d a - \eta \mathrm{KL} \left(\pi_ {k} (a | s) \| \pi_ {k} (a | s)\right) \tag {26}
$$

$$
= \int \pi_ {k} (a | s) Q ^ {\pi_ {k}} (s, a) d a = V ^ {\pi_ {k}} (s). \tag {27}
$$

Since the KL term is non-negative, it follows that 

$$
\int \pi_ {k + 1} (a | s) Q ^ {\pi_ {k}} (s, a) d a \geq V ^ {\pi_ {k}} (s). \tag {28}
$$

Using the Bellman equation and applying this inequality recursively yields: 

$$
Q ^ {\pi_ {k}} (s, a) = r _ {0} + \mathbb {E} [ \gamma V ^ {\pi_ {k}} (s) ] \tag {29}
$$

$$
\leq r _ {0} + \mathbb {E} \left[ \gamma \mathbb {E} _ {a _ {1} \sim \pi_ {k + 1}} \left(Q ^ {\pi_ {k}} \left(s _ {1}, a _ {1}\right)\right) \right] \tag {30}
$$

$$
= r _ {0} + \mathbb {E} \left[ \gamma r _ {1} + \gamma^ {2} \mathbb {E} _ {a _ {2} \sim \pi_ {k} (\cdot | s _ {2})} \left[ Q ^ {\pi_ {k}} (s _ {2}, a _ {2}) \right] \right] \tag {31}
$$

$$
\leq r _ {0} + \mathbb {E} \left[ \gamma r _ {1} + \gamma^ {2} \mathbb {E} _ {a _ {2} \sim \pi_ {k + 1} (\cdot | s _ {2})} \left[ Q ^ {\pi_ {k}} (s _ {2}, a _ {2}) \right] \right] \tag {32}
$$

$$
= \mathbb {E} _ {s \sim d ^ {\pi_ {k + 1}}, a \sim \pi_ {k + 1}} [ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r _ {t} ] = Q ^ {\pi_ {k + 1}} (s, a). \tag {33}
$$

By unrolling the Bellman inequality, we obtain a monotonic improvement of the value function. 

# A.4. Jensen Gap

Let x be a random variable with distribution $p ( x )$ . For a function f that is convex or non-convex, the Jensen gap is defined as: 

$$
\mathcal {J} (f, x \sim p (x)) = \mathbb {E} [ f (x) ] - f (\mathbb {E} [ x ]) \tag {34}
$$

If f has L-smooth, then for x, $y \in \mathbb R$ 

$$
f (x) \leq f (y) + \nabla f (y) ^ {\top} (x - y) + \frac {L}{2} \| x - y \| ^ {2} \tag {35}
$$

Set $y = \mathbb { E } [ x ] \colon$ : 

$$
f (x) \leq f (\mathbb {E} [ x ]) + \nabla f (\mathbb {E} [ x ]) ^ {\top} (x - \mathbb {E} [ x ]) + \frac {L}{2} \| x - \mathbb {E} [ x ] \| ^ {2} \tag {36}
$$

$$
\mathbb {E} [ f (x) ] \leq f (\mathbb {E} [ x ]) + \nabla f (\mathbb {E} [ x ]) ^ {\top} \mathbb {E} [ x - \mathbb {E} [ x ] ] + \frac {L}{2} \mathbb {E} \| x - \mathbb {E} [ x ] \| ^ {2} \tag {37}
$$

$$
\mathbb {E} [ f (x) ] \leq f (\mathbb {E} [ x ]) + \frac {L}{2} \mathbb {E} \| x - \mathbb {E} [ x ] \| ^ {2} \tag {38}
$$

$$
\text { Jensen   Gap } = \mathbb {E} [ f (x) ] - f (\mathbb {E} [ x ]) \leq \frac {L}{2} \mathbb {E} \| x - \mathbb {E} [ x ] \| ^ {2} \leq \frac {L}{2} \operatorname{tr} (\Sigma) \tag {39}
$$

# A.5. Sampling Actions with Critic Guidance

Fixing s and denote $\begin{array} { r } { Q ( s , a ) : = \frac 1 \eta f ( a ) : \mathbb { R } ^ { d }  \mathbb { R } } \end{array}$ , and the target distribution is as followed: 

$$
\pi_ {0} ^ {*} (a _ {0}) := \frac {1}{Z _ {0}} \pi_ {0} (a _ {0}) \exp (f (a _ {0})), \quad Z _ {0} = \int \pi_ {0} (a _ {0}) \exp (f (a _ {0})) d a _ {0}. \tag {40}
$$

The induced target marginal at noise level t is 

$$
\pi_ {t} ^ {*} \left(a _ {t}\right) := \int q _ {t} \left(a _ {t} \mid a _ {0}\right) \pi_ {0} ^ {*} \left(a _ {0}\right) d a _ {0} = \frac {1}{Z _ {0}} \int q _ {t} \left(a _ {t} \mid a _ {0}\right) \pi_ {0} \left(a _ {0}\right) e ^ {f \left(a _ {0}\right)} d a _ {0} \tag {41}
$$

$$
= \frac {1}{Z _ {0}} \pi_ {t} (a _ {t}) \int \frac {q _ {t} (a _ {t} \mid a _ {0}) \pi_ {0} (a _ {0})}{\pi_ {t} (a _ {t})} e ^ {f (a _ {0})} d a _ {0} \tag {42}
$$

$$
= \frac {1}{Z _ {0}} \pi_ {t} (a _ {t}) \int \pi (a _ {0} | a _ {t}) e ^ {f (a _ {0})} d a _ {0} \tag {43}
$$

$$
= \frac {1}{Z _ {0}} \pi_ {t} (a _ {t}) \mathbb {E} _ {\pi (a _ {0} | a _ {t})} \left[ e ^ {f (a _ {0})} \right]. \tag {44}
$$

Hence, the guided score can be approximated as 

$$
\nabla_ {a _ {t}} \log \pi_ {t} ^ {*} (a _ {t}) = \nabla_ {a _ {t}} \log \pi_ {t} (a _ {t}) + \nabla_ {a _ {t}} \log \mathbb {E} _ {\pi (a _ {0} | a _ {t})} \left[ e ^ {f (a _ {0})} \right] \tag {45}
$$

$$
\approx \nabla_ {a _ {t}} \log \pi_ {t} (a _ {t}) + \nabla_ {a _ {t}} \mathbb {E} _ {\pi (a _ {0} | a _ {t})} [ f (a _ {0}) ] \tag {46}
$$

$$
\approx \nabla_ {a _ {t}} \log \pi_ {t} (a _ {t}) + \nabla_ {a _ {t}} f (\mathbb {E} _ {\pi (a _ {0} | a _ {t})} [ a _ {0} ]) \tag {47}
$$

$$
= s _ {\theta} (a _ {t}) + \nabla_ {a _ {t}} Q (s, \hat {a} _ {0} (a _ {t})), \tag {48}
$$

where $s _ { \theta } ( a _ { t } )$ is the score of the diffusion model and $\hat { a } _ { 0 } ( a _ { t } )$ is the posterior estimation via tweedie’s formula. 

# B. More Details on Practical Implementation

# B.1. Diffusion Guidance with Spherical Gaussian Algorithm

This appendix provides the detailed pseudocode for the value-guided refinement operator used by CGPO to synthesize a single refined target action $a ^ { g } ( s )$ for each training state s. The refinement runs a reverse diffusion chain initialized from 

Gaussian noise and injects a critic-based guidance signal only in the final G denoising steps. This design keeps early denoising steps unguided (when $\scriptstyle { \hat { x } } _ { 0 }$ is still highly noisy and critic signals are less reliable) while concentrating refinement in late steps where action-level changes are semantically meaningful and sampling-based improvement is most prone to plateau.The guidance step follows DSG (Yang et al., 2024), which constrains the perturbation to the typical-radius spherical shell of the reverse Gaussian transition, mitigating manifold deviation and preserving stochasticity through a convex mixture with the unconditional Gaussian perturbation. 

Algorithm 2 uses the standard DDPM ϵ-parameterization to compute $\hat { x } _ { 0 } ( x _ { t } , s , t )$ and the reverse mean $\mu _ { \theta } ( x _ { t } , s , t )$ . When $t > G$ , the update reduces to the unconditional reverse transition. When $t \leq G ,$ , the algorithm computes a guidance direction from the critic evaluated at the clean prediction $\scriptstyle { \hat { x } } _ { 0 }$ and applies the DSG constrained update. The refined target is returned as the final clean prediction produced by the reverse chain. 


Algorithm 2 Value-Guided Refinement with DSG


Input: State s, critic $\hat{Q}_{\psi}$ , guidance step G, guidance rate $\rho$ , noise scale $\sigma_{t}$ .

Output: Refined action $a^{g}(s)$ .

Initialize $x_{T} \sim \mathcal{N}(0, \mathbf{I})$ .

for $t = T, T - 1, \ldots, 1$ do $\hat{x}_{0} \leftarrow \frac{1}{\sqrt{\bar{\alpha}_{t}}} \left( x_{t} - \sqrt{1 - \bar{\alpha}_{t}} \epsilon_{\theta}(x_{t}, s, t) \right)$ $\mu_{\theta}(x_{t}, t) \leftarrow \frac{\sqrt{\alpha_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} x_{t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}}{1 - \bar{\alpha}_{t}} \hat{x}_{0}$ if t > G then

Sample $\epsilon_{t} \sim \mathcal{N}(0, \mathbf{I})$ and update $x_{t-1} \leftarrow \mu_{\theta}(x_{t}, t) + \sigma_{t} \epsilon_{t}$ .

else

Compute guidance gradient: $g_{t} = \nabla_{x_{t}} - \hat{Q}(s, \hat{x}_{0}(x_{t}, s, t))$ .

Determine constrained direction: $d^{\star} \leftarrow -\sqrt{d} \sigma_{t} \frac{g_{t}}{\|g_{t}\|_{2} + \varepsilon} \quad \{Steepest-descent on sphere\}$ Blend with Gaussian perturbation:

Sample $\epsilon_{t} \sim \mathcal{N}(0, \mathbf{I})$ and compute mixed direction: $d_{m} \leftarrow \sigma_{t} \epsilon_{t} + \rho (d^{\star} - \sigma_{t} \epsilon_{t})$ Spherical projection and update: $x_{t-1} \leftarrow \mu_{\theta}(x_{t}, t) + \sqrt{d} \sigma_{t} \frac{d_{m}}{\|d_{m}\|_{2}}$ end if

end for

return $a^{g}(s) = \hat{x}_{0}$ at t = 0. 

# B.2. Real-World RL System Settings and Training Details

We implemented the real-world RL system on a Franka Emika Panda arm with a Robotiq 2F-85 gripper, controlled via the Deoxys (Zhu et al., 2022) interface within the HIL-SERL (Luo et al., 2024) framework. Specifically, a 3Dconnexion SpaceMouse is integrated into the system to enable precise teleoperation. Within the HIL-SERL (Luo et al., 2024) framework, this interface allows the human operator to seamlessly switch between collecting initial expert demonstrations and applying corrective interventions to guide the agent. As shown in Figure 6, our visual setup integrates an eye-in-hand ZED Mini camera $( 6 7 2 \times 3 7 6$ resolution) and a fixed third-person RealSense D435i (640 × 480 resolution). The policy input (state space) combines these multi-view RGB images with proprioceptive statesspecifically, the end-effector Cartesian position, velocity, and gripper position. The policy outputs actions in the form of end-effector Cartesian position commands. The SAC and CGPO policy take approximately 1 hours to converge. Notably, the human intervention time for both methods is approximately 20 minutes. 

# C. Hyper-parameters

# C.1. MuJoCo Gym

All experiments are conducted on a single NVIDIA GeForce RTX 4090D GPU (24GB) with an Intel(R) Core(TM) i9- 14900K CPU. We implement SAC, TD3, PPO, SPO, and DIPO by building on their public reference codebases and keep the overall training and evaluation protocol consistent across methods. The complete hyper-parameter configurations for all compared algorithms are summarized in Table 4, while CGPO-specific settings are reported separately in Table 5. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/3f82249e5d713d3a906cb6da52cf2a5a74668261f48a480eb13342f09e165a67.jpg)



Figure 6. Experimental setup with the Franka Emika Panda robot and Robotiq 2F-85 gripper.



Table 3. Hyperparameters for the real-world experiments.


<table><tr><td>Name</td><td>Value</td><td>Note</td></tr><tr><td>max_traj_length</td><td>100</td><td>Episode length limit for training</td></tr><tr><td>batch_size</td><td>256</td><td>Mini-batch size</td></tr><tr><td>cta_ratio</td><td>2</td><td>Collect-to-update ratio</td></tr><tr><td>discount</td><td>0.98</td><td><eq>\gamma</eq></td></tr><tr><td>replay_buffer_capacity</td><td>20000</td><td>Replay buffer size</td></tr><tr><td>steps_per_update</td><td>50</td><td>Actor network parameters update steps</td></tr><tr><td>encoder_type</td><td>resnet10-pretrained</td><td>Visual encoder type</td></tr></table>

For CGPO, we intentionally avoid environment-by-environment tuning and use a single task-invariant configuration across all MuJoCo v3 tasks $( \operatorname { A n t } .$ , HalfCheetah, Hopper, Humanoid, and Walker2d), including the sampling budget $N _ { e } ,$ , selection counts $( K _ { b } , K _ { t } )$ , guidance parameters $( G , \rho )$ , and entropy weight $\lambda _ { \mathrm { e n t } }$ (Table 5), to reduce reproduction overhead and isolate algorithmic effects. 

# C.2. Real-World Symtem

Hyperparameters used for the real-world experiments are shown in Table 3. 


Table 4. Hyperparameters used in the experiments.


<table><tr><td></td><td>CGPO</td><td>QVPO</td><td>DIPO</td><td>SAC</td><td>TD3</td><td>SPO</td><td>PPO</td><td>DACER</td></tr><tr><td colspan="9">Network</td></tr><tr><td>Hidden layers</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td></tr><tr><td>Hidden width</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td><td>64</td><td>256</td><td>256</td></tr><tr><td>Activation</td><td>mish</td><td>mish</td><td>mish</td><td>relu</td><td>relu</td><td>tanh</td><td>tanh</td><td>mish</td></tr><tr><td colspan="9">Optimization</td></tr><tr><td>Batch size</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Discount <eq>\gamma</eq></td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>Target smoothing <eq>\tau</eq></td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.0</td><td>0.005</td><td>0.005</td></tr><tr><td>Actor lr <eq>\eta_{\pi}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>7 \times 10^{-4}</eq></td><td><eq>1 \times 10^{-4}</eq></td></tr><tr><td>Critic lr <eq>\eta_Q</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td><td><eq>7 \times 10^{-4}</eq><eq>1 \times 10^{-4}</eq></td><td><eq>1 \times 10^{-4}</eq></td></tr><tr><td>Grad norm clip</td><td>N/A</td><td>N/A</td><td>2</td><td>N/A</td><td>N/A</td><td>0.5</td><td>0.5</td><td>N/A</td></tr><tr><td>Replay buffer size</td><td><eq>10^6</eq></td><td><eq>10^6</eq></td><td><eq>10^6</eq></td><td><eq>10^6</eq></td><td><eq>10^6</eq></td><td><eq>10^6</eq></td><td><eq>10^6</eq></td><td><eq>10^6</eq></td></tr><tr><td colspan="9">Exploration / Regularization</td></tr><tr><td>Entropy coef.</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.2</td><td>N/A</td><td>N/A</td><td>0.01</td><td>learned <eq>\alpha</eq></td></tr><tr><td>Value loss coef.</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.5</td><td>N/A</td></tr><tr><td>Exploration noise</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td><eq>\mathcal{N}(0,0.1)</eq></td><td>N/A</td><td>N/A</td><td><eq>\mathcal{N}(0,(0.15\alpha)^2)</eq></td></tr><tr><td>Policy noise</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td><eq>\mathcal{N}(0,0.2)</eq></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Noise clip</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.5</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Use GAE</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>True</td><td>True</td><td>N/A</td></tr><tr><td colspan="9">Diffusion</td></tr><tr><td>Diffusion steps T</td><td>20</td><td>20</td><td>20</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>20</td></tr></table>

# D. Additional Results

# D.1. Parameter Analysis

To further understand the role of training-time refinement in CGPO, we conduct a parameter analysis on two key hyperparameters of CGPO: the guidance rate $\rho$ and the guidance step G (number of guided denoising steps). We report results on Ant-v3 as a representative example; similar trends are observed across other locomotion tasks. 

Effect of guidance rate $\rho _ { \bullet }$ The left panel of Figure 7 compares different guidance rates $( \rho \in \{ 0 . 3 5 , 0 . 6 5 , 0 . 9 5 \} )$ ). We observe that stronger guidance generally leads to better asymptotic performance, with $\rho = 0 . 9 5$ achieving the highest final return, while smaller $\rho$ values converge to lower plateaus. Intuitively, increasing ρ places more emphasis on the critic-induced refinement direction in DSG, thereby producing higher-quality refined targets for the subsequent denoising regression update. In practice, we find that a relatively large $\rho$ is beneficial as long as guidance is restricted to the late denoising regime. 

Effect of guidance step G. The right panel of Figure 7 studies the number of guided denoising steps $( G \in \{ 5 , 1 0 , 1 5 \} )$ with the same total diffusion steps. A moderate step (G = 10) performs best, while using too few guided steps (G = 5) yields weaker gains, consistent with insufficient refinement strength. Conversely, pushing guidance too early (G = 15) can slightly degrade performance, which is consistent with the RL-specific challenge that early-step denoising variables (and their predicted clean actions) are more noise-dominated, making critic-based guidance less reliable and more prone to distribution shift. Overall, these results support the design choice of applying guidance only in the final denoising steps. 

# D.2. Additional Ablation Study

To further isolate the contribution of each component, we conduct additional ablations on Walker2d-v3, as shown in Figure 8. The left panel compares DSG guidance with unguided target generation and naive guidance. Removing guidance leads to weaker learning, while naive guidance improves over the unguided variant but still underperforms full CGPO, indicating that the form of guidance is important. This supports the use of DSG as a structured guidance rule that is better aligned with the diffusion reverse process. 


Table 5. CGPO hyper-parameters on MuJoCo v3 tasks.


<table><tr><td></td><td>Ant-v3</td><td>HalfCheetah-v3</td><td>Hopper-v3</td><td>Humanoid-v3</td><td>Walker2d-v3</td></tr><tr><td colspan="6">Sampling</td></tr><tr><td># uniform samples <eq>N_e</eq> from <eq>\mathcal{U}(\underline{a},\overline{a})</eq></td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td colspan="6">Action selection</td></tr><tr><td>Behavior selection count <eq>K_b</eq></td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Target selection count <eq>K_t</eq></td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td colspan="6">Guidance</td></tr><tr><td>Guidance step G</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Guidance rate ρ</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td colspan="6">Regularization</td></tr><tr><td>Entropy weight <eq>λ_{ent}</eq></td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/79f4aedb648d55461f45d33237656812b0dd37f1e1e802eb9540caf4e353720c.jpg)



(a) Effect of guidance rate ρ


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/f375a7b177305908e1b41c9c4f3a9f010dc73edd4a175e8e002d78b1a6962ac7.jpg)



(b) Effect of guidance step G



Figure 7. Parameter analysis of CGPO on Ant-v3


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/e0d1abe9eb9bffdc44b2435b249137fc386d56a7d91f046c591612eb590a4b1f.jpg)



(a) Guidance variants


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/8a95c99213d15f739cd4f2fcd9a03321a9bc075cd250e0711ed6ec2c5b202dbc.jpg)



(b) w/o DDQN


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/da552847f49db132060df199bd9ac18ea29c33e4d379b761bc4fdd789b6c7077.jpg)



(c) w/o TQC


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/d88dc2c509c2701c81964e99fd73f9054606b8ebc8386f51bdf7cb82e8f56add.jpg)



(d) w/o V-net



Figure 8. Component ablations of CGPO on Walker2d-v3.


The middle two panels evaluate the effect of DDQN target construction and truncated-quantile aggregation. Removing either component degrades performance, suggesting that overestimation control and stable scalar Q aggregation are important for both target generation and actor weighting. The right panel removes the base value network. The resulting performance drop confirms that anchoring the scale of critic-derived weights to unguided actor samples helps stabilize state reweighting. Overall, these results are consistent with the Ant-v3 analysis and verify that CGPO benefits from both guided target synthesis and calibrated critic-weighted actor learning. 

# D.3. Quantitative Analysis of Critic Contrast

To further validate the mechanism discussed in Section 4.1, we analyze the evolution of the critic contrast $\Delta _ { Q } ( s )$ during training. Recall that in Equation (9), $\Delta _ { Q } ( s )$ measures the value gap between the best sampled candidate and the average value of the finite candidate set. A larger $\Delta _ { Q } ( s )$ indicates that the critic can clearly distinguish a better action from typical samples, whereas a smaller value suggests that sampling-based improvement provides only a weak update signal. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/bb992e5b-407a-40fb-a1f6-6f9650066cc4/1bca3e5194c89da11d3e247e2b85a4c21791d59fa84859a43eee80001aae7b16.jpg)



Figure 9. Evolution of critic contrast $\Delta _ { Q } ( s )$ on Ant-v3. For the sampling-based variant, we sample $K = 6 4$ candidate actions and compute $\Delta _ { Q } ( s )$ according to Equation (9). For CGPO, we compute the value gap induced by the DSG-guided action target. Sampling-based improvement exhibits a large value gap early in training but gradually loses contrast as training progresses, while CGPO maintains a more stable improvement signal.


As shown in Figure 9, the sampling-based method obtains a relatively large $\Delta _ { Q } ( s )$ in the early stage, indicating that finite candidate sampling can initially discover actions with noticeably higher critic values. However, as training proceeds and the policy distribution becomes more concentrated, the value gap decreases and becomes more fluctuating. This supports the analysis in Section 4.1: when sampled candidates become similar, the critic values within the candidate set become less separable, and the effective improvement signal weakens. 

In contrast, CGPO variant maintains a more stable value gap in the later stage of training. This suggests that critic-guided target synthesis can provide a more persistent improvement signal after sampling-based candidate selection begins to lose discriminative power. Together with the learning-curve results, this quantitative analysis supports the central design motivation of CGPO: replacing finite candidate search with training-time critic-guided target synthesis can alleviate the weakening of sampling-driven policy improvement. 