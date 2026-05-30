# Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models

Jaa-Yeon Lee * 1 Yeobin Hong * 1 Taesung Kwon 1 Jong Chul Ye 1 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/28249a6b529a517d0b44d7c1fc06dbc2338c7fb3940cdac591168fc1837ef625.jpg)



"A park bench that has a teddy bear on it."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/1e05288c966f9638a52d31ccd900bd9a076f2db95aa40596749659b3f217dabb.jpg)



"A man standing on a rock looking at the sky."→".. with robot boat."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/6eb16c4f52efaa1af19ff364eed4b67a2eba63016a670d97d0d3a6dec2075bec.jpg)



"orange fire hydrant with a face and bowtie..."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/018a8e62e8ce6e6d2f591f2eb961a7a0084a95b29cb7ad93a73faf2051a21abe.jpg)



"A white vase with purple tulips on a pink background."→ "A cat and a..."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/daffa0fe24c09f70c1530284fe4e7f9541d0654630aa619703672d6ef5e8d95f.jpg)



".. tll lock tower next to a tallglass building."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/f13d789f31fa578cf64fc2d89fa592cc5713a8777c48a333821fcf68a1fed77b.jpg)



"The arctic sky."→“A castle in the arctic sky."



Figure 1. Representative results for text-to-image generation and image editing. Our alignment-guided fine-tuning improves semantic consistency between text and image by training soft tokens within the score-matching framework.


# Abstract

Diffusion models generate highly realistic images but often struggle with precise text–image alignment. While recent post-training methods improve alignment using external rewards or human preference signals, their performance heavily depends on reward quality and does not directly address alignment within the diffusion process itself. Recent reward-free approaches such as SoftREPA demonstrate that optimizing soft text tokens via 

contrastive learning can effectively improve textimage representation alignment, outperforming standard parameter-efficient fine-tuning baselines. However, the contrastive formulation can excessively penalize negative pairs, which manifests as characteristic failure cases such as overcounting and repetition. To address this issue, we propose a lightweight, reward-free post-training method that refines soft tokens by integrating contrastive alignment guidance directly into the score-matching objective of diffusion models. By assigning alignment directions at the score level, our approach mitigates these limitations and yields more coherent and semantically faithful generations. Experiments show that our method matches SoftREPA while substantially improving its failure cases, achieving over 35% improvement in counting accuracy on the GenEval benchmark. Our method is seamlessly applicable to existing diffusion backbones (SD1.5, SDXL, and SD3), and is complementary to existing RL-based diffusion post-training methods. Project page: https://jaayeon.github.io/AGSM/ 

# 1. Introduction

Diffusion models have achieved remarkable progress in high-fidelity image generation (Peebles & Xie, 2023; Esser et al., 2024; Podell et al., 2023). To further align their generative behavior with desired outcomes, recent work has explored post-training techniques such as policy gradient and preference optimization (Black et al., 2023; Xu et al., 2023; Clark et al., 2023; Fan et al., 2023; Wallace et al., 2024). However, most existing approaches rely on human preference annotations (Wallace et al., 2024; Karthik et al., 2024; Zhu et al., 2025; Liang et al., 2025) or externally designed reward models (Fan et al., 2023; Black et al., 2023; Xu et al., 2023; Clark et al., 2023). As a result, their effectiveness depends critically on reward quality and data availability, leaving the intrinsic text–image alignment signals within diffusion models underexplored. 

Recent studies have begun to revisit text–image alignment by leveraging diffusion models’ internal representations and score-matching dynamics (Xian et al., 2025; Lee et al., 2025a). In particular, SoftREPA (Lee et al., 2025a) demonstrated the potential of optimizing lightweight soft text tokens to maximize the mutual information between modalities by leveraging the diffusion score-matching loss as a proxy for alignment. However, we identify a fundamental instability in this contrastive formulation: while it minimizes score-matching loss for positive pairs, it simultaneously maximizes it for negative pairs. This adversarial pushing often forces the soft tokens to represent off-manifold regions, manifesting in characteristic failure cases such as object repetition, over-counting, and semantic incoherence. 

In parallel, recent advances in diffusion-based preference optimization provide a promising direction. Diffusion-DPO (Direct Preference Optimization) (Wallace et al., 2024) formulates preference alignment within the diffusion objective using the Bradley–Terry model (Bradley & Terry, 1952). DSPO (Direct Score Preference Optimization) (Zhu et al., 2025) further integrates preference learning into the scorematching framework. These approaches indicate that modeling preferences at the score level enables stable alignment while preserving the underlying diffusion dynamics. 

Building on these insights, we propose Alignment-Guided Score Matching, a reward-free post-training framework optimizing soft tokens that addresses contrastive instability by explicitly guiding both positive and negative text–image pairs within the score-matching objective. We formulate text–image alignment as preference learning under a Plackett–Luce (PL) model (Luce et al., 1959), where alignment preferences are derived from the diffusion model’s intrinsic log-likelihood without external rewards. Unlike prior approaches that penalize negative pairs implicitly, our method assigns explicit score-level guidance using separate soft tokens $( \psi ^ { + } , \psi ^ { - } )$ for positive and negative semantic regions, preventing off-manifold drift and preserving generative fidelity. 

Our main contributions are as follows: 

• Reward-free Plackett-Luce Formulation: We formulate text-image alignment as reward fine-tuning over diffusion scores. Employing a PL model, we enable a reward-free post-training objective that leverages the model’s internal priors. 

• Stability via Explicit Negative Guidance: We mitigate the unbounded divergence of prior contrastive loss by assigning explicit, bounded preference directions to negative samples, preventing the failure cases of SoftREPA. 

• Efficiency and Versatility: The proposed approach is lightweight, model-agnostic, and complementary to existing RL-based diffusion post-training methods. 

# 2. Preliminaries

SoftREPA SoftREPA (Lee et al., 2025a) fine-tunes diffusion models by aligning text and image representations through contrastive learning. Given a soft text token s and a paired sample (x, c), the similarity score is defined as 

$$
\tilde {\ell} (\boldsymbol {x}, \boldsymbol {c}, \boldsymbol {s}) = \exp \left(- \mathbb {E} _ {t, \epsilon} \left[ \frac {\| \epsilon_ {\theta} (\boldsymbol {x} _ {t} , t , \boldsymbol {c} , \boldsymbol {s}) - \epsilon_ {t} \| _ {2} ^ {2}}{\tau (t)} \right]\right), \tag {1}
$$

where $\epsilon _ { \theta }$ is the noise prediction of the diffusion model, τ (t) denotes a temperature-scaled time weighting, and $\epsilon _ { t }$ is the Gaussian noise at step t. In practice, SoftREPA approximates the expectation with a Monte Carlo estimate using sampled t and ϵ during training. The soft-token training objective adopts a contrastive form: 

$$
\mathcal{L}(\boldsymbol {s}) = -\mathbb{E}_{\substack{(\boldsymbol {x}, \boldsymbol {c})\sim p_{\text{data}},\\ t\sim U(0,1),\\ \boldsymbol {\epsilon}\sim \mathcal{N}(\boldsymbol {0},\boldsymbol {I})}}\log \frac{\exp\left(\tilde{\ell}(\boldsymbol{x},\boldsymbol{c},\boldsymbol{s})\right)}{\sum_{j}\exp\left(\tilde{\ell}(\boldsymbol{x},\boldsymbol{c}^{j},\boldsymbol{s})\right)}, \tag{2}
$$

where $c ^ { j }$ denotes negative text pairs within the minibatch. This objective optimizes the soft token s to maximize text–image representation alignment by increasing the mutual information between the two modalities. 

DSPO Direct Score Preference Optimization (DSPO) (Zhu et al., 2025) fine-tunes diffusion models by directly incorporating human preference signals into the score-matching framework. Given a text-conditioned diffusion model $p _ { \theta } ( \pmb { x } _ { t } | \pmb { c } )$ and a preference pair $( \pmb { x } _ { t } , \pmb { x } _ { t } ^ { l } , \pmb { c } )$ , human preference is modeled by the Bradley–Terry formulation (Bradley & Terry, 1952): 

$$
p (\pmb {y} | \pmb {x} _ {t}, \pmb {c}) = \sigma \big (r (\pmb {x} _ {t}, \pmb {c}) - r (\pmb {x} _ {t} ^ {l}, \pmb {c}) \big), \qquad (3)
$$

where $p ( \pmb { y } | \pmb { x } _ { t } , \pmb { c } )$ denotes the probability that $( \boldsymbol { x } _ { t } , \boldsymbol { c } )$ is preferred to $( \pmb { x } _ { t } ^ { l } , \pmb { c } )$ and $r ( \pmb { x } _ { t } , \pmb { c } )$ is an implicit reward estimated from DiffusionDPO (Wallace et al., 2024). The DSPO objective aligns the diffusion score with the human-preferred score, $\nabla _ { \pmb { x } _ { t } }$ log $\mathbf { \nabla } _ { p } ( \pmb { x } _ { t } | \mathbf { c } , \pmb { y } )$ using Bayes’ rule, as 

$$
\min _ {\theta} \| \nabla \log p _ {\theta} (\boldsymbol {x} _ {t} | \boldsymbol {c}) - (\nabla \log p (\boldsymbol {x} _ {t} | \boldsymbol {c}) + \gamma \nabla \log p (\boldsymbol {y} | \boldsymbol {x} _ {t}, \boldsymbol {c}) \| _ {2} ^ {2} \tag {4}
$$

where $\gamma$ controls the preference strength. The implicit reward can be expressed as a log-density ratio between the current and reference models: 

$$
r (\boldsymbol {x} _ {t}, \boldsymbol {c}) = \lambda_ {t} \log \frac {p _ {\theta} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {c})}{p _ {\mathrm{ref}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {c})}. \tag {5}
$$

Combining Eq. (3) and Eq. (5) into Eq. (4), DSPO training objective becomes: 

$$
\min _ {\theta} \left\| \boldsymbol {\epsilon} _ {\theta , t} - \boldsymbol {\epsilon} _ {t} - \lambda_ {t} w (\boldsymbol {x} _ {t}, \boldsymbol {x} _ {t} ^ {l}, \boldsymbol {c}) (\boldsymbol {\epsilon} _ {\theta , t} - \boldsymbol {\epsilon} _ {\mathrm{ref}, t}) \right\| _ {2} ^ {2}, \tag {6}
$$

where $w ( \pmb { x } _ { t } , \pmb { x } _ { t } ^ { l } , \pmb { c } ) = 1 - \sigma \big ( r ( \pmb { x } _ { t } , \pmb { c } ) - r ( \pmb { x } _ { t } ^ { l } , \pmb { c } ) \big )$ is a preference score-based weighting term that modulates the guidance induced by the discrepancy between the online and reference scores. 

# 3. Alignment-Guided Score Matching

Since the similarity score in Eq. (1) is a strictly decreasing function, minimizing $\operatorname { E q . } \left( 2 \right)$ may lead to increasing the score matching loss $\| \epsilon _ { \theta } ( \pmb { x } _ { t } , t , \pmb { c } ^ { j } , \pmb { s } ) - \epsilon _ { t } \| _ { 2 } ^ { 2 }$ for negative pairs $c ^ { j }$ . Therefore, the SoftREPA objective does not constrain negative pairs to remain on the diffusion manifold, allowing unbounded divergence of the denoising error. 

To circumvent the instabilities of contrastive pushing, we propose a guidance-based framework that treats text alignment as a bounded preference optimization problem. Our approach consists of three components: (i) a normalized alignment reward derived via a Plackett-Luce formulation (Section 3.1), (ii) a modified score-matching objective that transforms the target score for both positive and negative pairs (Section 3.2), and (iii) a dual-token training scheme that explicitly separates positive and negative guidance with stability analysis (Section 3.3). 

# 3.1. Alignment Reward via Plackett–Luce Modeling

To define the probability that an image $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is aligned with a specific text c among a set of candidates $\{ c ^ { i } \}$ , we employ the Plackett-Luce (PL) model (Luce et al., 1959). This formulation generalizes the pairwise Bradley-Terry model used in Eq. (3) to a multi-class preference framework: 

$$
p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}) = \frac {\exp (r (\boldsymbol {x} _ {t} , \boldsymbol {c}))}{\sum_ {i} \exp (r (\boldsymbol {x} _ {t} , \boldsymbol {c} ^ {i}))}, \tag {7}
$$

where z serves as a binary random variable: $z { = } 1$ indicates the pair $( \boldsymbol { \boldsymbol { x } } _ { t } , \boldsymbol { \boldsymbol { c } } )$ is aligned and $z { = } 0$ indicates opposite. 

Inspired by SoftREPA (Lee et al., 2025a), we define an implicit alignment reward as the expected conditional loglikelihood of the model reverse transition under DDPM (Ho et al., 2020) posterior: 

$$
r (\boldsymbol {x} _ {t}, \boldsymbol {c}) := \lambda_ {t} \mathbb {E} _ {q (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0})} [ \log p _ {\theta} (\boldsymbol {x} _ {t - 1} \mid \boldsymbol {x} _ {t}, \boldsymbol {c}) ], (8)
$$

where $\lambda _ { t }$ controls the reward scale. Thus, higher reward is assigned to text–image pairs whose reverse transition is better predicted under condition c, without requiring an external reward model. 

# 3.2. Alignment-Guided Score Matching

Target Distribution. To maximize the alignment of the joint pdf of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and c, we divide data into positive and negative subsets $( \mathcal { D } ^ { + } , \mathcal { D } ^ { - } )$ using a binary random variable z. $\mathcal { D } ^ { + }$ consists of aligned text–image pairs (z=1), whereas $\mathcal { D } ^ { - }$ consists of mismatched pairs $( z { = } 0 )$ . The goal is to increase the probability in aligned regions while suppressing it in the mismatched regions through explicit score modification. 

The corresponding tilted target conditional distributions are defined as 

$$
\begin{array}{l} p _ {t} ^ {+} (\boldsymbol {x} _ {t} | \boldsymbol {c}) := p _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c}, z = 1) \propto p _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c})   p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}) ^ {\gamma^ {+}}, \\ p _ {t} ^ {-} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}\right) := p _ {t} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}, z = 0\right) \propto p _ {t} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}\right) p (z = 1 \mid \boldsymbol {x} _ {t}, \boldsymbol {c}) ^ {- \gamma^ {-}} \tag {9} \\ \end{array}
$$

where $\gamma ^ { + }$ and $\gamma ^ { - }$ regulate the influence of alignment reward on the resulting posterior. This formulation is closely related to classifier-free guidance (CFG) (Ho & Salimans, 2022), which tilts the sampling distribution with weighted posterior probability, $p _ { \theta } ( { \pmb x } | { \pmb c } ) p _ { \theta } ( { \pmb c } | { \pmb x } _ { t } ) ^ { w }$ . Similarly, the negative branch acts as a repulsive tilting term analogous to negative prompting (Gandikota et al., 2023). The resulting inverse weighting serves as a direction-preserving surrogate for the Bayes-consistent negative guidance term $\scriptstyle p ( z = 0 | \mathbf { x } _ { t } , \mathbf { c } )$ , differing only by a positive scaling factor (Appendix A). 

A unified expression of the modified target distribution is 

$$
\tilde {p} _ {t} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}, z\right) := \mathbf {1} \{z = 1 \} p _ {t} ^ {+} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}\right) + \mathbf {1} \{z = 0 \} p _ {t} ^ {-} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}\right). \tag {10}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/f08fb7d5290f4332c0d04ae52fe0f7bf0c7b781ccded22863e6013057b52ccd5.jpg)



Figure 2. Alignment-Guided Score Matching improves text–image alignment by increasing alignment rewards for positive pairs and decreasing those for negative pairs. Noise predictions $\epsilon _ { \theta } ^ { + }$ and $\epsilon _ { \theta } ^ { - }$ are conditioned on positive and negative soft tokens $\mathbf { \dot { \Phi } } ( \psi ^ { + } , \psi ^ { - } )$ . Target noise is adjusted using alignment guidance derived from implicit-reward–weighted EMA predictions $( \hat { \epsilon } _ { \theta } ^ { + } , \hat { \epsilon } _ { \theta } ^ { - } )$ . For clarity, the figure shows a single negative pair, while multiple negatives are used during training.


Taking the gradient with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ gives the new target score: 

$$
\nabla \log \tilde {p} _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c}, z) = \nabla \log p _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c}) + \gamma_ {z} \nabla \log p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}), \tag {11}
$$

where $\gamma _ { z } = \gamma ^ { + } \mathbf { 1 } \{ z { = } 1 \} - \gamma ^ { - } \mathbf { 1 } \{ z { = } 0 \}$ . The first term corresponds to the standard diffusion score, while the second term explicitly pushes samples in the direction of higher reward gradients when it comes to positive pairs, pushes samples in the opposite direction when it comes to negative pairs, encouraging $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to move toward better-aligned image–text regions. 

Training Objective. We train learnable soft tokens $\psi _ { z }$ to match the new target score: 

$$
\min _ {\psi_ {z}} \mathbb {E} \left[ w (t) \| \nabla \log p _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t} | \boldsymbol {c}) - \nabla \log \tilde {p} _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c}, z) \| _ {2} ^ {2} \right], \tag {12}
$$

where $w ( t )$ is a time-dependent weighting function. We denote by $p _ { t , \theta } ^ { \psi _ { z } } ( { \pmb x } _ { t } | { \pmb c } )$ the diffusion model with frozen backbone θ conditioned on soft token $\psi _ { z }$ . 

To compute the gradient term in Eq. (11), we differentiate the PL likelihood in Eq. (7) with respect to $\mathbf { \nabla } _ { \mathbf { x } _ { t } : }$ 

$$
\nabla \log p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}) = \nabla r (\boldsymbol {x} _ {t}, \boldsymbol {c}) - \sum_ {i} w _ {i} \nabla r (\boldsymbol {x} _ {t}, \boldsymbol {c} ^ {i}), \tag {13}
$$

where wi = $\begin{array} { r } { w _ { i } = \frac { \exp \left( r \left( \mathbf { x } _ { t } , \mathbf { c } ^ { i } \right) \right) } { \sum _ { i } \exp \left( r \left( \mathbf { x } _ { t } , \mathbf { c } ^ { j } \right) \right) } } \end{array}$ represents the normalized reward weight over textual alternatives. This equation shows that the gradient of the PL likelihood compares the reward gradient of the current pair $( \boldsymbol { x } _ { t } , \boldsymbol { c } )$ against the weighted average of competing candidates, producing a contrast signal for alignment. 

Since diffusion models parameterize the reverse denoising process using a neural network as in DDPM (Ho et al., 

# Algorithm 1 Alignment-Guided Score Matching

Require: Dataset D; backbone $\theta ;$ soft tokens $\psi ^ { \pm }$ ; EMA soft tokens $\hat { \psi } ^ { \pm } ;$ ; guidance scales $\gamma ^ { \pm }$ 

1: while not converged do 

2: ▷ Sample positive and negative pairs 

3: $( \boldsymbol { x } _ { 0 } ^ { i } , \boldsymbol { c } ^ { \flat } ) _ { i , j = 1 } ^ { B ^ { \iota } } { \sim } \mathcal { D } \mathrm { \ } \{ \mathcal { D } ^ { + } \mathrm { \ i f \ } i = j , \mathcal { D } ^ { - } \mathrm { \ i f \ } i \neq j \}$ 

4: $\boldsymbol { i } { \sim } \boldsymbol { U } ( 0 , 1 ) , \boldsymbol { \epsilon } { \sim } \mathcal { N } ( 0 , \mathbf { I } )$ 

5: ▷ use same t and ϵ for all pairs√ √ 

6: $\pmb { x } _ { t } ^ { i } = \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } ^ { i } + \sqrt { 1 - \bar { \alpha } _ { t } } \hat { \epsilon } , \forall i$ 

7: (ϵ(i,j)pred , ϵ(i,j)ema $( \epsilon _ { \mathrm { p r e d } } ^ { ( i , j ) } , \epsilon _ { \mathrm { e m a } } ^ { ( i , j ) } ) \gets \left\{ ( \epsilon _ { t , \theta } ^ { \psi ^ { + } } ( \cdot ) , \epsilon _ { t , \theta } ^ { \hat { \psi } ^ { + } } ( \cdot ) ) , \ i = j \right.$ where $\mathbf { \Psi } ( \cdot ) = ( \mathbf { * } _ { t } ^ { i } , \mathbf { c } ^ { j } )$ 

8: $w _ { i , j } = \mathrm { S o f t m a x } _ { j } \big ( - \| \epsilon _ { \mathrm { e m a } } ^ { ( i , j ) } - \epsilon \| _ { 2 } ^ { 2 } \big )$ 

9: $\begin{array} { r } { \Delta _ { \hat { \psi } } ^ { ( i , j ) } = \epsilon _ { \mathrm { e m a } } ^ { ( i , j ) } - \sum _ { k } w _ { i , k } \epsilon _ { \mathrm { e m a } } ^ { ( i , k ) } } \end{array}$ ϵema k {Eq.13} 

10: $\mathbf { \Sigma } _ { \epsilon } ( i , j ) \mathbf { \epsilon } _ { - \epsilon \bot } \int + \gamma ^ { + } \tilde { A } ( t ) \Delta _ { \hat { \psi } } ^ { ( i , j ) } , \quad i = j$ $\epsilon _ { \mathrm { t g t } } ^ { ( \iota , \ j ) }  \epsilon + \{ \_ { - \gamma ^ { - } } \tilde { A } ( t ) \Delta _ { \hat { \psi } } ^ { ( \iota , \ j ) } , \ i \neq j \}$ ϵ(i,j)tgt ← ϵ + 

11: ▷ Update $\psi ^ { \pm }$ and $\hat { \psi } ^ { \pm }$ 

12: $\begin{array} { r } { \mathcal { L } ( \dot { \psi } ^ { \pm } ) \propto \sum _ { i , j } \big \| \dot { \epsilon } _ { \mathrm { p r e d } } ^ { ( i , j ) } - \epsilon _ { \mathrm { t g t } } ^ { ( i , j ) } \big \| _ { 2 } ^ { 2 } } \end{array}$ ϵpred ϵ tgt 

13: end while 

2020), we instantiate the alignment reward using denoising error: 

$$
r (\boldsymbol {x} _ {t}, \boldsymbol {c}) = - \frac {A (t)}{2} \| \epsilon_ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - \epsilon \| _ {2} ^ {2}, \tag {14}
$$

where $\begin{array} { r } { \alpha _ { t } { = } 1 { - } \beta _ { t } , \bar { \alpha } _ { t } { = } \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ , and $\begin{array} { r } { A ( t ) = \frac { \lambda _ { t } \beta _ { t } } { \alpha _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) } } \end{array}$ (see Appendix B for details). We compute the reward using EMA-updated soft tokens $( \hat { \psi } _ { z } )$ to stabilize the alignment signal. A lower denoising error corresponds to a higher reward, directly coupling text–image alignment with diffusion consistency. 

By substituting Eq. (11), Eq. (14), and Eq. (13) into the score-matching objective in Eq. (12), we obtain the final Alignment-Guided Score Matching loss: 

$$
\min _ {\psi_ {z}} \mathbb {E} \left[ \left\| \boldsymbol {\epsilon} _ {t, \theta} ^ {\psi_ {z}} - \left(\boldsymbol {\epsilon} _ {t} + \gamma_ {z} \tilde {A} (t) \left(\boldsymbol {\epsilon} _ {t, \theta} ^ {\hat {\psi} _ {z}} - \sum_ {i} w _ {i} \boldsymbol {\epsilon} _ {t, \theta} ^ {\hat {\psi} _ {z}, i}\right)\right) \right\| ^ {2} \right]. \tag {15}
$$

Here, ϵψz,t,θ $\epsilon _ { t , \theta } ^ { \hat { \psi } _ { z } , i }$ denotes the denoising prediction conditioned on the i-th text candidate $c ^ { i }$ . We omit the timestep weighting for brevity, with $\begin{array} { r } { \tilde { A } ( t ) = \frac { \lambda _ { t } \beta _ { t } \sqrt { 1 - \bar { \alpha } _ { t } } } { \alpha _ { t } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } } \end{array}$ A comprehensive derivation is provided in Appendix C. 

For the flow model, the Alignment-Guided Score Matching loss can be formulated as 

$$
\min _ {\psi_ {z}} \mathbb {E} \left[ \right.\left\| \right. v _ {t, \theta} ^ {\psi_ {z}} - \left( \right.v _ {t} + \gamma_ {z} B (t) \left(v _ {t, \theta} ^ {\hat {\psi} _ {z}} - \sum_ {i} w _ {i} v _ {t, \theta} ^ {\hat {\psi} _ {z}, i}\right)\left. \right\| _ {2} ^ {2} \left. \right], \tag {16}
$$

which closely resembles the alignment-guided loss for the diffusion model (Eq. (15)). A detailed derivation is given in Appendix D. 

# 3.3. Negative Sample Optimization and its Stability

Dual-Token Parameterization. To effectively capture both generative capability and alignment performance, we decouple the positive and negative alignment guidance through separate soft token optimization. Separate soft tokens $( \psi ^ { + }$ and $\psi ^ { - } )$ are updated for the corresponding positive and negative data pairs (Figure 2). Concretely, the resulting objective Eq. (15) can be rewritten as: 

$$
\begin{array}{l} \min _ {\psi^ {+}, \psi^ {-}} \Big (\underset {(\boldsymbol {x}, \boldsymbol {c}) \sim \mathcal {D} ^ {+}} {\mathbb {E}} \left[ \left\| \boldsymbol {\epsilon} _ {t, \theta} ^ {\psi^ {+}} - \big (\boldsymbol {\epsilon} _ {t} + \gamma^ {+} \tilde {A} (t)   \Delta_ {\hat {\psi} _ {z}} \big) \right\| _ {2} ^ {2} \right] \\ \left. + \underset {(\boldsymbol {x}, \boldsymbol {c}) \sim \mathcal {D} ^ {-}} {\mathbb {E}} \left[ \left\| \boldsymbol {\epsilon} _ {t, \theta} ^ {\psi^ {-}} - \left(\boldsymbol {\epsilon} _ {t} - \gamma^ {-} \tilde {A} (t) \Delta_ {\hat {\psi} _ {z}}\right) \right\| _ {2} ^ {2} \right]\right), \tag {17} \\ \end{array}
$$

where ∆ψˆz = ϵψˆzt,θ $\begin{array} { r } { \Delta _ { \hat { \psi } _ { z } } = \epsilon _ { t , \theta } ^ { \hat { \psi } _ { z } } - \sum _ { i } w _ { i } \epsilon _ { t , \theta } ^ { \hat { \psi } _ { z } , i } } \end{array}$ − P i w i ϵψˆz,it,θ . Decoupling the parameters provides a mechanism to partition the alignment and contrastive signals, allowing for targeted semantic refinement without the risk of over-optimizing the negative pairs at the expense of generative fidelity. The complete training algorithm is summarized in Algorithm 1. 

Stability of Alignment Guidance. Unlike SoftREPA, whose log-sum-exp contrastive objective admits descent directions that inflate negative denoising errors, our alignmentguided objective introduces a normalized preference correction within score matching. When the matched candidate dominates in Plackett–Luce (PL) form, $\Delta _ { \hat { \psi } _ { z } }$ becomes small so that the target score reduces toward the standard denoising objective. For negative pairs, $\Delta _ { \hat { \psi } , }$ contributes only through a normalized weighted correction term. 

Concretely, the alignment term $\gamma _ { z } \nabla _ { \pmb { x } _ { t } } \log p ( z | \pmb { x } _ { t } , \pmb { c } )$ takes the PL form $\begin{array} { r } { \nabla r ( { \pmb x } _ { t } , { \pmb c } ) - \sum _ { i } w _ { i } \nabla r ( { \pmb x } _ { t } , { \pmb c } ^ { i } ) } \end{array}$ , whose norm is bounded by a weighted combination of reward gradients, 

$$
\left\| \nabla \log p (z | \boldsymbol {x} _ {t}, \boldsymbol {c}) \right\| \leq \| \nabla r (\boldsymbol {x} _ {t}, \boldsymbol {c}) \| + \sum_ {i} w _ {i} \| \nabla r (\boldsymbol {x} _ {t}, \boldsymbol {c} ^ {i}) \|. \tag {18}
$$

With the reward instantiated as a scaled denoising error, the resulting correction remains finite and does not encourage unbounded growth of negative diffusion losses. In practice, this leads to substantially more stable training dynamics compared to SoftREPA, which exhibits gradual degradation without early-stopping (Training Dynamics Analysis in Section 4). 

# 4. Experiments

Implementation Details. We conducted experiments on SD1.5, SDXL, and SD3. For SD1.5 and SDXL, we trained soft tokens applied to the Down and Middle blocks of the UNet backbone, with 8 (4 positive and 4 negative) soft text tokens. For SD3, we trained 8 soft text tokens on the upper 5 transformer layers, which is the same configuration as Soft-REPA. The batch size was set to 16, which makes 3 negative prompts for each text-image pair across all models. Regarding SoftREPA, we used official checkpoints trained with larger negative pools: 7 negatives per positive (batch 64) for SD1.5/SDXL, and 3 negatives (batch 16) for SD3. During sampling, we dropped the negative soft tokens $( \psi ^ { - } )$ and used only the positive soft tokens $( \psi ^ { + } )$ for both conditional and unconditional generation. The positive and negative guidance scales $( \gamma ^ { + } , \gamma ^ { - } )$ were set to (1, 1) for SD1.5 and SDXL, and (1, 0.1) for SD3. For simplicity, we set the timedependent reward scale $\lambda _ { t }$ so that $\tilde { A } ( t ) = 1$ during training. Further implementation details are provided in Appendix G. 

Training Dynamics Analysis. We further analyze training stability against SoftREPA by tracking validation ImageReward throughout post-training. As shown in Figure 4, SoftREPA reaches peak ImageReward at early iterations and then substantially degrades, whereas our method maintains stable performance over longer training. In particular, Soft-REPA’s loss continues to decrease even when ImageReward drops, indicating over-optimization of the contrastive objective and the need for heuristic early stopping. By contrast, our PL-based score-matching objective uses a bounded and normalized correction, which reduces late-stage deterioration from amplified negative signals. 

Text to Image Generation. We conducted text-to-image generation experiments comparing our method with the baseline and SoftREPA (Lee et al., 2025a) on SD1.5, SDXL, and SD3. All models were trained on the COCO-train dataset (Lin et al., 2014) and evaluated on the COCO-val and GenEval benchmarks (Ghosh et al., 2023). 

In Table 1, our method achieves improved text–image alignment and image quality compared to the baselines. For the GenEval benchmark, we additionally compare against CaPO (Lee et al., 2025b) and RankDPO (Karthik et al., 2024), recent preference-based post-training approaches. Notably, our method significantly improves counting accuracy by +35%, effectively mitigating the over-emphasizing behavior observed in prior methods. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/800ca25470f588b304618a560dce4e8df8830cf5ab376105130b05b1fea1073d.jpg)



"A blue and white street sign that reads ‘Othello'."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/a53debca1c4db527ea5df6b42199f16d558d5b240042c7a5d47b6a0425d2dd8d.jpg)



"A green utility truck is parked on a street whileaman climbs inside."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/034f4c15b62b49f3000555c803b93cd042ab661f2f1c54da7462e18c21fc4e0b.jpg)



"A green netted bed ina light filled bedroom."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/63ec15133ca860f572a4aba66230867be5dea8f068008356f2bb6ff5fe8c5010.jpg)



“A box contains six donuts with varying types of glazes and toppings."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/b09e44c9c31db9f075662eb38eeaa2ae9d24507fce7a5c7e100d8cf7b14272ff.jpg)



"Two children smiling on top of luggage in parking lot."


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/c52c9507486590fab81d6a2b119a72036957b5786fee9ddcbfdcb43912e0df8f.jpg)



"A cat sleeping on a bed next toa laptop computer."



Figure 3. Qualitative comparison of text-to-image generation among SD3, SoftREPA, and our method. Prompts are sampled from the COCO validation set. Compared to SD3 and SoftREPA, our method produces images that better reflect the input text, while reducing common failure modes such as object repetition of SoftREPA.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/9dddddd9e22ec5f42909c8f63086db83e8f239f02641719f32cd63687d69e4b9.jpg)



Figure 4. Training stability comparison between AGSM (Ours) and SoftREPA. SoftREPA’s validation ImageReward degrades despite decreasing training loss, while our method remains stable throughout the later stages of training.


In Figure 5-(left), we evaluate performance along two complementary axes: human preference metrics (ImageReward) and image quality and diversity (FID). While a trade-off exists between these metrics, our method consistently improves preference-aligned performance over the baseline while maintaining substantially better FID than SoftREPA. Detailed quantitative results are provided in Appendix E. Figure 3 presents qualitative comparisons among SD3, Soft-REPA (Lee et al., 2025a), and our method, showing that our approach reduces redundant object generation and adheres more faithfully to the given text conditions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/70cb3130e342ec31c3444c315b0879772f7d9b99d5d68c4fc14b8e017900c405.jpg)



Figure 5. Comparison between the baseline, SoftREPA, and our method on image generation (ImageReward vs. FID) and image editing (CLIP vs. LPIPS) tasks. Optimal performance corresponds to the top-left region of the plot.


Text Guided Image Editing. We evaluate our method on PIE-Bench (Ju et al., 2023), a standard benchmark containing 700 images, containing source, target prompts and background mask. We compare our method against several training-free text-based editing baselines for SD1.5 and SD3. For SD1.5, we include PnP (Tumanyan et al., 2023) and MasaCtrl (Cao et al., 2023), evaluated with both direct (Ju et al., 2023) and DDIM (Song et al., 2021a) inversion. For SD3, we select methods representing different strategies: RF-Inversion (Rout et al., 2024) for an inversion-based approach and FlowEdit (Kulikov et al., 2024), FlowAlign (Kim et al., 2025) for inversion-free methods. Detailed experimental configurations are available in Appendix F. 

<table><tr><td colspan="9">COCO val5K</td></tr><tr><td>Model</td><td colspan="2">ImageReward↑</td><td colspan="2">PickScore↑</td><td colspan="2">CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>SD1.5</td><td colspan="2">17.72</td><td colspan="2">21.47</td><td colspan="2">26.4</td><td>25.08</td><td>24.59</td></tr><tr><td>Ours</td><td colspan="2">34.50</td><td colspan="2">21.59</td><td colspan="2">27.23</td><td>25.66</td><td>25.94</td></tr><tr><td>SDXL</td><td colspan="2">75.06</td><td colspan="2">22.38</td><td colspan="2">26.76</td><td>27.35</td><td>24.69</td></tr><tr><td>Ours</td><td colspan="2">84.22</td><td colspan="2">22.57</td><td colspan="2">26.86</td><td>27.96</td><td>24.83</td></tr><tr><td>SD3</td><td colspan="2">94.27</td><td colspan="2">22.54</td><td colspan="2">26.30</td><td>28.09</td><td>31.59</td></tr><tr><td>Ours</td><td colspan="2">103.3</td><td colspan="2">22.39</td><td colspan="2">27.00</td><td>28.22</td><td>34.08</td></tr><tr><td colspan="9">GenEval</td></tr><tr><td>Model</td><td># Trainable Params</td><td>Mean↑</td><td>Single↑</td><td>Two↑</td><td>Counting↑</td><td>Colors↑</td><td>Position↑</td><td>Color Attribution↑</td></tr><tr><td>SD3</td><td>-</td><td>0.68</td><td>0.99</td><td>0.86</td><td>0.56</td><td>0.85</td><td>0.27</td><td>0.55</td></tr><tr><td>CaPO (Lee et al., 2025b)</td><td>2B</td><td>0.71</td><td>0.99</td><td>0.87</td><td>0.63</td><td>0.86</td><td>0.31</td><td>0.59</td></tr><tr><td>RankDPO (Karthik et al., 2024)</td><td>2B</td><td>0.74</td><td>1.00</td><td>0.90</td><td>0.72</td><td>0.87</td><td>0.31</td><td>0.66</td></tr><tr><td>SoftREPA (Lee et al., 2025a)</td><td>0.9M</td><td>0.70</td><td>1.00</td><td>0.95</td><td>0.29</td><td>0.92</td><td>0.34</td><td>0.68</td></tr><tr><td>Ours</td><td>1.8M</td><td>0.72</td><td>1.00</td><td>0.91</td><td>0.64</td><td>0.89</td><td>0.26</td><td>0.64</td></tr></table>


Table 1. Quantitative evaluation of T2I generation on SD1.5, SDXL, and SD3. Generation quality is evaluated on the COCO-val 5K (Lin et al., 2014) and GenEval (Ghosh et al., 2023) benchmark. ImageReward, CLIP, HPS, and LPIPS are scaled by ×102.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/5f0bfc37be114309fe3e4ad770b3471a2d33702af468fac978434f1616af3b3c.jpg)



Figure 6. Qualitative comparison of image editing results from baseline methods, SoftREPA, and our method. The proposed method demonstrates a superior balance between text alignment and structural consistency.


As shown in Figure 5-(right), we evaluate performance from two complementary perspectives: text alignment (using CLIP similarity) and source consistency (using background LPIPS). While a trade-off exists in the two metrics, our method demonstrates a consistently superior balance, establishing a Pareto front compared to all baselines. While some baselines (e.g., FlowEdit, FlowAlign, RF-Inversion) excel in source consistency, they struggle to achieve strong text alignment. Conversely, SoftREPA (Lee et al., 2025a) often achieve high CLIP similarity at the expense of source consistency. As shown qualitatively in Figure 6, SoftREPA (Lee et al., 2025a) often over-edits or generates artifacts, which distort the original image structure. Additionally, Table 2 provides detailed quantitative results, including human preference scores alongside text alignment and structural preservation metrics. 

Complementarity with Diffusion RL methods. To further compare with other DPO-based methods and examine their complementarity with our approach, we conducted additional experiments by integrating our pretrained soft tokens into existing DPO frameworks. While DPO-based methods focus on preference alignment, our method targets representation alignment between text and image features by training soft text tokens to modulate the propagated features within a contrastive framework. To demonstrate the complementarity between the two paradigms, we combined the pretrained soft tokens with Diffusion-DPO (Wallace et al., 2024), SPO (Liang et al., 2025), and InPO (Lu et al., 2025) on SD1.5 and SDXL. As shown in Table 3, this simple integration consistently improves performance across all DPO-based baselines. The results suggest that combining explicit preference alignment with our representation-level text-image alignment provides a unified and more robust approach for enhancing text–image generation quality. 

<table><tr><td rowspan="2"></td><td rowspan="2">Inversion</td><td rowspan="2">Method</td><td colspan="2">Human Preference</td><td colspan="3">Text Alignment</td><td colspan="3">Background Preservation</td></tr><tr><td>Image-Reward↑</td><td>Pick-Score↑</td><td>CLIP/Edited↑</td><td>CLIP/Whole↑</td><td>HPSv2↑</td><td>PSNR↑</td><td>LPIPS/Whole↓</td><td>SSIM↑</td></tr><tr><td rowspan="6">SD1.5</td><td>ddim</td><td>MasaCtrl</td><td>-13.94</td><td>21.03</td><td>21.20</td><td>24.18</td><td>20.27</td><td>22.31</td><td>14.59</td><td>80.41</td></tr><tr><td>ddim</td><td>MasaCtrl + SoftREPA</td><td>-12.76</td><td>21.05</td><td>21.27</td><td>24.44</td><td>20.27</td><td>22.27</td><td>14.5</td><td>80.27</td></tr><tr><td>ddim</td><td>MasaCtrl + Ours</td><td>-8.71</td><td>21.01</td><td>21.86</td><td>25.08</td><td>20.24</td><td>21.60</td><td>11.65</td><td>79.30</td></tr><tr><td>direct</td><td>MasaCtrl</td><td>4.77</td><td>21.39</td><td>21.47</td><td>24.54</td><td>20.53</td><td>22.82</td><td>12.21</td><td>82.02</td></tr><tr><td>direct</td><td>MasaCtrl + SoftREPA</td><td>4.48</td><td>21.40</td><td>21.49</td><td>24.69</td><td>20.52</td><td>22.77</td><td>12.19</td><td>81.85</td></tr><tr><td>direct</td><td>MasaCtrl + Ours</td><td>7.79</td><td>21.41</td><td>22.19</td><td>25.53</td><td>20.52</td><td>21.99</td><td>9.87</td><td>80.78</td></tr><tr><td rowspan="3">SD3</td><td>-</td><td>RF-Inversion</td><td>128.0</td><td>22.07</td><td>24.17</td><td>27.26</td><td>20.84</td><td>13.10</td><td>36.10</td><td>57.17</td></tr><tr><td>-</td><td>RF-Inversion + SoftREPA</td><td>128.5</td><td>21.98</td><td>24.70</td><td>28.88</td><td>20.38</td><td>12.80</td><td>38.02</td><td>56.93</td></tr><tr><td>-</td><td>RF-Inversion + Ours</td><td>132.3</td><td>22.13</td><td>24.72</td><td>29.07</td><td>20.58</td><td>12.90</td><td>37.17</td><td>57.98</td></tr></table>


Table 2. Quantitative evaluation of image editing performance of baseline, SoftREPA (Lee et al., 2025a) and our method on PnP (Tumanyan et al., 2023), MasaCtrl (Cao et al., 2023), and RF-Inversion (Rout et al., 2024). ImageReward, CLIP, HPS, LPIPS, and SSIM are scaled by $\times 1 0 ^ { \dot { 2 } }$ and Distance is scaled by $\times 1 0 ^ { 3 }$ .


<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">COCO val5K</td></tr><tr><td>Model</td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td></tr><tr><td rowspan="7">SD1.5</td><td>Ours</td><td>34.50</td><td>21.59</td><td>27.23</td><td>25.66</td><td>25.94</td></tr><tr><td>DiffusionDPO (Wallace et al., 2024)</td><td>29.09</td><td>21.65</td><td>26.52</td><td>26.46</td><td>27.85</td></tr><tr><td>DiffusionDPO (Wallace et al., 2024) + Ours</td><td>42.47</td><td>21.79</td><td>27.34</td><td>26.28</td><td>27.27</td></tr><tr><td>SPO (Liang et al., 2025)</td><td>18.98</td><td>21.58</td><td>25.83</td><td>26.51</td><td>33.76</td></tr><tr><td>SPO (Liang et al., 2025) + Ours</td><td>34.86</td><td>21.94</td><td>26.53</td><td>26.92</td><td>30.67</td></tr><tr><td>InPO (Lu et al., 2025)</td><td>62.12</td><td>21.83</td><td>27.01</td><td>28.80</td><td>34.47</td></tr><tr><td>InPO (Lu et al., 2025) + Ours</td><td>67.95</td><td>22.02</td><td>27.37</td><td>28.33</td><td>33.67</td></tr><tr><td rowspan="7">SDXL</td><td>Ours</td><td>84.22</td><td>22.57</td><td>26.86</td><td>27.96</td><td>24.83</td></tr><tr><td>DiffusionDPO (Wallace et al., 2024)</td><td>91.67</td><td>22.65</td><td>27.36</td><td>28.90</td><td>28.64</td></tr><tr><td>DiffusionDPO (Wallace et al., 2024) + Ours</td><td>93.12</td><td>22.73</td><td>27.06</td><td>28.94</td><td>28.18</td></tr><tr><td>SPO (Liang et al., 2025)</td><td>96.95</td><td>23.20</td><td>25.93</td><td>30.85</td><td>31.84</td></tr><tr><td>SPO (Liang et al., 2025) + Ours</td><td>98.64</td><td>23.23</td><td>26.07</td><td>30.35</td><td>31.68</td></tr><tr><td>InPO (Lu et al., 2025)</td><td>94.05</td><td>22.74</td><td>26.91</td><td>29.54</td><td>27.78</td></tr><tr><td>InPO (Lu et al., 2025) + Ours</td><td>96.07</td><td>22.81</td><td>26.94</td><td>29.38</td><td>27.33</td></tr></table>


Table 3. Quantitative evaluation of comparison and complementarity with other Diffusion-RL methods on COCO-val5K dataset(Lin et al., 2014). ImageReward, CLIP, HPS, and LPIPS are scaled by $\times 1 0 ^ { 2 }$ .


<table><tr><td></td><td>Tokens</td><td>Data</td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>(i)</td><td><eq>\psi^{+}</eq></td><td><eq>\mathcal{D}^{+}</eq></td><td>94.79</td><td>22.26</td><td>26.93</td><td>27.81</td><td>34.46</td></tr><tr><td>(ii)</td><td>shared <eq>\psi</eq></td><td><eq>\mathcal{D}^{+},\mathcal{D}^{-}</eq></td><td>47.33</td><td>21.69</td><td>25.68</td><td>25.82</td><td>31.20</td></tr><tr><td>(iii)</td><td><eq>\psi^{+},\psi^{-}</eq></td><td><eq>\mathcal{D}^{+},\mathcal{D}^{-}</eq></td><td>103.3</td><td>22.39</td><td>27.00</td><td>28.22</td><td>34.08</td></tr></table>


Table 4. Ablation study on training strategy, including separated objectives over positive/negative subsets and soft token parameterization. In Tokens, shared ψ uses a single token set for both $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - }$ .


Ablation Study on Training Strategy. To isolate the effect of the positive/negative subset training $( \mathcal { D } ^ { + } , \mathcal { D } ^ { - } )$ and the dual-token parameterization, we conduct a controlled ablation while keeping the total number of learnable soft tokens fixed. As shown in Table 4, we compare three variants: (i) training only positive soft tokens $\psi ^ { + }$ on $\mathcal { D } ^ { + }$ ; (ii) training on both $D ^ { + }$ and $D ^ { - }$ with shared soft tokens, without explicitly decoupling positive and negative guidance; and (iii) our AGSM, which uses separate soft tokens, $\psi ^ { + }$ and $\psi ^ { - }$ , for the two subsets. 

As shown in Table $^ { 4 , }$ the results indicate that using both $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - }$ is critical for improving alignment metrics such as ImageReward, PickScore, CLIP, and HPSv2 scores. However, when positive and negative guidance are optimized with shared soft tokens, the improvement comes at the cost of degraded generative quality. This confirms that AGSM’s gains come from the combination of explicit positive/negative subset training and the structured dual-token design. 

<table><tr><td></td><td>Tokens</td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>(i)</td><td><eq>\psi^{+}, \psi^{-}</eq></td><td>84.53</td><td>22.18</td><td>26.71</td><td>27.64</td><td>36.47</td></tr><tr><td>(ii)</td><td><eq>\psi^{+}</eq> (Ours)</td><td>103.3</td><td>22.39</td><td>27.00</td><td>28.22</td><td>34.08</td></tr></table>


Table 5. Ablation study on sampling strategy. (i) uses negative tokens for unconditonal prediction in CFG and (ii) samples only with positive tokens.


Ablation Study on Sampling Strategy. We examine sampling behavior by comparing two strategies: (i) sampling with only positive soft tokens $( \psi ^ { + } )$ for both conditional and unconditional generation; (ii) sampling with positive soft tokens $( \psi ^ { + } )$ for conditional prediction and negative soft tokens $( \psi ^ { - } )$ for unconditional prediction. As illustrated in Table ${ 5 , }$ sampling without negative tokens yields notably higher image quality, especially better in ImageReward and FID, suggesting that negative-token sampling can overly suppress important visual information, degrading fidelity and diversity. We normalize all metrics to a consistent scale for comparison. Additional qualitative examples can be found in Appendix H. 

<table><tr><td><eq>\gamma^{-}</eq></td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>1</td><td>96.71</td><td>22.36</td><td>26.78</td><td>28.22</td><td>35.21</td></tr><tr><td>0.5</td><td>98.44</td><td>22.31</td><td>27.02</td><td>27.96</td><td>33.88</td></tr><tr><td>0.1 (Ours)</td><td>103.3</td><td>22.39</td><td>27.00</td><td>28.22</td><td>34.08</td></tr><tr><td>0.05</td><td>100.1</td><td>22.46</td><td>26.95</td><td>28.11</td><td>33.04</td></tr><tr><td>0</td><td>94.79</td><td>22.26</td><td>26.93</td><td>27.81</td><td>34.46</td></tr></table>


Table 6. Comparison on negative guidance scale on SD3, $\gamma ^ { - } \in$ {0, 0.05, 0.1, 0.5, 1}. $\gamma ^ { + }$ is set to 1.


<table><tr><td></td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>BT</td><td>29.67</td><td>21.52</td><td>27.13</td><td>25.31</td><td>24.76</td></tr><tr><td>PL (ours)</td><td>34.50</td><td>21.59</td><td>27.23</td><td>25.66</td><td>25.94</td></tr></table>


Table 7. Comparison of loss objective between Bradley-Terry (BT) and Plackett-Luce(PL) model. BT consists of pairwise positivenegative components, and PT generalizes BT into multi-negative components.


The Sensitivity on the Negative Guidance Scale. We further analyze the sensitivity to the negative guidance scale $\gamma ^ { - }$ . In practice, we use a larger value for models with stronger CFG (SD1.5, SDXL), and a smaller value for smaller CFG model (SD3). As shown in Table 6, SD3 remains stable across a range of scales, with $\gamma ^ { - } { = } 0 .$ 1 showing the best performance. After training, a fixed scale generalizes well across datasets and tasks at inference time without retraining. 

BT vs PT Loss Objective. We further compare our PLbased multi-candidate formulation with a simpler pairwise Bradley–Terry (BT) alternative. BT performs pairwise positive–negative comparison, whereas PL naturally handles multiple in-batch negative prompts through normalized multi-candidate preference modeling. PL consistently improves all alignment metrics over BT, while BT gives a slightly lower FID. This supports the use of PL for multicandidate alignment rather than reducing the objective to independent pairwise comparisons. 

# 5. Related Works

Recent work has extended Direct Preference Optimization (DPO) to diffusion models. Diffusion-DPO (Wallace et al., 2024) adapts DPO (Rafailov et al., 2023) to text-conditioned diffusion processes, and several follow-up methods improve preference modeling in different ways. SPO (Liang et al., 2025) introduces step-aware preference signals during onpolicy sampling. RankDPO (Karthik et al., 2024) generalizes preference learning to multi-sample ranking, while CaPO (Lee et al., 2025b) enhances fidelity by aggregating multiple reward signals. InPO (Lu et al., 2025) uses DDIM inversion to identify preference-relevant latent variables for selective finetuning, and DSPO (Zhu et al., 2025) integrates preference learning into the score-matching objective. While these approaches focus on human preference optimization, our method instead targets intrinsic text–image representation alignment, offering a complementary direction to DPO-based RL finetuning. 

Beyond preference-pair optimization, recent reward-based diffusion RL methods optimize text-to-image diffusion models using scalar feedback from external reward models (Liu et al., 2026; Zheng et al., 2025). DiffusionNFT (Zheng et al., 2025) adapts Negative-aware Fine-Tuning (NFT) (Chen et al., 2025) to diffusion models by using negative policy for policy optimization. Unlike DiffusionNFT, which defines implicit positive and negative samples through external rewards, AGSM derives them from intrinsic text–image representation alignment and injects the resulting guidance directly into score matching. 

# 6. Conclusion

We introduced Alignment-Guided Score Matching, a training-light approach that fine-tunes soft tokens to enhance intrinsic text–image representation alignment in diffusion models. By replacing explicit contrastive objectives with a score-based formulation using PL preference model and training negative samples with explicit preference directions, our method stabilizes soft-token optimization and mitigates off-manifold divergence. Through extensive experiments on text-to-image generation and text-guided image editing, we demonstrate consistent gains in alignment quality across multiple diffusion backbones. Our approach is further shown to be complementary to existing DPO-based post-training methods, yielding additional improvements when combined with preference-optimization techniques. Ablation studies confirm the importance of utilizing negative samples on training and highlight the impact of sampling strategies involving negative tokens. Overall, this work provides a simple yet effective framework for strengthening text–image alignment within the diffusion training dynamics, offering a broadly applicable enhancement for modern generative models. 

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

# Acknowledgement

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25468886), the National Research Foundation of Korea under Grant RS-2024-00336454, the AI Computing Infrastructure Enhancement (GPU Rental Support) User Support Program funded by the Ministry of Science and ICT (MSIT), Republic of Korea (RQT-25- 120217), the Advanced GPU Utilization Support Program funded by the Government of the Republic of Korea (Ministry of Science and ICT) (02-26-01-0404). 

# References



Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 





Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 





Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., and Zheng, Y. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22560–22570, October 2023. 





Chen, H., Zheng, K., Zhang, Q., Cui, G., Cui, Y., Ye, H., Lin, T.-Y., Liu, M.-Y., Zhu, J., and Wang, H. Bridging supervised learning and reinforcement learning in math reasoning. arXiv preprint arXiv:2505.18116, 2025. 





Clark, K., Vicol, P., Swersky, K., and Fleet, D. J. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 





Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 





Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. Dpok: Reinforcement learning for fine-tuning text-toimage diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023. 





Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., and Bau, D. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2426–2436, 2023. 





Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 





Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 





Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 





Ju, X., Zeng, A., Bian, Y., Liu, S., and Xu, Q. Direct inversion: Boosting diffusion-based editing with 3 lines of code. CoRR, abs/2310.01506, 2023. URL https: //doi.org/10.48550/arXiv.2310.01506. 





Karthik, S., Coskun, H., Akata, Z., Tulyakov, S., Ren, J., and Kag, A. Scalable ranked preference optimization for text-to-image generation. arXiv preprint, 2024. 





Kim, J., Hong, Y., Park, J., and Ye, J. C. Flowalign: Trajectory-regularized, inversion-free flow-based image editing, 2025. URL https://arxiv.org/abs/ 2505.23145. 





Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., and Michaeli, T. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024. 





Lee, J.-Y., Cha, B., Kim, J., and Ye, J. C. Aligning text to image in diffusion models is easier than you think. arXiv preprint arXiv:2503.08250, 2025a. 





Lee, K., Li, X., Wang, Q., He, J., Ke, J., Yang, M.-H., Essa, I., Shin, J., Yang, F., and Li, Y. Calibrated multipreference optimization for aligning diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18465–18475, 2025b. 





Liang, Z., Yuan, Y., Gu, S., Chen, B., Hang, T., Cheng, M., Li, J., and Zheng, L. Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13199–13208, 2025. 





Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740– 755. Springer, 2014. 





Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026. 





Lu, Y., Wang, Q., Cao, H., Wang, X., Xu, X., and Zhang, M. Inpo: Inversion preference optimization with reparametrized ddim for efficient diffusion model alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 28629–28639, 2025. 





Luce, R. D. et al. Individual choice behavior, volume 4. Wiley New York, 1959. 





Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023. 





Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 





Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023. 





Rout, L., Chen, Y., Ruiz, N., Caramanis, C., Shakkottai, S., and Chu, W.-S. Semantic image inversion and editing using rectified stochastic differential equations, 2024. URL https://arxiv.org/abs/2410.10792. 





Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR, 2021a. 





Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum? id=PxTIG12RRHS. 





Tumanyan, N., Geyer, M., Bagon, S., and Dekel, T. Plugand-play diffusion features for text-driven image-toimage translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1921–1930, June 2023. 





Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024. 





Wang, Y., Li, Z., Zang, Y., Bu, J., Zhou, Y., Xin, Y., He, J., Wang, C., Lu, Q., Jin, C., and Wang, J. Unigenbench++: A unified semantic evaluation benchmark for text-toimage generation, 2026. URL https://arxiv.org/ abs/2510.18701. 





Xian, J. J. C., Li, M., Yang, H., Tao, X., Wan, P., Sigal, L., and Liao, R. Free lunch alignment of text-to-image diffusion models without preference image pairs. arXiv preprint arXiv:2509.25771, 2025. 





Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pp. 15903–15935, 2023. 





Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., and Liu, M.-Y. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025. 





Zhu, H., Xiao, T., and Honavar, V. G. Dspo: Direct score preference optimization for diffusion model alignment. In The Thirteenth International Conference on Learning Representations, 2025. 



# A. Interpretation of Negative Target Distribution

A Bayes-consistent negative conditional distribution from Eq. (9) can be written as 

$$
p _ {t} ^ {-} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}\right) := p _ {t} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}, z = 0\right) \propto p _ {t} \left(\boldsymbol {x} _ {t} \mid \boldsymbol {c}\right) p (z = 0 \mid \boldsymbol {x} _ {t}, \boldsymbol {c}). \tag {19}
$$

Since $p ( z { = } 0 | x _ { t } , c ) = 1 { - } p ( z { = } 1 | x _ { t } , c )$ , the corresponding guidance term becomes 

$$
\nabla_ {\boldsymbol {x} _ {t}} \log p (z = 0 | \boldsymbol {x} _ {t}, \boldsymbol {c}) = \nabla_ {\boldsymbol {x} _ {t}} \log \left(1 - p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c})\right) \tag {20}
$$

$$
= - \frac {\nabla_ {\boldsymbol {x} _ {t}} p (z = 1 | \boldsymbol {x} _ {t} , \boldsymbol {c})}{1 - p (z = 1 | \boldsymbol {x} _ {t} , \boldsymbol {c})}. \tag {21}
$$

Instead of directly using $\scriptstyle p ( z = 0 | \mathbf { x } _ { t } , \mathbf { c } )$ , we adopt the surrogate form $p ( z { = } 1 | x _ { t } , c ) ^ { - \gamma ^ { - } }$ , which yields 

$$
\nabla_ {\boldsymbol {x} _ {t}} \log p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}) ^ {- \gamma^ {-}} = - \gamma^ {-} \nabla_ {\boldsymbol {x} _ {t}} \log p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}) \tag {22}
$$

$$
= - \gamma^ {-} \frac {\nabla_ {\boldsymbol {x} _ {t}} p (z = 1 | \boldsymbol {x} _ {t} , \boldsymbol {c})}{p (z = 1 | \boldsymbol {x} _ {t} , \boldsymbol {c})}. \tag {23}
$$

Therefore, 

$$
\nabla_ {\boldsymbol {x} _ {t}} \log p (z = 0 | \boldsymbol {x} _ {t}, \boldsymbol {c}) \| \nabla_ {\boldsymbol {x} _ {t}} \log p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c}) ^ {- \gamma^ {-}}. \tag {24}
$$

That is, both gradients of Eq. (20) and Eq. (22) are pointwise proportional with a positive scaling factor and share the same directional component $- \nabla _ { \pmb { x } _ { t } } p ( z = 1 | \pmb { x } _ { t } , \pmb { c } )$ . 

Therefore, both formulations induce repulsive updates away from regions with high alignment probability, differing only in their local scaling factors. 

Accordingly, the proposed inverse weighting can be interpreted as a surrogate negative guidance term that preserves the repulsive direction of the Bayes-consistent formulation while yielding a unified additive score form. 

# B. Reward Function Derivation

We derive the implicit reward used in our alignment objective. Starting from the definition in Equation (8), we define the reward as the expected log-likelihood under the DDPM posterior: 

$$
r (\boldsymbol {x} _ {t}, \boldsymbol {c}) := \lambda_ {t} \mathbb {E} _ {q (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0})} [ \log p _ {\theta} (\boldsymbol {x} _ {t - 1} \mid \boldsymbol {x} _ {t}, \boldsymbol {c}) ]. \tag {25}
$$

The reverse process of DDPM (Ho et al., 2020) provides an explicit Gaussian parameterization for $p _ { \theta } ( \pmb { x } _ { t - 1 } \mid \pmb { x } _ { t } , \pmb { c } )$ : 

$$
p _ {\theta} \left(\boldsymbol {x} _ {t - 1} \mid \boldsymbol {x} _ {t}, \boldsymbol {c}\right) = \mathcal {N} \left(\boldsymbol {x} _ {t - 1}; \mu_ {\theta} \left(\boldsymbol {x} _ {t}, t, \boldsymbol {c}\right), \sigma_ {t} ^ {2} \mathbf {I}\right), \tag {26}
$$

where µθ(xt, t, c) = √1αt  $\begin{array} { r } { \mu _ { \theta } ( \pmb { x } _ { t } , t , \pmb { c } ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \Big ( \pmb { x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( \pmb { x } _ { t } , t , \pmb { c } ) \Big ) } \end{array}$ , and $\begin{array} { r } { \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } , \alpha _ { t } = 1 - \beta _ { t } , \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ . Since both ${ q } ( \pmb { x } _ { t - 1 } \ | \ \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ and $p _ { \theta } ( \pmb { x } _ { t - 1 } \mid \pmb { x } _ { t } , \pmb { c } )$ are Gaussian, expanding the log-density yields 

$$
r \left(\boldsymbol {x} _ {t}, \boldsymbol {c}\right) = - \frac {\lambda_ {t}}{2 \sigma_ {t} ^ {2}} \mathbb {E} _ {q} \left[ \| \boldsymbol {x} _ {t - 1} - \mu_ {\theta} \| _ {2} ^ {2} \right] + C _ {1} \tag {27}
$$

where $C _ { 1 }$ is a timestep dependent constant. Using 

$$
\mathbb {E} \left\| \boldsymbol {x} _ {t - 1} - \mu_ {\theta} \right\| ^ {2} = \left\| \tilde {\mu} _ {t} - \mu_ {\theta} \right\| ^ {2} + d \tilde {\beta} _ {t} \tag {28}
$$

with $\tilde { \mu } _ { t } = \mathbb { E } [ { \pmb x } _ { t - 1 } \mid { \pmb x } _ { t } , { \pmb x } _ { 0 } ]$ and posterior variance $\tilde { \beta } _ { t }$ , 

$$
r (\boldsymbol {x} _ {t}, \boldsymbol {c}) = - \frac {\lambda_ {t}}{2 \sigma_ {t} ^ {2}} \| \tilde {\mu} _ {t} - \mu_ {\theta} \| _ {2} ^ {2} + C _ {2}. \tag {29}
$$

Since $C _ { 2 }$ is independent of θ and ${ \mathbf { \nabla } } x _ { t } .$ expressing both $\tilde { \mu } _ { t }$ and $\mu _ { \theta }$ in the noise parameterization gives 

$$
r \left(\boldsymbol {x} _ {t}, \boldsymbol {c}\right) = - \frac {\lambda_ {t} \beta_ {t}}{2 \alpha_ {t} \left(1 - \bar {\alpha} _ {t - 1}\right)} \left\| \boldsymbol {\epsilon} _ {\theta} ^ {\hat {\psi} _ {z}} \left(\boldsymbol {x} _ {t}, t, \boldsymbol {c}\right) - \boldsymbol {\epsilon} _ {t} \right\| _ {2} ^ {2} \tag {30}
$$

EMA soft tokens $\hat { \psi } _ { z }$ are used to stabilize the reward evaluation. Thus, higher reward corresponds to closer agreement between model-predicted noise and the forward noise realization. 

# C. Alignment-Guided Score Matching Derivation

To compute the gradient needed in the alignment score model (Equation (13)), we differentiate the reward w.r.t. $\mathbf { \nabla } _ { \mathbf { x } _ { t } : }$ 

$$
\nabla_ {\boldsymbol {x} _ {t}} r (\boldsymbol {x} _ {t}, \boldsymbol {c}) = - \frac {\lambda_ {t} \beta_ {t}}{\alpha_ {t} (1 - \bar {\alpha} _ {t - 1})} \mathbf {J} _ {\epsilon_ {\theta} ^ {\hat {\psi} _ {z}}} (\boldsymbol {x} _ {t}) ^ {\top} \left(\boldsymbol {\epsilon} _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - \boldsymbol {\epsilon} _ {t}\right), \tag {31}
$$

where we omit computing the jacobian $\begin{array} { r } { \mathbf { J } _ { \epsilon _ { \theta } ^ { \hat { \psi } _ { z } } } ( \pmb { x } _ { t } ) = \frac { \partial \epsilon _ { \theta } ^ { \hat { \psi } _ { z } } ( \pmb { x } _ { t } , t , \pmb { c } ) } { \partial \pmb { x } _ { t } } } \end{array}$ . Plugging Equation (31) into Equation (13), the gradient of the alignment score model becomes 

$$
\nabla_ {\boldsymbol {x} _ {t}} \log p (z = 1 \mid \boldsymbol {x} _ {t}, \boldsymbol {c}) = - A (t) \left(\epsilon_ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - \epsilon_ {t} - \sum_ {i} w _ {i} \left(\epsilon_ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c} ^ {i}) - \epsilon_ {t}\right)\right) \tag {32}
$$

$$
= - A (t) \left(\boldsymbol {\epsilon} _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - \sum_ {i} w _ {i} \boldsymbol {\epsilon} _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c} ^ {i})\right), \tag {33}
$$

where A(t) = αt(1−α¯t−1) . $\begin{array} { r } { A ( t ) = \frac { \lambda _ { t } \beta _ { t } } { \alpha _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) } } \end{array}$ λtβt With the same realization of $\epsilon _ { t } , \epsilon _ { t }$ cancels between positive and negative terms, giving a clean contrast between the positive prediction and the weighted average. Using the definition of the score function which connects the score model and diffusion models described in (Song et al., 2021b), we can derive $\begin{array} { r } { \nabla _ { \pmb { x } _ { t } } \log \frac { p _ { \theta } ( \pmb { x } _ { t } | \pmb { c } ) } { p _ { d a t a } ( \pmb { x } _ { t } | \pmb { c } ) } = } \end{array}$ $\begin{array} { r } { - \frac { 1 } { \sqrt { 1 - \bar { \alpha } _ { t } } } \big ( \epsilon _ { \theta } \big ( { \pmb x } _ { t } , { \pmb c } , t \big ) - { \pmb \epsilon } _ { t } \big ) } \end{array}$ . By combining this and Equation (11), Equation (12) becomes 

$$
\mathcal {L} (\psi_ {z}) = \underset { \begin{array}{c} t \sim U (0, 1), \\ \epsilon \sim \mathcal {N} (0, \mathbf {I}) \end{array} } {\mathbb {E}} _ {\substack {(\boldsymbol {x}, \boldsymbol {c}) \sim D, \\ t \sim U (0, 1),}} \left[ w (t) \| \nabla \log p _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t} | \boldsymbol {c}) - \nabla \log \tilde {p} _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c}, z) \| _ {2} ^ {2} \right] \tag{34}
$$

$$
= \mathbb {E} _ {\substack {(\boldsymbol {x}, \boldsymbol {c}) \sim D, \\ t \sim U (0, 1), \\ \epsilon \sim \mathcal {N} (0, \mathbf {I})}} \left[ w (t) \| \nabla \log p _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t} | \boldsymbol {c}) - (\nabla \log p _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {c}) + \gamma_ {z} \nabla \log p (z = 1 \mid \boldsymbol {x} _ {t}, \boldsymbol {c})) \| _ {2} ^ {2} \right] \tag{35}
$$

$$
= \mathbb {E} _ {\substack {(\boldsymbol {x}, \boldsymbol {c}) \sim D, \\ t \sim U (0, 1), \\ \epsilon \sim \mathcal {N} (0, \mathbf {I})}} \left[ w (t) \| - \frac {1}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {\theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - (- \frac {1}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {t} + \gamma_ {z} \nabla \log p (z = 1 | \boldsymbol {x} _ {t}, \boldsymbol {c})) \| _ {2} ^ {2} \right]. \tag{36}
$$

By leveraging the gradient of the alignment score modelEq. (33), the final score matching loss can be rewritten as follows: 

$$
\mathcal{L}(\psi_{z}) = \mathbb{E}_{\substack{(\boldsymbol {x},\boldsymbol {c})\sim D,\\ t\sim U(0,1),\\ \boldsymbol {\epsilon}\sim \mathcal{N}(0,\boldsymbol {\mathrm{I}})}}[\frac{w(t)}{1 - \bar{\alpha}_{t}} \| \boldsymbol{\epsilon}_{\theta}^{\psi_{z}}(\boldsymbol{x}_{t},t,\boldsymbol {c}) - (\boldsymbol{\epsilon}_{t} + \frac{\gamma_{z}\lambda_{t}\beta_{t}\sqrt{1 - \bar{\alpha}_{t}}}{\alpha_{t}(1 - \bar{\alpha}_{t - 1})} (\boldsymbol{\epsilon}_{\theta}^{\hat{\psi}_{z}}(\boldsymbol{x}_{t},t,\boldsymbol {c}) - \sum_{i}w_{i}\boldsymbol{\epsilon}_{\theta}^{\hat{\psi}_{z}}(\boldsymbol{x}_{t},t,\boldsymbol{c}^{i}))\|_{2}^{2}]
$$

$$
= \mathbb {E} _ {\substack {(\boldsymbol {x}, \boldsymbol {c}) \sim D, \\ t \sim U (0, 1), \\ \boldsymbol {\epsilon} \sim \mathcal {N} (0, \mathbf {I})}} [ \tilde {w} (t) \| \boldsymbol {\epsilon} _ {\theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - (\boldsymbol {\epsilon} _ {t} + \gamma_ {z} \tilde {A} (t) (\boldsymbol {\epsilon} _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - \sum_ {i} w _ {i} \boldsymbol {\epsilon} _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t}, t, \boldsymbol {c} ^ {i})) \| _ {2} ^ {2} ] \tag{37}
$$

where A˜(t) = λtβt√1−α¯t $\begin{array} { r } { \tilde { A } ( t ) = \frac { \lambda _ { t } \beta _ { t } \sqrt { 1 - \bar { \alpha } _ { t } } } { \alpha _ { t } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } } \end{array}$ and $\begin{array} { r } { \tilde { w } ( t ) = \frac { w ( t ) } { 1 - \bar { \alpha } _ { t } } } \end{array}$ . 

# D. Alignment-Guided Score Matching in Flow Model

Flow model. Flow model defines the interpolant $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ between data $\pmb { x } _ { 0 } \sim p ( \pmb { x } )$ and noise $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ as 

$$
\boldsymbol {x} _ {t} = (1 - t) \boldsymbol {x} _ {0} + t \boldsymbol {\epsilon}, \quad t \in [ 0, 1 ], \tag {38}
$$

and trains the flow model $v _ { \theta }$ to match the target velocity via 

$$
\mathcal {L} _ {\text { flow }} = \mathbb {E} _ {\boldsymbol {x} _ {0}, \epsilon , t, \boldsymbol {c}} \left[ \| v _ {\theta} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) - (\epsilon - \boldsymbol {x} _ {0}) \| _ {2} ^ {2} \right]. \tag {39}
$$

Text-image alignment in flow model. In the reward fitting process, we measure the alignment between an image and a text pair $( \pmb { x } _ { t } , \pmb { c } )$ , denoted as z, using a Plackett-Luce (PL) model: 

$$
p (z | \boldsymbol {x} _ {t}, \boldsymbol {c}) = \frac {\exp \left(r (\boldsymbol {x} _ {t} , \boldsymbol {c})\right)}{\sum_ {i} \exp \left(r (\boldsymbol {x} _ {t} , \boldsymbol {c} ^ {i})\right)}, \quad w _ {i} = \frac {\exp \left(r (\boldsymbol {x} _ {t} , \boldsymbol {c} ^ {i})\right)}{\sum_ {j} \exp \left(r (\boldsymbol {x} _ {t} , \boldsymbol {c} ^ {j})\right)}. \tag {40}
$$

Following SoftREPA (Lee et al., 2025a), which interprets the negative denoising score-matching loss as a logit of contrastive learning, we extend this idea to the flow model by defining the reward using the flow model’s conditional likelihood: 

$$
r (\pmb {x} _ {t}, \pmb {c}) = \lambda_ {t} \log p _ {\theta} (\pmb {x} _ {t} | \pmb {x} _ {t + \Delta}, \pmb {c}), \qquad p _ {\theta} = \mathcal {N} \big (\pmb {x} _ {t}; \mu_ {\theta}, \sigma_ {t + \Delta} ^ {2} \mathbf {I} \big), \quad \mu_ {\theta} = \pmb {x} _ {t + \Delta} - \Delta v _ {\theta} (\pmb {x} _ {t + \Delta}, t + \Delta , \pmb {c}). (4 1)
$$

Here $\Delta > 0$ denotes a small time step, and $\sigma _ { t + \Delta } ^ { 2 }$ represents the local transition variance (e.g., from the underlying probability–flow ODE). Under this local approximation, the flow dynamics 

$$
\frac {d \boldsymbol {x} _ {t}}{d t} = v _ {\theta} (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) \tag {42}
$$

can be locally approximated (via first-order Euler discretization) as a Gaussian transition: 

$$
p _ {\theta} (\boldsymbol {x} _ {t} \mid \boldsymbol {x} _ {t + \Delta}, \boldsymbol {c}) \simeq \mathcal {N} \big (\boldsymbol {x} _ {t}; \boldsymbol {x} _ {t + \Delta} - \Delta v _ {\theta} (\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c}), \sigma_ {t + \Delta} ^ {2} \mathbf {I} \big), \tag {43}
$$

which describes the probability of reaching $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ from $\pmb { x } _ { t + \Delta }$ under the flow model $v _ { \theta }$ . Then, the estimated reward becomes 

$$
r (\boldsymbol {x} _ {t}, \boldsymbol {c}) = - \frac {\lambda_ {t}}{2 \sigma_ {t + \Delta} ^ {2}} \left\| \boldsymbol {x} _ {t} - \mu_ {\theta} ^ {\hat {\psi} _ {z}} \right\| _ {2} ^ {2}, \quad \mu_ {\theta} ^ {\hat {\psi} _ {z}} = \boldsymbol {x} _ {t + \Delta} - \Delta v _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c}). \tag {44}
$$

To further stabilize the reward calculation, we used flow model with soft tokens updated by exponential moving average $( \mathrm { E M A } ) , \hat { \psi } _ { z }$ . The gradient of the reward with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is 

$$
\nabla_ {\boldsymbol {x} _ {t}} r (\boldsymbol {x} _ {t}, \boldsymbol {c}) = - \frac {\lambda_ {t}}{\sigma_ {t + \Delta} ^ {2}} \left(\boldsymbol {x} _ {t} - \mu_ {\theta} ^ {\hat {\psi} _ {z}}\right). \tag {45}
$$

Plugging Equation (45) into Equation (40), we obtain 

$$
\begin{array}{l} \nabla_ {\pmb {x} _ {t}} \log p (z = 1 | \pmb {x} _ {t}, \pmb {c}) = \nabla_ {\pmb {x} _ {t}} r (\pmb {x} _ {t}, \pmb {c}) - \sum_ {i} w _ {i} \nabla_ {\pmb {x} _ {t}} r (\pmb {x} _ {t}, \pmb {c} ^ {i}) \\ = - \frac {\lambda_ {t}}{\sigma_ {t + \Delta} ^ {2}} (\boldsymbol {x} _ {t} - \mu_ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {c})) + \sum_ {i} w _ {i} \frac {\lambda_ {t}}{\sigma_ {t + \Delta} ^ {2}} (\boldsymbol {x} _ {t} - \mu_ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {c} ^ {i})) \\ = \frac {\lambda_ {t}}{\sigma_ {t + \Delta} ^ {2}} \left(\mu_ {\theta^ {\prime}} (\boldsymbol {c}) - \sum_ {i} w _ {i} \mu_ {\theta} ^ {\hat {\psi} _ {z}} \left(\boldsymbol {c} ^ {i}\right)\right). \tag {46} \\ \end{array}
$$

Substituting $\mu _ { \theta } ^ { \hat { \psi } _ { z } } ( c ) = x _ { t + \Delta } - \Delta v _ { \theta } ^ { \hat { \psi } _ { z } } ( x _ { t + \Delta } , t + \Delta , c )$ cancels out $\pmb { x } _ { t + \Delta }$ and yields 

$$
\nabla_ {\boldsymbol {x} _ {t}} \log p (z = 1 \mid \boldsymbol {x} _ {t}, \boldsymbol {c}) = - \frac {\lambda_ {t} \Delta}{\sigma_ {t + \Delta} ^ {2}} \left(v _ {\theta} ^ {\hat {\psi} _ {z}} \left(\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c}\right) - \sum_ {i} w _ {i} v _ {\theta} ^ {\hat {\psi} _ {z}} \left(\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c} ^ {i}\right)\right). \tag {47}
$$

Score matching loss in flow model. Starting from the score matching objective 

$$
\mathcal {L} (\psi_ {z}) = \underset { \begin{array}{c} t \sim U (0, 1), \\ \epsilon \sim \mathcal {N} (0, \mathbf {I}) \end{array} } {\mathbb {E}} _ {(\boldsymbol {x}, \boldsymbol {c}) \sim D,} \left[ w (t) \left\| \nabla_ {\boldsymbol {x} _ {t}} \log p _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t} \mid \boldsymbol {c}) - \left(\nabla_ {\boldsymbol {x} _ {t}} \log p _ {t} (\boldsymbol {x} _ {t} \mid \boldsymbol {c}) + \gamma_ {z} \nabla_ {\boldsymbol {x} _ {t}} \log p (z = 1 \mid \boldsymbol {x} _ {t}, \boldsymbol {c})\right) \right\| _ {2} ^ {2} \right], \tag {48}
$$

we use the probability–flow ODE relation 

$$
v (\boldsymbol {x} _ {t}, t, \boldsymbol {c}) = - K (t) \nabla_ {\boldsymbol {x} _ {t}} \log p _ {t} (\boldsymbol {x} _ {t} \mid \boldsymbol {c}), \quad K (t) > 0, \tag {49}
$$

where $K ( t )$ is a time-dependent scaling factor determined by the flow formulation (e.g., $\begin{array} { r } { K ( t ) = \frac { 1 } { 2 } g ( t ) ^ { 2 } } \end{array}$ in a probability–flow ODE). 

Substituting 

$$
v _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t}, \boldsymbol {c}) := - K (t)   \nabla_ {\boldsymbol {x} _ {t}} \log p _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t} \mid \boldsymbol {c}),
$$

and the gradient of alignment score model 

$$
\nabla_ {\pmb {x} _ {t}} \log p (z = 1 | \pmb {x} _ {t}, \pmb {c}) = - \frac {\lambda_ {t} \Delta}{\sigma_ {t + \Delta} ^ {2}} \Big (v _ {\theta} ^ {\hat {\psi} _ {z}} (\pmb {x} _ {t + \Delta}, t + \Delta , \pmb {c}) - \sum_ {i} w _ {i} v _ {\theta} ^ {\hat {\psi} _ {z}} (\pmb {x} _ {t + \Delta}, t + \Delta , \pmb {c} ^ {i}) \Big),
$$

into the score matching objective, we obtain 

$$
\begin{array}{l} \mathcal {L} (\psi_ {z}) = \mathbb {E} \Big [ w (t) \left\| \nabla_ {\pmb {x} _ {t}} \log p _ {t, \theta} ^ {\psi_ {z}} (\pmb {x} _ {t} | \pmb {c}) - \left(\nabla_ {\pmb {x} _ {t}} \log p _ {t} (\pmb {x} _ {t} | \pmb {c}) + \gamma_ {z} \nabla_ {\pmb {x} _ {t}} \log p (z = 1 | \pmb {x} _ {t}, \pmb {c})\right) \right\| _ {2} ^ {2} \Big ] \\ = \mathbb {E} \left[ \frac {w (t)}{K (t) ^ {2}} \left\| v _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t}, \boldsymbol {c}) - \left(v _ {t} (\boldsymbol {x} _ {t}, \boldsymbol {c}) + \gamma_ {z} K (t) \frac {\lambda_ {t} \Delta}{\sigma_ {t + \Delta} ^ {2}} \left(v _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c}) - \sum_ {i} w _ {i} v _ {\theta} ^ {\hat {\psi} _ {z}} (\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c} ^ {i})\right)\right) \right\| _ {2} ^ {2} \right] \tag {50} \\ \end{array}
$$

where the expectation is over $( \pmb { x } , \pmb { c } , z ) \sim D , t \sim U ( 0 , 1 )$ , and $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ . 

$\tilde { w } ( t ) = w ( t ) / K ( t ) ^ { 2 }$ $\begin{array} { r } { B ( t ) = K ( t ) \frac { \lambda _ { t } \Delta } { \sigma _ { t + \Delta } ^ { 2 } } } \end{array}$ 

$$
\mathcal {L} (\psi_ {z}) = \mathbb {E} \left[ \tilde {w} (t) \left\| v _ {t, \theta} ^ {\psi_ {z}} (\boldsymbol {x} _ {t}, \boldsymbol {c}) - \left(v _ {t} (\boldsymbol {x} _ {t}, \boldsymbol {c}) + \gamma_ {z} B (t) \left(v _ {\theta} ^ {\hat {\psi} _ {z}} \left(\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c}\right) - \sum_ {i} w _ {i} v _ {\theta} ^ {\hat {\psi} _ {z}} \left(\boldsymbol {x} _ {t + \Delta}, t + \Delta , \boldsymbol {c} ^ {i}\right)\right)\right) \right\| _ {2} ^ {2} \right], \tag {51}
$$

which mirrors the diffusion-based objective in Equation (37) while entirely derived within the flow-based formulation. 

# E. Additional Results on Image Generation

COCO-val T2I Generation. In this section, we present detailed quantitative results on the COCO-val image generation task, comparing the baseline, SoftREPA, and our method across different backbones. As shown in Table 8, our method consistently outperforms the baselines in both human preference scores and CLIP scores. With respect to the trade-off between human preference scores and FID, our method achieves lower FID than SoftREPA while maintaining strong preference-aligned performance. 

<table><tr><td rowspan="2"></td><td rowspan="2">Model</td><td colspan="5">COCO val5K</td></tr><tr><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td rowspan="3">SD1.5</td><td>SD1.5</td><td>17.72</td><td>21.47</td><td>26.4</td><td>25.08</td><td>24.59</td></tr><tr><td>SoftREPA</td><td>40.02</td><td>21.64</td><td>27.09</td><td>26.05</td><td>29.25</td></tr><tr><td>Ours</td><td>34.50</td><td>21.59</td><td>27.23</td><td>25.66</td><td>25.94</td></tr><tr><td rowspan="3">SDXL</td><td>SDXL</td><td>75.06</td><td>22.38</td><td>26.76</td><td>27.35</td><td>24.69</td></tr><tr><td>SoftREPA</td><td>85.28</td><td>22.63</td><td>26.78</td><td>28.41</td><td>26.42</td></tr><tr><td>Ours</td><td>84.22</td><td>22.57</td><td>26.86</td><td>27.96</td><td>24.83</td></tr><tr><td rowspan="3">SD3</td><td>SD3</td><td>94.27</td><td>22.54</td><td>26.30</td><td>28.09</td><td>31.59</td></tr><tr><td>SoftREPA</td><td>108.5</td><td>22.55</td><td>26.91</td><td>28.91</td><td>36.21</td></tr><tr><td>Ours</td><td>103.3</td><td>22.39</td><td>27.00</td><td>28.22</td><td>34.08</td></tr></table>


Table 8. Quantitative evaluation of T2I generation on SD1.5, SDXL, and SD3. Generation quality is evaluated on the COCO-val 5K (Lin et al., 2014) and GenEval (Ghosh et al., 2023) benchmark. ImageReward, CLIP, HPS, and LPIPS are scaled by $\times 1 0 ^ { 2 }$ .


Long-prompt T2I Generation. To further evaluate generalization to long-text prompts, we additionally test on UniGen-Bench++ (Wang et al., 2026) with 600 long-text T2I prompts, comparing SD3, SoftREPA, and our method. As shown in Table 9, our method achieves the best ImageReward, CLIP, PickScore, and HPSv2, indicating that the proposed method remains effective beyond short COCO-style captions. 

# F. Additional Results on Image Editing

In this section, we provide implementation details on the baseline methods used in the image editing experiments. We provide quantitative evaluation results of all editing methods in Table 10. Our method enhances text alignment of baseline methods with comparable or superior background preservation. In Figure 7, we present additional qualitative comparison on baseline editing methods. 

<table><tr><td>Model</td><td>ImageReward</td><td>CLIP</td><td>PickScore</td><td>HPSv2</td></tr><tr><td>SD3 Base</td><td>82.33</td><td>29.50</td><td>21.32</td><td>28.87</td></tr><tr><td>SoftREPA</td><td>90.63</td><td>28.93</td><td>21.18</td><td>28.61</td></tr><tr><td>Ours</td><td>98.01</td><td>29.56</td><td>21.48</td><td>29.66</td></tr></table>


Table 9. Comparison on UniGenBench++ (Wang et al., 2026).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/9c4f10200e02aad6b5d42233df526b525c26131acce3f5d89f806381c738b985.jpg)



Figure 7. Qualitative comparison of our proposed method with different editing methods.


# Baseline methods - SD3

1. RF-Inversion (Rout et al., 2024) : We employ RF-Inversion as a representative baseline for image editing utilizing the inversion process within the SD3 framework. Based on the official implementation, we configure the parameters as follows: $\gamma = 0 . 5 , \eta = 0 . 9$ , a starting time s = 0, and a stopping time $\tau = 0 . 2 5$ . Null-text embedding is leveraged for the inversion stage, and a Classifier-Free Guidance (CFG) scale of 13.5 is applied during the sampling phase to ensure consistency with the FlowEdit implementation. 

2. FlowEdit (Kulikov et al., 2024) : FlowEdit is selected as a baseline method that effectively bypasses the inversion process in SD3. We utilize a CFG scale of 3.5 for the source direction and 13.5 for the target direction. Additionally, the starting index of timestep is set to 18 out of a total of 50 timesteps, resulting in 33 timesteps. 

3. FlowAlign (Kim et al., 2025) : We include FlowAlign, a method that improves FlowEdit by regularizing the editing trajectory, demonstrating superior source consistency. Consistent with the FlowEdit configuration, the target CFG scale is set to 13.5. We adopt the official implementation’s setting for the regularization coefficient, ζ = 0.01. 

# Baseline methods - SD1.5

1. MasaCtrl (Cao et al., 2023) : For an editing method that adapts the self-attention mechanism to enable consistent synthesis, we select MasaCtrl. Following the original work, we configure the initial layout synthesis to stop at step $S = 4$ and initiate the mutual self-attention control from layer $L = 1 0$ . We evaluate this method using both DDIM inversion and a direct inversion approach. NFE is set to 50 with a CFG scale of 7.0. 

2. PnP (Tumanyan et al., 2023) : PnP performs text-driven image-to-image translation by extracting and injecting spatial features (f) from intermediate decoder layers and self-attention maps (A) from the guidance image’s generation into the target image’s generation. For our experiments, which use 50 total sampling steps, we set the injection thresholds as $\tau _ { A } = 2 5$ and $\tau _ { f } = 4 0$ , consistent with the official implementation. PnP is implemented on both DDIM and direct inversion with a CFG scale of 7.0. 

<table><tr><td rowspan="2"></td><td rowspan="2">Inversion</td><td rowspan="2">Method</td><td colspan="2">Human Preference</td><td colspan="3">Text Alignment</td><td colspan="3">Background Preservation</td></tr><tr><td>Image -Reward↑</td><td>Pick -Score ↑</td><td>CLIP/ Edited ↑</td><td>CLIP/ Whole ↑</td><td>HPS↑</td><td>PSNR↑</td><td>LPIPS/ Whole ↓</td><td>SSIM↑</td></tr><tr><td rowspan="8">SD1.5</td><td>ddim</td><td>PnP</td><td>32.29</td><td>21.53</td><td>22.56</td><td>25.59</td><td>20.55</td><td>22.31</td><td>14.48</td><td>79.58</td></tr><tr><td>ddim</td><td>PnP + Ours</td><td>26.74</td><td>21.49</td><td>22.95</td><td>26.10</td><td>20.52</td><td>22.57</td><td>10.81</td><td>80.14</td></tr><tr><td>direct</td><td>PnP</td><td>40.85</td><td>21.65</td><td>22.68</td><td>25.64</td><td>20.61</td><td>22.46</td><td>13.44</td><td>80.22</td></tr><tr><td>direct</td><td>PnP + Ours</td><td>36.59</td><td>21.62</td><td>22.99</td><td>26.18</td><td>20.62</td><td>22.74</td><td>10.08</td><td>80.70</td></tr><tr><td>ddim</td><td>MasaCtrl</td><td>-13.94</td><td>21.03</td><td>21.20</td><td>24.18</td><td>20.27</td><td>22.31</td><td>14.59</td><td>80.41</td></tr><tr><td>ddim</td><td>MasaCtrl + Ours</td><td>-8.71</td><td>21.01</td><td>21.86</td><td>25.08</td><td>20.24</td><td>21.60</td><td>11.65</td><td>79.30</td></tr><tr><td>direct</td><td>MasaCtrl</td><td>4.77</td><td>21.39</td><td>21.47</td><td>24.54</td><td>20.53</td><td>22.82</td><td>12.21</td><td>82.02</td></tr><tr><td>direct</td><td>MasaCtrl + Ours</td><td>7.79</td><td>21.41</td><td>22.19</td><td>25.53</td><td>20.52</td><td>21.99</td><td>9.87</td><td>80.78</td></tr><tr><td rowspan="6">SD3</td><td>o</td><td>RF-Inversion</td><td>128.0</td><td>22.07</td><td>24.17</td><td>27.26</td><td>20.84</td><td>13.10</td><td>36.10</td><td>57.17</td></tr><tr><td>o</td><td>RF-Inversion + Ours</td><td>132.3</td><td>22.13</td><td>24.72</td><td>29.07</td><td>20.58</td><td>12.90</td><td>37.17</td><td>57.98</td></tr><tr><td>x</td><td>FlowEdit</td><td>103.0</td><td>22.08</td><td>23.35</td><td>26.46</td><td>21.11</td><td>21.44</td><td>10.84</td><td>81.36</td></tr><tr><td>x</td><td>FlowEdit + Ours</td><td>114.8</td><td>22.36</td><td>24.59</td><td>28.47</td><td>21.02</td><td>20.20</td><td>13.86</td><td>78.54</td></tr><tr><td>x</td><td>FlowAlign</td><td>68.00</td><td>21.50</td><td>22.38</td><td>25.65</td><td>20.85</td><td>24.21</td><td>6.89</td><td>86.44</td></tr><tr><td>x</td><td>FlowAlign + Ours</td><td>81.97</td><td>21.63</td><td>23.33</td><td>27.40</td><td>20.67</td><td>22.65</td><td>9.58</td><td>83.95</td></tr></table>


Table 10. Quantitative evaluation of image editing performance our method compared to baseline methods. ImageReward, CLIP, HPS, LPIPS, and SSIM are scaled by $\times 1 0 ^ { 2 }$ and Distance is scaled by $\times 1 0 ^ { 3 }$ . 


# G. Implementation Details

All experiments were performed on two NVIDIA A100 GPUs, and detailed training configurations can be found in Table 11, using a configuration almost identical to that of SoftREPA (Lee et al., 2025a). For inference, we used positive soft tokens $( \psi ^ { + } )$ for both conditional and unconditional generation. In Appendix H, additional results for using both positive and negative soft tokens together are provided. 

<table><tr><td>Models</td><td>lr</td><td>wd</td><td>total batch size (positive:negative)</td><td>iterations</td><td>token init</td><td>optimizer</td><td>lr scheduler</td></tr><tr><td>SD1.5</td><td>1e-3</td><td>1e-4</td><td>16(1:3)</td><td>100,000</td><td>∅</td><td>AdamW</td><td>CosineAnnealingWarmRestarts</td></tr><tr><td>SDXL</td><td>1e-3</td><td>1e-4</td><td>16(1:3)</td><td>1,000</td><td>N(0,0.02)</td><td>AdamW</td><td>CosineAnnealingWarmRestarts</td></tr><tr><td>SD3</td><td>1e-3</td><td>1e-4</td><td>16(1:3)</td><td>100,000</td><td>N(0,0.02)</td><td>AdamW</td><td>CosineAnnealingWarmRestarts</td></tr></table>


Table 11. The implementation details for training.


# H. Additional Results on Ablation Studies

The Number of Soft Tokens. We also study the effect of the number and type of optimized soft tokens. As shown in Table 12, using eight text tokens does not improve performance, consistent with SoftREPA’s observation that performance is stable around four tokens and can degrade with more tokens. Interestingly, optimizing both text and image tokens remains effective in our framework, demonstrating flexibility of our proposed method. 

The Effect of Batch Size. We further analyze the effect of batch size, which determines the size of the in-batch negative prompt pool. As shown in Table 13, increasing the batch size from 4 to 16 improves alignment-related metrics, while increasing it further to 64 yields only marginal gains. This suggests that a larger negative pool is useful up to a point, but our method does not rely on very large batch sizes. 

Sampling Strategy on Negative Tokens. To evaluate the role of negative soft tokens, we compare alternative sampling strategies during inference. Our default configuration uses only positive soft tokens for both the conditional and unconditional predictions in classifier-free guidance (CFG). To assess whether negative soft tokens can further suppress the modeling of complement set or counterfactual concepts of given prompt, we also test a variant where the conditional prediction uses positive soft tokens and the unconditional prediction uses negative soft tokens during CFG. In Figure 8, it shows that injecting negative tokens into the unconditional CFG prediction produces images that may appear visually clean and high-quality, but overly suppress details and background elements which are not mentioned in the captions, resulting in overly simple images with reduced variation and consequently lower human-preference scores. 

<table><tr><td># text (pos) / # image</td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>8 / 0</td><td>94.27</td><td>22.09</td><td>26.95</td><td>27.44</td><td>32.37</td></tr><tr><td>4 / 4</td><td>103.0</td><td>22.42</td><td>27.06</td><td>28.26</td><td>33.11</td></tr><tr><td>4 / 0 (ours)</td><td>103.3</td><td>22.39</td><td>27.00</td><td>28.22</td><td>34.08</td></tr></table>


Table 12. Ablation on the number of soft tokens. # text (pos) denotes the number of $\psi ^ { + }$ , and # image denotes the number of learnable soft image tokens.


<table><tr><td>Batch size (pos:neg)</td><td>ImageReward↑</td><td>PickScore↑</td><td>CLIP↑</td><td>HPSv2↑</td><td>FID↓</td></tr><tr><td>4 (1:1)</td><td>29.67</td><td>21.52</td><td>27.13</td><td>25.31</td><td>24.76</td></tr><tr><td>16 (1:3)</td><td>34.50</td><td>21.59</td><td>27.23</td><td>25.66</td><td>25.94</td></tr><tr><td>64 (1:7)</td><td>32.32</td><td>21.59</td><td>27.11</td><td>25.83</td><td>26.64</td></tr></table>


Table 13. Ablation study on the effect of batch size (SD1.5). pos:neg denotes the number of in-batch negative prompts for each positive text-image pair to calculate the guidance.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/73f286e6993ac2b56b2fb8743755bef9d349a8b76c6d1d0db9b3fe25a20c1925.jpg)



Figure 8. Qualitative results of the ablation study. (1) Effect of negative soft tokens optimization. (2) Effect of applying negative tokens to the unconditional prediction of CFG during sampling.


# I. Additional Qualitative Results with DPO methods

In this section, we present qualitative comparisons of DPO methods with and without soft-token integration. As shown in Figure 9, incorporating soft tokens helps the model generate images that follow the text constraints more faithfully. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-30/1bcacea3-b88c-4a3f-93d4-833246d0f05d/5046b23c6249730ebd106c60c56b7631ce2f761b0cd4f7aec4ae65ee9bac8e25.jpg)



Figure 9. Qualitative evaluation of complementarity with other Diffusion-RL methods on COCO-val5K dataset(Lin et al., 2014).


# J. Trade-off between Text Alignment metrics and FID

In reward or preference-optimization methods for diffusion models, improving text alignment often comes with reduced diversity or coverage, which can negatively affect FID. Our results follow this general trend, but the degradation is modest relative to the base model while alignment metrics improve consistently. As shown in the main quantitative results, our method improves ImageReward, CLIP, and HPSv2 across backbones, with only a moderate FID increase compared to the base model. 

Importantly, compared with SoftREPA, our method achieves a better alignment–fidelity trade-off ( Figure 5). Across SD1.5, SDXL, and SD3, AGSM obtains lower FID than SoftREPA while maintaining strong alignment performance, yielding a better ImageReward–FID Pareto front. Moreover, when combined with diffusion-RL baselines, our soft-token integration consistently improves both FID and alignment ( Table 3), suggesting that the proposed representation-level alignment can complement preference optimization without simply sacrificing image quality. 