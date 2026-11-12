# Literature: one shared predictor vs one predictor per moving thing

Checked 2026-11-12 (extended web searches plus earlier notes in TWO_VS_ONE.md). Second, wider check the same day for the two-losses paper; results in the last two sections.

## Who predicts several moving things with one network

| Work | What one network predicts | Per-thing networks compared at equal size? |
|---|---|---|
| Scene Transformer (Ngiam et al., ICLR 2022, arXiv 2106.08417) | all agents in a driving scene, jointly | no |
| Wayformer (Nayakanti et al., 2022, arXiv 2207.05844) | agent futures from one shared encoder | no |
| MotionLM (Seff et al., ICCV 2023, arXiv 2309.16534) | vehicles, cyclists and pedestrians, one model | no |
| UniAD (Hu et al., CVPR 2023, arXiv 2212.10156) | detection and motion of vehicles and pedestrians in one network | no |
| GAIA-1 (Hu et al., 2023, arXiv 2309.17080) | whole driving video, every object at once | no |
| DreamerV3 (Hafner et al., 2023, arXiv 2301.04104) | whole scene in one latent world model | no |
| VP-LSTM (Bi et al., ICCV 2019) | separate LSTMs for vehicles and pedestrians | no same-size test against a shared model |

A review of pedestrian prediction in mixed traffic (arXiv 2308.06419) notes that vehicle-focused benchmarks such as nuScenes rarely report pedestrian and vehicle results separately.

## Known pieces

| Piece | Source | Done? |
|---|---|---|
| A task with bigger loss crowds out a smaller one in a shared network | GradNorm (Chen et al., 2018, arXiv 1711.02257); uncertainty weighting (Kendall et al., 2018, arXiv 1705.07115), which notes that the best task weighting depends on the measurement scale | done, for tasks |
| With MSE, a target at a bigger scale swamps a smaller one; scaling a target by z scales its gradient by z²; a limited model then predicts the small target poorly | Khakhar & Buckman, "Neural Regression for Scale-Varying Targets" (arXiv 2211.07447), section 2 and Figures 1–2 | **done, qualitatively** (no same-size test, no law) |
| Tasks with bigger reward scale dominate a shared RL agent; fix by normalizing each task's targets | PopArt (van Hasselt et al., 2016, arXiv 1602.07714); Hessel et al., AAAI 2019 (arXiv 1809.04474) | **done, and the fix is the same idea as our per-thing scaling** |
| Loss-scale balancing fixes (log loss, learned per-task scale) | DB-MTL (arXiv 2308.12029), IMTL-L, Lee et al. ICCV 2021, BPGS (arXiv 2608.21653) | done |
| "A shared model acts like a single-task model with a fraction f of the weights", f independent of model size | Fernandes et al., ICML 2023, "Scaling Laws for Multilingual NMT" (arXiv 2302.09650), Eq. 8–12: L_i(N; p) = L_i(N·f_i(p); 1), with f_i(p) ≈ p for En→XX | **done, for data-sampling weight p** (same idea as our hand split) |
| Interference grows when the model is small relative to data; tuned by each task's data share | Shaham et al., ACL 2023 (arXiv 2212.07530) | done, for data share |
| Additive competition/synergy term added to single-modality scaling laws | Aghajanyan et al., ICML 2023 (arXiv 2301.03728) | done, for modalities |
| Gradient descent learns big directions first; small ones are delayed | Saxe et al., 2014 (arXiv 1312.6120); Lampinen & Ganguli, ICLR 2019 (arXiv 1809.10374) | done (basis of our claim 4) |
| A limited network keeps the most valuable units first (quanta ordered by frequency) | Michaud et al., NeurIPS 2023 (arXiv 2303.13506) | done, same picture as our rank-k theory |
| Rank-k fit to a weighted sum keeps the top-k scaled directions | Eckart–Young / reduced-rank regression | textbook |
| Imbalance correlates with task gradient norms; MTL worse than single-task | arXiv 2509.23915; IJCAI 2025 imbalance ratio | correlation only, no law |
| High-variance parts dominate a shared prediction target | Orthogonal JEPA (arXiv 2608.20065), argued, not measured | argued only |

## Not found in any search (12 searches, 2026-11-12)

| Piece | Done? |
|---|---|
| Harm predicted from the loss ratio Λ alone, measured from each thing's own network: P − 1 ≈ 0.15 Λ^0.75, one law across four worlds and several architectures | **not done** |
| Same-size test of one shared predictor vs one per moving thing in world models | **not done** |
| Closed form P = ((1+Λ^(1/β))/2)^(β−1) for a rank-k linear network with power-law spectra | **not done** (the parts are textbook; the formula is a short new derivation) |
| The shared network's curve equals the own-network curve slowed by a fixed factor F along the step axis | **not done** |
| Harm shrinks only when the own networks stop gaining from size (Fernandes find f independent of size, in a data-weight setting) | **not done** |
| Constant learning rate inflating the measured harm up to 3× | **not done** |

The qualitative effect (the smaller-scale target is learned worse) and the fix (normalize each target) are known. The paper must cite Khakhar & Buckman, PopArt and Fernandes et al., and claim only the rows in this last table.
