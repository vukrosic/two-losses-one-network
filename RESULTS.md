# Results

Every number behind the README and the paper, with the commands that produce it. The lab notes in [LAB-NOTES.md](LAB-NOTES.md) have the full history, including the predictions written before each run.

**Terms used throughout**

- **Shared network**: one network trained on the sum of both losses.
- **Own network** (also "split"): one network per thing, each trained only on its own loss. Both designs have the same total number of weights, the same data and the same training.
- **P**: the smaller-loss thing's error in the shared network ÷ its error in its own network. P − 1 is the extra error.
- **Λ (loss ratio)**: the bigger thing's loss ÷ the smaller thing's loss, each measured with its own network.
- **r (gap)**: how many times bigger the bigger thing's numbers are (for moving things, how many times farther it moves per step).
- **N**: total weights.

Each run is deterministic and takes 2–9 seconds on a laptop CPU. Neighbouring settings differ by about ±0.15 in P, which we treat as the noise level.

---

## 1. The loss-ratio law: P − 1 ≈ 0.15 Λ^0.75

Fitted on 14 driving scenarios (308 points): **P − 1 = 0.149 Λ^0.77** (R² 0.80). Scored with no refit on the other three worlds:

| World | Shared networks | Measured ÷ law (median) | Own fit |
|---|---|---|---|
| Driving: car and pedestrians | 260 | ×1.00 | 0.149 Λ^0.77 |
| Robot arm: slow base, fast arm | 48 | ×1.69 | – |
| Two dots seen in pictures | 33 | ×0.91 | ×0.83–0.97 of the driving law |
| Teacher networks, no physics | 178 | ×0.95 | 0.147 Λ^0.74 (95 points); depth grid 0.117 Λ^0.82 (46 points) |
| **All** | **519** | **377 within ×2.5** | **P − 1 ≈ 0.15 Λ^0.75** |

Points with P − 1 above 0.05. Over settings (each setting = one scenario's whole gap × size grid), the law is within ×2.5 in **31 of 35**. All misses fall below the line: GRU ×0.12, hard teacher (its own error barely improves) ×0.19–0.40, teacher at 300 steps ×0.16.

| Λ | P from the law |
|---|---|
| 1 | 1.15 |
| 10 | 1.84 |
| 100 | 5.7 |
| 1,000 | 27.7 |

**Gap alone does not collapse the worlds.** At fixed size, P − 1 ∝ r^0.9 to r^1.9 with a different constant per world. Λ ≈ r² × (how much harder the bigger thing is to predict relative to its size).

### Driving grid that started it (6,000 decayed steps, car in a× bigger units)

P = pedestrian error, shared ÷ own:

| Size | Weights | r 2.6 | r 5.2 | r 10.4 | r 21 | r 42 |
|---|---|---|---|---|---|---|
| xs | 360 | 1.36 | 1.92 | 9.66 | 19.0 | 27.9 |
| s | 1.1k | 0.95 | 1.33 | 2.58 | 6.75 | 18.5 |
| sm | 3k | 0.97 | 1.22 | 2.03 | 5.25 | 8.59 |
| m | 8k | 1.18 | 1.30 | 1.81 | 3.39 | 6.49 |
| ml | 28k | 1.10 | 1.13 | 1.28 | 1.83 | 3.07 |
| l | 91k | 1.20 | 1.23 | 1.35 | 1.74 | 3.29 |

### Every setting against the law

Driving and arm grids, 6,000 decayed steps unless noted. Gap exponent: P − 1 ∝ r^x at fixed size. Size exponent: P − 1 ∝ N^y at fixed gap. Own-error slope: how fast the smaller-loss thing's own error falls with size. Law offset: measured ÷ 0.149 Λ^0.77, median over the setting.

| Setting | Gap exponent | Size exponent | Own-error slope | Law offset |
|---|---|---|---|---|
| Driving MLP, Adam | 1.45 | −0.52 | 0.21 | ×0.95 |
| Driving, fine gap grid | 1.18 | −0.82 | −0.03 | ×0.55 |
| Driving, 18k steps | 1.57 | −1.07 | 0.22 | ×1.23 |
| Driving, constant learning rate | 1.06 | −0.40 | 0.00 | ×1.42 |
| Driving, learning rate 0.003 | 1.58 | −0.80 | 0.04 | ×0.88 |
| Driving, learning rate 0.0003 | 1.10 | −0.48 | 0.26 | ×0.76 |
| Driving, batch 64 | 1.53 | −0.50 | 0.25 | ×0.82 |
| Driving, 250 clips | 1.36 | −0.61 | 0.08 | ×1.18 |
| Driving, faster pedestrian | 1.31 | −0.62 | 0.12 | ×1.82 |
| Driving, weaving pedestrian | 1.60 | −0.55 | 0.17 | ×1.65 |
| Two pedestrians (first / second) | 1.37 / 1.21 | −0.53 / −0.58 | 0.16 / 0.15 | ×1.07 / ×0.67 |
| Driving, car is the smaller-loss thing | 1.18 | −0.64 | 0.36 | ×0.35 |
| Arm / faster arm | 1.38 / 1.24 | −1.50 / −1.48 | −0.02 / −0.03 | ×1.34 / ×1.57 |
| Driving, absolute-error loss | 1.35 | −0.50 | 0.39 | ×0.93 |
| Driving, Muon on hidden matrices | 1.20 | −0.54 | 0.47 | ×1.69 |
| MLP depth 3 / depth 4 | 1.76 / 1.40 | −0.31 / −0.27 | 0.41 / 0.42 | ×1.82 / ×2.44 |
| GRU | 0.94 | −0.39 | 0.29 | ×0.12 |
| Per-thing loss scaling | flat | – | 0.21 | – |
| Dot world, Chase | 0.96 | +0.07 | 0.36 | ×0.83 |
| Dot world, Chase, 250 clips | 0.96 | −0.04 | 0.29 | ×0.85 |
| Dot world, Switch | 1.22 | −0.46 | 0.05 | ×0.91 |

Teacher world (3,000 steps): fresh data and separate inputs 0.147 Λ^0.74 (95 points); hard teacher with fresh data ×0.30, with fixed data ×0.19; separate inputs ×1.33; depth 2 / 3 / 4 offsets ×0.92 / ×1.06 / ×1.38.

**Direction checks** (`experiments/monotone.py`, a break must exceed 0.15 in P): P rises with the gap in 1,057 of 1,066 neighbouring pairs (the 9 breaks are depth 1 and per-thing scaling, which is flat by design). P falls with size in 520 of 558; most breaks are at the smallest gap, where P ≈ 1.

---

## 2. The cause: a loss weight a² does what units ×a does

| a² | Driving, loss × a² | Driving, units × a | Dots, loss × a² | Dots, units × a |
|---|---|---|---|---|
| ¼ | 1.06 | 1.18 | 1.30 | 1.31 |
| 4 | 2.22 | 1.81 | 2.29 | 2.43 |
| 16 | 4.25 | 3.39 | 3.66 | 3.78 |
| 64 | 7.03 | 6.49 | – | – |

Driving size m, 6,000 decayed steps. Dots: Chase stranger, width 32.

Loss weighting also reproduces the size effect of unit scaling (driving, gap ×4, sizes xs → ml): weight 16 gives P 12.2 / 5.9 / 5.5 / 4.3 / 2.5; units ×4 give 19.0 / 6.8 / 5.3 / 3.4 / 1.8. So scaling the inputs is not what drives it.

### Hand split (dot world, 16,000 weights in total)

| Weights given to the chasing dot (the smaller loss) | Robot error | Chasing dot error | Total loss |
|---|---|---|---|
| 50% (even split) | 6.2% | 10.7% | 0.0091 |
| 30% | 4.5% | 13.2% | 0.0079 |
| 20% (lowest total loss) | 3.8% | 14.5% | 0.0075 |
| 10% | 3.5% | 19.0% | 0.0083 |
| **One shared network** | **4.9%** | **18.3%** | **0.0096** |

The shared network acts like separate networks that give the smaller thing 10% of the weights, half of the loss-optimal 20%.

---

## 3. Theory check

`P = ((1 + Λ^(1/β)) / 2)^(β − 1)` for the best rank-k linear network, when each thing's i-th direction is worth c·i^−β of loss. Checked by exact top-k selection over two lists of 200,000 directions (`experiments/rankcheck.py`, under one second):

| Λ | β = 2, k = 40 | β = 2, k = 400 | β = 3, k = 400 | β = 4, k = 400 | Formula, β = 2 / 3 / 4 | Measured law |
|---|---|---|---|---|---|---|
| 10 | 2.16 | 2.08 | 2.47 | 2.67 | 2.08 / 2.49 / 2.68 | 1.84 |
| 100 | 5.82 | 5.50 | 7.86 | 8.97 | 5.50 / 7.96 / 9.01 | 5.7 |
| 1000 | 13.2 | 16.1 | 30.2 | 36.4 | 16.3 / 30.3 / 36.3 | 27.7 |

The proof of the general result (at the best weights, raising one loss's weight can only lower that loss and raise the other) is in section 7.1 of the [paper](paper/paper.html).

---

## 4. Training length

P at 6,000 / 12,000 / 24,000 steps, each run with the learning rate decayed to zero:

| Setting | 6k | 12k | 24k |
|---|---|---|---|
| Driving, r 5.3, size m | 1.30 | 1.33 | 1.41 |
| Driving, r 10.5, size m | 1.81 | 1.81 | 1.72 |
| Driving, r 21, size m | 3.39 | 3.04 | 3.08 |
| Driving, r 21, size ml | 1.83 | 2.02 | 2.03 |
| Dots, Chase, width 32, robot ×4 | 3.78 | 3.82 | 3.70 |
| Arm, size m, gap ×4 | 2.75 | 2.20 | 1.67 |
| Driving, size s, gap ×4 (3.39 at 3k) | 6.75 | 8.01 | 4.52 |

The arm breaks because its base reaches an error floor (own error 0.016 / 0.010 / 0.009). Driving size s falls at 24k for the same reason.

**Fixed slowdown** (`experiments/slowdown.py`, `experiments/fslow.py`): the shared network's error on the smaller thing after S steps equals the own network's error after S/F steps, F constant over training. F measured at 6k predicts the shared error at 12k and 24k within ×0.80–1.15 (driving m), ×0.83–1.40 (driving s), ×1.11–1.34 (arm). If the own error falls like S^−α, P = F^α. F does not follow one law across worlds (teacher world: F against Λ has R² 0.05); P does.

---

## 5. Network size

Size exponent at fixed Λ (`experiments/regime.py`, fit log(P − 1) = a + b log Λ + c log N per setting):

| Group | Settings | c (size at fixed Λ) | b (Λ exponent) |
|---|---|---|---|
| Driving and arm, all MLP settings | 16 | −0.22 to −0.73 (mean −0.48) | 0.57–0.86 |
| Teacher, own error still falling with size (3,000 steps) | 7 | −0.08 to +0.38 (mean +0.12) | 0.46–0.94 |
| Teacher, step-limited (300 steps) | 1 | −0.45 | 0.53 |
| Teacher, slower target stuck (hard teacher, fixed data) | 1 | −0.03 | 0.46 |

Correlation of c with how fast the own error falls with size: 0.36 over 25 settings. No single threshold across worlds.

**Three direct switches, one knob each:**

| Switch | Own networks gain from size? | P against size |
|---|---|---|
| Teacher, 300 steps | little | 1.94 / 2.40 / 1.49 / 1.18 / 1.21 (N^−0.45) |
| Same teacher, 3,000 steps | yes | 1.78 / 1.79 / 2.49 / 3.32 / 4.10 (N^+0.30) |
| Dots, Switch stranger, gap ×4, widths 12 → 196 | little | 3.63 → 2.17 → 1.27 (N^−0.62) |
| Dots, Chase stranger | yes | flat (N^+0.07) |
| Dots, Chase with a random push (floor), gap ×4, widths 8–128 | little | 1.53 / 1.55 / 1.51 / 1.35 / 1.19 (N^−0.29) |

Teacher at gap ×8, widths 16 / 32 / 64 / 128 / 256 = 1,224 / 3,464 / 11,016 / 38,408 / 142,344 weights. Own error at 300 steps: 0.398 / 0.295 / 0.276 / 0.258 / 0.230. At 3,000 steps: 0.243 / 0.185 / 0.127 / 0.084 / 0.059.

**Explanations we tested and rejected** for driving's size effect: separate inputs for each thing (teacher `SEP=1`), finite data (teacher `NTRAIN=2000`), input scaling (loss weight instead of units), the step slope α, and a law in the slowdown F.

---

## 6. The fix

- **Per-thing loss scaling** (`NORM=agent`: each thing's loss divided by its own typical squared change): P flat across gaps 2.6 to 42 at every size; P − 1 above 0.1 in 1 of 25 runs. Scaling per number (`NORM=1`) breaks training: numbers that almost never change get a huge weight.
- **One network per thing** removes the harm when each part is big enough and sees what its thing reacts to. Separate networks lost when one part got too few weights in a tiny network, when half the weights went to a thing that never moves, and when a chasing dot's network did not see the robot's moves.

---

## 7. Constant learning rate

Driving, same 6,000 steps:

| Same 6,000 steps, but… | Pedestrian P | Car P |
|---|---|---|
| constant learning rate | 1.81 | 0.80 |
| learning rate decays to zero | 1.30 | 0.95 |
| 4× bigger batch | 1.19 | 1.08 |
| evaluate the average of the last quarter of weights | 1.23 | 1.03 |

| World | r | Smaller-loss thing, constant rate | Smaller-loss thing, decayed |
|---|---|---|---|
| Dot world, chasing dot | 2.05 | 1.9 | 2.0 |
| Arm, medium speed | 3.0 | 2.4 | 1.06 |
| Arm, fast | 7.4 | 4.1 | 1.34 |
| Driving, default pedestrian, 8k weights | 5.2 | 1.8 | 1.41 |

---

## How to run

Everything runs on a CPU with Python 3, PyTorch, NumPy and Matplotlib. Run scripts from inside `experiments/`. Each run appends one JSON line to a log in the current folder (`drive.jsonl`) or next to the script (`teacher.jsonl`).

```bash
cd toy
# driving, car in 4x bigger units, size m, decayed learning rate: shared vs own networks (about 4 s each)
DECAY=1 CARSCALE=4 python3 drive.py split mlp m
DECAY=1 CARSCALE=4 python3 drive.py single mlp m
# -> pedestrian error 0.179 (own) vs 0.606 (shared): P = 3.39
# teacher world, gap x8, width 32 (about 2 s each)
A=8 python3 teacher.py split 32
A=8 python3 teacher.py single 32
# fits and checks on the saved runs
python3 lawfit.py; python3 teachfit.py; python3 dotgrid.py; python3 regime.py; python3 rules.py; python3 monotone.py
python3 rankcheck.py     # theory check, no training
cd .. && python3 charts.py   # every figure in figures/
```

The dot world needs a trained encoder first: `python3 train_one.py` (see its docstring), then `REPO=dots python3 two_vs_one.py ...`.

### Knobs

**`experiments/drive.py NET ARCH size [steps]`**, NET = `split` | `single`, ARCH = `mlp` | `tf` | `gru`, size = `xs` … `l` or a width.

| Variable | Meaning |
|---|---|
| `WORLD` | `drive` (car + pedestrians, default) or `arm` (slow base + fast arm tip) |
| `CARSCALE=a`, `SCALETHING=i` | measure thing i (default 0 = car) in a× bigger units; the gap grows by a, the motion stays the same |
| `LW=w0,w1,...` | loss weight per thing |
| `NORM` | `agent`: divide each thing's loss by its own typical squared change (the fix); `1`: per number |
| `LOSS=l1` | absolute error instead of squared error |
| `PED`, `PEDS` | pedestrian speed; several pedestrians |
| `CURVE=a` | pedestrians weave sideways |
| `PNOISE=s` | random push on each pedestrian every step |
| `ARMGAIN` | how fast the arm turns |
| `NCLIP` | training clips (default 1,000) |
| `DECAY=1`, `LR`, `BATCH`, `AVG=f` | cosine decay to zero; peak rate; batch; evaluate the weight average over the last f of steps |
| `DEPTH` | hidden layers of the MLP |
| `OPT`, `MUONLR` | `adam`, `sgd` or `muon` (Muon on hidden matrices, Adam elsewhere) |

**`experiments/teacher.py split|single width [steps]`**

| Variable | Meaning |
|---|---|
| `A` | scale of the bigger target (the gap) |
| `HF`, `HS` | hidden width of the bigger / smaller target's teacher (larger = harder) |
| `SNOISE` | noise on the smaller target (an error floor) |
| `SEP=1` | each teacher reads its own half of the input |
| `NTRAIN` | fixed training set of this size instead of fresh samples |
| `DEPTH`, `LR`, `BATCH`, `OPT`, `MUONLR`, `TAG` | as above |

**`experiments/two_vs_one.py`** (dot world): `LATSCALE=a` scales the robot's targets, `RW`/`SW` are the robot / stranger loss weights, `DECAY=1`, `NORM=1`, `NOISE` pushes the stranger, `MODE` changes the stranger's behaviour.

---

## Literature

| Work | What it does | What it does not do |
|---|---|---|
| Kendall, Gal, Cipolla 2018 (arXiv 1705.07115) | learns task weights from uncertainty; notes that the best weighting depends on units | no law for the harm to the smaller task |
| GradNorm, Chen et al. 2018 (arXiv 1711.02257) | rebalances gradient sizes across tasks | no equal-size comparison with one network per task |
| PCGrad, Yu et al. 2020 (arXiv 2001.06782) | removes conflicting gradient parts | same |
| PopArt, van Hasselt et al. 2016 (arXiv 1602.07714) | normalizes value targets across games | same |
| DreamerV3, Hafner et al. 2023 (arXiv 2301.04104) | squashes large targets (symlog) in a world model | same |
| Switch Transformer, Fedus et al. 2021 (arXiv 2101.03961) | router balancing loss with weight 0.01 | no measure of how well the smaller term is learned |
| DeepSeek-V3, 2024 (arXiv 2412.19437) | multi-token prediction loss with weight 0.3 then 0.1; router balanced without an auxiliary loss | same |
| Scene Transformer (2106.08417), Wayformer (2207.05844), MotionLM (2309.16534), UniAD (2212.10156), GAIA-1 (2309.17080) | one network for every agent in a driving scene | no comparison with per-type networks at equal size |
| Diffusion Policy, Chi et al. 2023 (arXiv 2303.04137) | rescales each action dimension to [−1, 1] | – |
| Eckart and Young 1936 | best low-rank approximation keeps the largest directions | used in our Result 2 |
| Saxe, McClelland, Ganguli 2014 (arXiv 1312.6120) | deep linear networks learn large directions first | related to our fixed-slowdown finding; not derived here |
