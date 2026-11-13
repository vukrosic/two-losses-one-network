# Lab notes

> [!NOTE]
> These are the raw research notes, kept in the order they were written. Predictions were written before each run. Early sections use the speed gap r and a yes/no rule; findings 19–22 replace them with the loss-ratio law in the README. Script paths such as `experiments/drive.py` refer to the `experiments/` folder of this repo.

## 1 vs 2 predictor networks

The tracking file for the paper question. Update it after every step.

## The question

When a world model predicts another agent, should that agent get its own predictor network, separate from the robot's, or should both share one network? In which cases, and why?

## Paper plan (2026-11-12)

**Title:** When One Network Predicts Two Moving Things, the Slower One Is Predicted Worse

**Subtitle:** Tested on robots and on a car with a pedestrian, in MLPs and transformers. A bigger network does not fix it.

**Short title (for slides and posts):** Shared Predictors Neglect Slower-Moving Objects

**The rule in one sentence:** when one network predicts how two things move, the one that moves less is predicted worse than it would be by its own network, once that network is big enough to learn it. The gap grows with the speed difference, and a bigger network does not close it.

- **Scope of "always":** the same number of weights. It held in every test where each separate network was big enough to learn its thing (findings 9–15): two toys, MLP, transformer and GRU, squared and absolute error, 2–4 things, and robot parts. It broke for very small networks (GRU 0.9k–5k, L1 at 1.1k).
- **The word "agent" is left out of the title,** because readers outside the field may not know it. In the paper, "thing" means a robot, a car, a pedestrian or any other moving object the world model tracks.
- **Untested:** pixel input in the driving world, large models and real data. Robot parts and 3–4 things are now tested (finding 15).

**Main question:** why does one shared world model predict an agent about r× worse when that agent's numbers move r× less than the other agent's, and does this hold beyond the toy? The 1-vs-2 networks choice is the practical part: measure r, then decide whether to split, scale or grow.

**Scope:** two agents in total, the robot and one other agent. Both agents can be made more complex.

**Claims that held in every case so far (the headline):**
1. In one plain shared network, the agent that moves less is the one that gets worse. This holds in both directions: a slower robot suffers the same way (finding 12).
2. A bigger network does not remove this with the plain loss: the ratio stays about 2× from 3k to 122k weights (finding 9), and with 3× longer training (finding 10).
3. *Dot toy only; it failed in the driving toy (finding 14):* scaling each error by its typical one-step change removes about half of the gap. With scaled errors, size removes the rest: one network ties with two (gap ≤0.02) for every stranger tried (findings 9–11). In the driving toy, two networks still won with per-agent scaling.
4. Splitting hurts only when the fixed division of weights is wrong: the robot's half too small in a tiny network, or half the weights wasted on a stranger that does not move (finding 9, the Statue).
5. A split network must see everything its agent reacts to. Without the robot's moves, the split loses (Chaser 5 steps ahead, Instant). This is a setup rule, not a weakness of splitting.

**The size of the effect, "r× worse", in three tiers:**
- **Tier 1, learnable agents (the core):** agents the network gets nearly right on its own (2-network miss below about 0.05 at 122k): Chaser at all speeds, Flee, the robot-speed cases. "r× worse" fits with mean error about 0.02. "Learnable" means what the network achieves, not "deterministic": Instant is deterministic and still plateaus at 0.06–0.07.
- **Tier 2, known randomness:** we set the random push ourselves, so the unpredictable share u is known. "u + r × (2-network miss − u)" fits to 0.01 (finding 12).
- **Tier 3, hard agents (Instant, Orbit, Switcher):** reported, not corrected. Their penalty is below r× (1.0–1.5× vs r = 1.6–2.1). Claim only "r× is roughly the most a slower agent loses". A correction would need u picked after seeing results, which is not a fair test.

**Known exceptions and limits (must be in the paper):**
- Off by up to about 25% in some tier-1 cases: Chaser very slow (r 4.7, actual 3.6×), robot slower (r 1.7, actual 2.3×, which also breaks the "at most r×" bound), equal movement (both still 1.3× worse at 16k).
- The faster agent gains about 15% in one network.
- Training choices change the size: 250 clips gives 1.5× instead of 2.1×, and SGD gives 1.0–1.4×.
- Why the penalty is r and not r² is unexplained. Adam is not the reason (finding 13).
- Untested: more than two agents, encoders trained together with the predictor, bigger models, real data, planning.

**Paper steps:**
1. Explain the r with a tiny model that can be worked out on paper.
2. Fair tests: predict the penalty for new agents before running, then on a harder setting with bigger models.
3. Show it matters: the slower agent's worse prediction should cause more planning failures.

**Industry angle:** in a driving world model the car moves a lot and a pedestrian moves little, so a shared model should predict the pedestrian worse, which matters for safety. The fix is cheap: scale each agent's error, or split. The law applies directly to models that keep agents separate (object-centric or per-agent tracking); whether it shows up in models that mix everything into one image code is open. The first test is the small driving toy below.

**Status as of 2026-11-12:** strong enough for a workshop paper. For ICML it needs scale, the explanation of r, a planning result, and a clear step beyond the multi-task loss-imbalance literature.

## Setup (experiments)

- **Dots:** the robot (we steer it) and one other agent, the stranger. The stranger is the Statue (still), the Circler (patrol), the Chaser (chase) or the Switcher (switch: patrol or chase, flipping at hidden times every 6 to 14 steps).
- **Picture and encoders:** a 16×16 picture, each dot in its own colour. Each dot has its own encoder (16 numbers each, 32 in all), so the pixels are already split by dot.
- **Predictors:** both predict the next 32 numbers from the last 3 frames plus the robot's move.
  - 2 networks: robot net width 49 and stranger net width 49, 16,104 weights.
  - 1 network: width 77, 16,125 weights.
- **Training:**
  - The encoder is trained first (600 steps, split predictor), then frozen.
  - Fresh predictors are trained on its numbers for 6,000 steps on 1,000 clips, with Adam at lr 0.001.
- **Measure:** the miss, meaning the share of a dot's one-step change the predictor misses on 32 held-out clips (lower is better).
- **Limits:** runs ≤10 s on the MacBook.

## Findings so far

1. **Short training misleads.**
   - At 600 steps the predictors had not finished learning: the robot miss is 0.49, but 0.15 after 4,000 steps on the same encoder.
   - With only 48 clips, the predictors memorise the training clips (Chaser stranger: train 0.01, held-out 0.38).
2. **At the same size, 2 networks beat 1 on the stranger, and the robot is predicted just as well.** Robot / stranger miss:

   | Stranger | 2 networks | 1 network | 1 network 2.7× wider | 1 network, stranger error ×4 | 1 network 2.7× wider, stranger ×4 |
   |---|---|---|---|---|---|
   | Circler | 0.04 / 0.00 | 0.03 / 0.01 | 0.02 / 0.00 | 0.03 / 0.00 | – |
   | Chaser | 0.05 / 0.09 | 0.04 / 0.17 | 0.03 / 0.12 | 0.06 / 0.12 | 0.03 / 0.08 |
   | Switcher | 0.05 / 0.23 | 0.05 / 0.27 | 0.03 / 0.25 | 0.08 / 0.25 | 0.05 / 0.24 |

3. **It is not the move, and not the encoder.**
   - Letting the stranger net see the robot's move gives the same numbers.
   - An encoder trained with a 1-network predictor gives the same Chaser result (0.09 vs 0.17).
4. **Why, for the Chaser:**
   - The robot's numbers move 4× more per step (0.104 vs 0.025).
   - The shared network ends with equal raw error on both (0.0034 vs 0.0033), which is 4× worse for the stranger relative to its movement.
   - The two errors do not pull against each other: the gradient cosine on the shared layers is about 0.
5. **It is a trade-off for 1 network.** Lowering the robot's weight in the loss buys stranger accuracy with robot accuracy (Chaser):

   | Robot weight | Robot / stranger miss |
   |---|---|
   | 1 | 0.04 / 0.17 |
   | 0.25 | 0.06 / 0.12 |
   | 0.05 | 0.12 / 0.09 |
   | 0 | 1.01 / 0.07 |

   The 2 networks sit off this curve.
6. **Not slower training alone.** On the Chaser, the 2 networks reach 0.13 by 2,000 steps; 1 network needs 20,000 steps to reach the same.
7. **The Switcher barely benefits.** Its hidden flips dominate its error. Its raw error stays 3× the robot's, and late in training its pull on the shared layers is the larger one.
8. **Scaling each error removes about half the gap, not all of it.**
   - Method: each of the 32 numbers' error is divided by that number's typical squared one-step change on the training clips (`NORM=1`).
   - Robot / stranger miss:

     | Stranger | 2 networks | 1 network | 1 network, scaled | 1 network, scaled, 2.7× wider |
     |---|---|---|---|---|
     | Circler | 0.04 / 0.00 | 0.03 / 0.01 | 0.03 / 0.00 | – |
     | Chaser | 0.05 / 0.09 | 0.04 / 0.17 | 0.06 / 0.13 | 0.03 / 0.08 |
     | Switcher | 0.05 / 0.23 | 0.05 / 0.27 | 0.07 / 0.25 | – |

   - The prediction was 0.10 to 0.12; the result was 0.13. With 2 networks, scaling changes nothing (Chaser 0.05 / 0.09).
   - So on the Chaser, about half the gap is counting (0.17 → 0.13). The rest is the stranger sharing the network's units with the robot, and only extra width removes it (0.08).

**Simple one sentence explanation:** one shared network focuses on whatever moves most, so a slower, predictable agent is predicted better with its own network.

## Size sweep (Chaser): prediction written before running

- **Question:** if both setups get bigger together (same total weights at each size), does the sharing cost go away?
- **Test:** five sizes, each pair matched:

  | 2 networks (each) | 1 network | Weights |
  |---|---|---|
  | 12 | 20 | 3.1k |
  | 24 | 39 | 6.7k |
  | 49 | 77 | 16k |
  | 98 | 149 | 42k |
  | 196 | 289 | 122k |

  Run with the scaled loss (counting removed) and the plain loss, at 6,000 steps on 1,000 clips.
- **Prediction:**
  - With the scaled loss, the gap is biggest at the smallest size and shrinks to ≤0.01 at 122k weights, where both reach the same floor.
  - With the plain loss, a gap stays at every size, because counting does not go away with size.
- **Result (finding 9):** both predictions held. Stranger miss:

  | Weights | 2 networks | 1 network, scaled | 1 network, plain | Plain ÷ 2 networks | Scaled ÷ 2 networks |
  |---|---|---|---|---|---|
  | 3.1k | 0.18 | 0.26 | 0.35 | 1.9× | 1.4× |
  | 6.7k | 0.13 | 0.19 | 0.25 | 1.9× | 1.5× |
  | 16k | 0.09 | 0.13 | 0.17 | 1.9× | 1.4× |
  | 42k | 0.06 | 0.08 | 0.12 | 2.0× | 1.3× |
  | 122k | 0.04 | 0.05 | 0.08 | 2.0× | 1.25× |

  - With the plain loss, 1 network misses about 2× as much of the stranger as 2 networks at every size. Growing both does not close the ratio. 1 network at 122k (0.08) matches 2 networks at about 22k, so it needs about 5× the weights.
  - With the scaled loss, the ratio falls from 1.4× to 1.25× and the absolute gap from 0.08 to 0.01. The sharing part fades with size; the counting part does not.
  - The 2 networks give the same numbers with or without scaling at every size checked (12, 24, 49, 98), so the 122k row uses the scaled run.
  - The robot goes the other way at the smallest size: 2 networks 0.46, 1 network 0.24 at 3.1k. From 42k up, both are ≤0.03.
- **Rule for the Chaser:** the plain-loss stranger cost of sharing is a fixed factor (about 2×) that size does not remove. Scaling each error removes about half of it, and the rest shrinks as the network grows.

## Does "scaled + big = no need to split" always hold? (Chaser) Predictions written before running

The claim under test: with scaled errors and a big network (122k), 1 network does as well as 2 on the stranger (0.05 vs 0.04).

| # | Change | Prediction |
|---|---|---|
| A | Train 3× longer at 16k | Plain: the 1-network gap shrinks but stays above 1.5×. Scaled: 1 network closes to within 0.02 |
| B | Less data: 250 clips instead of 1,000 | Both get worse. The big 1 network overfits more, so the scaled tie breaks (gap ≥0.03) |
| C | Stranger speed: 0.01, 0.03 (normal), 0.04 (same as robot), 0.06 | Plain: the gap grows as the stranger slows and is gone when the stranger moves as much as the robot. Scaled + big: tie at every speed |
| D | Predict 5 steps ahead | The job gets harder, so 122k acts like a small network again: the scaled gap comes back (≥0.03) |

**Results (finding 10).** Stranger miss for 2 networks / 1 network scaled / 1 network plain, with the ratio to 2 networks in brackets.

| Test | Size | 2 networks | 1 scaled | 1 plain | Prediction held? |
|---|---|---|---|---|---|
| Normal (finding 9) | 16k | 0.09 | 0.13 (1.4×) | 0.17 (1.9×) | – |
| Normal (finding 9) | 122k | 0.04 | 0.05 (1.25×) | 0.08 (2.0×) | – |
| A: 3× longer training | 16k | 0.07 | 0.09 (1.3×) | 0.14 (2.0×) | Scaled yes; plain no: the 2× did not shrink |
| B: 250 clips | 122k | 0.08 | 0.10 (1.25×) | 0.12 (1.5×) | No: both overfit about equally and the scaled tie held (gap 0.02) |
| D: 5 steps ahead, stranger net sees the moves | 16k | 0.04 | 0.05 (1.25×) | 0.07 (1.75×) | – |
| D: 5 steps ahead, stranger net sees the moves | 122k | 0.02 | 0.02 (1.0×) | 0.03 (1.5×) | No: the tie held |

C, the stranger's speed. The robot's numbers move 0.322 per step (typical size) in every row:

| Stranger speed | Stranger moves | Robot moves × more | 2 networks (16k) | 1 scaled (16k) | 1 plain (16k) | 122k: 2 / scaled / plain |
|---|---|---|---|---|---|---|
| 0.01 | 0.068 | 4.7× | 0.03 | 0.05 (1.7×) | 0.12 (4.0×) | 0.02 / 0.03 / 0.06 (3×) |
| 0.03 (normal) | 0.157 | 2.1× | 0.09 | 0.13 (1.4×) | 0.17 (1.9×) | 0.04 / 0.05 / 0.08 (2×) |
| 0.04 (robot's speed) | 0.193 | 1.7× | 0.11 | 0.15 (1.4×) | 0.18 (1.6×) | – |
| 0.06 | 0.264 | 1.2× | 0.15 | 0.18 (1.2×) | 0.20 (1.3×) | – |

- **The plain penalty roughly equals how many times more the robot's numbers move than the stranger's.** 4.7× more movement gives 4.0×; 2.1× gives 1.9×; 1.7× gives 1.6×; 1.2× gives 1.3×. Five steps ahead, 1.9× more movement gives 1.75×. This is the counting effect.
- **Scaled + big held in every test:** the gap was ≤0.02 at 122k with a slow stranger, with 250 clips, and 5 steps ahead.
- **New catch, 5 steps ahead:** if the stranger's own network does not see the robot's moves, the 2 networks lose (16k: 0.14 vs 0.07 for 1 network). The Chaser follows the robot, so over 5 steps it needs to know where the robot went. One step ahead, the moves made no difference (finding 3).
- The equal-speed setting (0.04) still has the robot moving 1.7× more, because the robot's numbers move more per pixel than the stranger's. "Speed" should be read as movement in the model's numbers, not on screen.
- Code: `PSPEED=` sets the stranger's speed and `AHEAD=` the steps ahead in experiments/two_vs_one.py. The world default is unchanged.

**Rule for the Chaser so far:**
1. With the plain loss, a shared network misses about R× as much of the stranger, where R is how many times more the robot's numbers move. Size, longer training and more steps ahead do not remove it.
2. Scaling each error removes most of it. What is left fades with size or training.
3. A separate stranger network must get every input the stranger depends on, here the robot's moves when predicting further ahead.

## Do the Chaser rules hold for other strangers? Predictions written before running

The rules from the Chaser:
1. With the plain loss, a shared network misses about R× as much of the stranger, where R is how many times more the robot's numbers move.
2. Scaled errors + a big network (122k) tie with 2 networks (gap ≤0.02).
3. A split stranger network must see everything the stranger reacts to.

How freely a stranger can be defined: any rule for its next position. Four knobs matter for this question:
- how much it moves (R);
- how predictable it is (hidden mode, noise);
- what it reacts to (nothing, the robot's position, the robot's current move);
- how hard its motion is to learn (straight, circle, chase).

| Stranger | What is new | Prediction |
|---|---|---|
| Circler | Fully predictable, ignores the robot | Every miss ≤0.01 at 122k, so nothing to choose |
| Switcher | Hidden mode flips | The flips set a floor (about 0.2) that no setup removes. Rule 1 applies only above the floor: plain ratio < R. Rule 2 holds |
| Instant | Reacts to the robot's current move (the split stranger net gets the moves) | Same as the Chaser: plain ratio ≈ R, scaled + big ties |

**Results (finding 11).** Stranger miss for 2 networks / 1 network scaled / 1 network plain (6,000 steps, 1,000 clips). Instant uses the Chaser's encoder.

| Stranger | R | 16k | Plain ÷ 2 nets | 122k | Plain ÷ 2 nets |
|---|---|---|---|---|---|
| Circler | 1.8 | 0.000 / 0.00 / 0.007 | – | not run | – |
| Chaser | 2.1 | 0.09 / 0.13 / 0.17 | 1.9× | 0.04 / 0.05 / 0.08 | 2.0× |
| Instant | 2.1 | 0.12 / 0.14 / 0.17 | 1.4× | 0.07 / 0.08 / 0.09 | 1.3× |
| Switcher | 1.6 | 0.23 / 0.25 / 0.27 | 1.2× | 0.23 / 0.24 / 0.24 | 1.0× |

- **Rule 2 held for every stranger:** scaled + 122k is within 0.01 of 2 networks for the Chaser, Instant and Switcher.
- **Rule 1 held only for the Chaser.** For Instant the plain penalty is 1.3–1.4× with R = 2.1; for the Switcher it is 1.0–1.2× with R = 1.6. Prediction for the Switcher held (ratio < R, floor about 0.23); prediction for Instant failed.
- **Pattern:** the penalty is large only when the stranger is easy to predict on its own (the Chaser's 2-network miss is 0.04 at 122k). It shrinks when the stranger has a part that is hard for every setup: hidden flips (Switcher, 0.23) or reacting to the current move (Instant, 0.07). It is zero when the stranger is trivially easy (Circler). One reading, not yet tested: the counting penalty multiplies only the learnable part of the stranger's error.
- **Rule 3 held for Instant one step ahead:** without the moves the split stranger net misses 0.22, with them 0.12.

**Rules after four strangers:**
1. Scaled errors + a big enough network make splitting unnecessary for every stranger tested.
2. With the plain loss, sharing costs the stranger at most about R×, and the full R× only when the stranger is easy to predict.
3. A split network must see everything its agent reacts to.
4. Splitting hurts only when the fixed division of weights is wrong (finding 9 robot at 3k; the Statue).

## Looking for a general rule (two agents). Formulas written before running

Per agent i: P_i = its miss in 1 plain network ÷ its miss in 2 networks (the sharing penalty). r_i = how many times more the *other* agent's numbers move than agent i's (robot: r = 1/R, stranger: r = R).

| Formula | Says |
|---|---|
| F1 | P_i = r_i when r_i > 1, else 1 |
| F2 | P_i = r_i² when r_i > 1, else 1 (the loss counts squared changes) |
| F3 | Inside 1 plain network, both agents end with equal raw error, so stranger miss ÷ robot miss = R² |

New knobs, all on a frozen Chaser encoder, 16k weights, 6,000 steps, 1,000 clips:
- robot speed: 0.01, 0.02, 0.04 (normal), 0.08;
- randomness: a random push on the stranger each step, std 0.005, 0.01, 0.02;
- new strangers: Flee (moves away from the robot) and Orbit (circles around the robot).

Predictions:
- A slower robot (R < 1) makes the robot the one that suffers in 1 network, and the stranger stops suffering.
- More randomness shrinks P for the stranger.
- Flee and Orbit follow F1 like the Chaser.
- F1 beats F2 (the Chaser speed data already pointed that way).

**Results (finding 12).** 13 conditions at 16k weights, plain loss. "Slower" means the agent whose numbers move less (r > 1). Two formulas were added after seeing the first results:
- F4: miss in 1 network = f + r × (miss in 2 networks − f), with f = the 2-network miss at 122k.
- F4n: the same, but f = the share of the stranger's movement that is random push (known from the noise setting), for the noise rows only.

Miss of the slower agent in 1 network, actual vs each formula:

| Case | r | 2 nets | Actual 1 net | F1 | F2 | F4 | F4n |
|---|---|---|---|---|---|---|---|
| Chaser | 2.05 | 0.087 | 0.167 | 0.178 | 0.366 | 0.131 | – |
| Chaser slow (0.01) | 4.74 | 0.033 | 0.118 | 0.156 | 0.740 | 0.082 | – |
| Chaser 0.04 | 1.67 | 0.114 | 0.179 | 0.190 | 0.317 | – | – |
| Chaser fast (0.06) | 1.22 | 0.147 | 0.198 | 0.179 | 0.219 | – | – |
| Robot fast (0.08) | 3.68 | 0.069 | 0.229 | 0.254 | 0.934 | 0.152 | – |
| Robot slow (0.01): the *robot* is slower | 1.67 | 0.040 | 0.093 | 0.067 | 0.111 | 0.059 | – |
| Robot 0.02 (equal movement) | 1.07 | 0.145 | 0.185 | 0.155 | 0.166 | – | – |
| Flee | 2.85 | 0.035 | 0.094 | 0.100 | 0.284 | 0.063 | – |
| Instant | 2.09 | 0.118 | 0.171 | 0.247 | 0.516 | 0.170 | – |
| Orbit | 2.12 | 0.132 | 0.194 | 0.280 | 0.594 | 0.153 | – |
| Noise 0.005 | 1.99 | 0.142 | 0.214 | 0.283 | 0.563 | 0.176 | 0.227 |
| Noise 0.01 | 1.86 | 0.266 | 0.333 | 0.495 | 0.921 | 0.282 | 0.344 |
| Noise 0.02 | 1.52 | 0.528 | 0.580 | 0.804 | 1.224 | 0.489 | 0.570 |

Mean absolute error over the slower agents: no penalty 0.069, F1 0.060, F2 0.326, F4 0.044 (10 cases). F4n on the 3 noise rows: 0.011.

What this says:
- **F2 is wrong everywhere:** the penalty follows the movement ratio r, not r². F3 also failed: in 1 network the two misses do not come out in the ratio R².
- **F1 (penalty = r) works for clean, easy strangers:** Chaser at all speeds, Flee and both robot-speed cases have errors of 0.006–0.038, mean about 0.02.
- **F1 fails when part of the stranger cannot be learned:** noise rows (errors 0.07–0.22), Instant (0.08) and Orbit (0.09). There the penalty hits only the learnable part.
- **F4n, "r× on the learnable part", is the best formula when the unlearnable part is known:** errors 0.010–0.013. Using the 122k 2-network miss as that part (F4) works for Instant (0.001) but undershoots the clean strangers, because their 122k miss is still learnable, not a true floor.
- **The rule is symmetric:** when the robot moves less than the stranger, the robot suffers (0.040 → 0.093) and the stranger barely changes (0.125 → 0.139). The faster agent usually gains a little in 1 network (P ≈ 0.85), because it gets more than half the shared units.
- **Small extra cost even at equal movement:** at r ≈ 1 both agents are about 1.3× worse in 1 network (16k). This is the shared-capacity part, and it matches the scaled-loss gap at this size.

**Rule candidate (two agents):** in one plain network, the agent whose numbers move r× less ends with miss ≈ u + r × (its 2-network miss − u), where u is the part of its motion nobody can predict. The faster agent ends a little better than with 2 networks. Scaling each error removes the r×; size removes the rest.

## What is the unpredictable part u? Predictions written before running

Test: give every network the true state at time t (both positions, the orbit angle, the switch mode) on top of its usual inputs (`ORACLE=1`), at 122k weights. What the oracle 2-network setup still misses is the part nobody can predict from the state.

| Stranger | Prediction for the oracle 2-network miss |
|---|---|
| Noise 0.01 | about 0.18, the known random share |
| Chaser | ≤0.03 |
| Orbit | well below its normal 0.11, because the orbit angle is hidden in the normal inputs |
| Instant | about its normal 0.07: nothing is hidden, so its leftover is hardness, not missing information |

Extra test: why r and not r²? The loss counts squared changes, so plain gradient descent should weight the stranger r² less. Adam divides each weight's step by its typical gradient size, which may turn r² into r. Prediction: with SGD (momentum 0.9) instead of Adam, the Chaser penalty at 16k moves from 1.9× toward r² ≈ 4.2×.

**Results (finding 13).** 2-network stranger miss at 122k:

| Stranger | Normal inputs, 6k steps | Normal, 18k steps | With true state (oracle), 6k steps | Prediction held? |
|---|---|---|---|---|
| Chaser | 0.045 | – | 0.04 | yes |
| Noise 0.01 | 0.25 | – | 0.26 | partly: the known random share is 0.18, the oracle leaves 0.26 |
| Orbit | 0.11 | 0.10 | 0.07 | partly: the hidden angle explains 0.04 of 0.11 |
| Instant | 0.07 | 0.06 | 0.07 | yes: nothing hidden |

- **Instant's leftover is not missing information and not too little training.** It sits at 0.06–0.07 with the true state given and with 3× more training. Orbit's leftover is part hidden angle (0.04) and part the same kind of plateau (0.07).
- **Every stranger has a plateau the 122k predictor does not get below:** 0.04 for the Chaser, 0.07 for Instant and Orbit with the oracle, about 0.08 on top of the random share for noise. Its cause is still unknown. The encoder's own position error is one candidate.
- **Formula fit, choosing u per kind:** u = 0 for clean strangers, u = the random share for noise, u = the 18k-step plateau for Instant (0.06) and Orbit (0.10). Mean absolute error over the 13 slower agents: 0.018 (F1 0.060). The u values for Instant and Orbit were picked after seeing the data, so this fit is not yet a fair test.
- **The SGD prediction failed.** Chaser at 16k with SGD gives penalties of 1.04× (lr 0.05), 1.25× (lr 0.2) and 1.4× (lr 0.6; 2 networks 0.12, 1 network 0.17). Adam at the same 2-network quality (2,000 steps, 0.13) gives 1.6×. SGD does not push the penalty toward r², so Adam is not why the penalty is r rather than r². Why it is r is still open.
- Wall times: the oracle runs took 15–17 s and the 18k-step runs 50 s, over the 10 s budget, because of 3× longer training and another program's 5 Python workers loading the CPU.

## Driving toy: car + pedestrian, MLP and transformer. Predictions written before running

Setup (`experiments/drive.py`, no world.py change):
- **Car** (the robot): x, y, vx, vy; throttle and steer inputs; smooth random driving, lane keeping, brakes when the pedestrian is ahead and close.
- **Pedestrian** (the other agent): walks across the road at 0.006 per step, turns back at the far kerb, slows to a stop when the car is within about 0.15.
- Both are tracked as 4 numbers each, like an object-tracking stack (no pixels, no encoder). Input: last 3 frames of both + the car's 2 controls.
- Movement ratio r = 5.2 (the car moves 5.2× more per step).
- **Main measure:** error ÷ the error of a constant-velocity guess (position += velocity). Below 1 means it beats simple extrapolation. Also measured near the car (distance < 0.25, 43% of steps).
- **Architectures:** MLP (as before; split is two MLPs that both see all inputs) and a transformer with one token per agent (2 attention blocks, shared weights across tokens; split is two transformers each predicting one agent). Weights matched within 5%.
- MLP 6,000 steps; transformer 2,000 steps (to fit about 10 s); compare only within one architecture.

| # | Prediction |
|---|---|
| D1 | MLP, plain loss: the pedestrian's error in 1 network is 2–5× its 2-network error at every size; the car's error stays about equal |
| D2 | MLP, scaled loss + largest size: tie within 1.25× |
| D3 | The 1-network plain penalty is largest near the car, where the pedestrian stops |
| D4 | Transformer, plain loss: same direction (pedestrian worse in 1 network), smaller penalty, 1.5–3×, because each agent has its own token and output |
| D5 | Transformer, scaled loss: tie within 1.25× |
| D6 | Pedestrian speed 0.003 / 0.012 / 0.024 (r about 10 / 2.6 / 1.3): the MLP plain penalty follows r and is about 1× at r ≈ 1.3 |

**Two setup fixes made during the runs (before reading the 1-vs-2 comparison):**
- **Plain loss:** the first plain runs used raw numbers of about 0.003 per step, so Adam's built-in epsilon slowed learning; the split nets gave different results under plain and per-agent loss, which should not happen. Fix: divide every error by one shared constant (the mean typical squared change), which keeps the car/pedestrian balance. The table uses the fixed runs.
- **Per-number scaling (as in the dot toy) breaks here:** the pedestrian's velocity numbers almost never change, so dividing by their tiny typical change gives them huge weight; the 1-network car error went up to 27× worse. Added **per-agent scaling** (`NORM=agent`): each agent's error divided by that agent's mean typical change. That is the version the law needs.

**Results (finding 14).** Error ÷ the error of a constant-velocity guess, held-out, 1 network ÷ 2 networks (above 1 = splitting wins):

| Architecture, loss | Size | Pedestrian | Pedestrian near the car | Car |
|---|---|---|---|---|
| MLP, plain | 1.1k | 2.0× (0.21 → 0.41) | 1.7× | 0.96× |
| MLP, plain | 8k | 1.6× (0.20 → 0.31) | 1.2× | 0.87× |
| MLP, plain | 90k | 1.6× (0.25 → 0.39) | 1.0× | 1.2× |
| MLP, per-agent scaled | 1.1k / 8k / 90k | 1.4× / 1.5× / 1.3× | 1.2× / 1.1× / 1.3× | 2.0× / 1.9× / 1.75× |
| Transformer, plain | 6k | 1.5× (0.68 → 1.04) | 1.3× | 1.2× |
| Transformer, plain | 27k | 3.5× (0.31 → 1.07) | 3.0× | 0.93× |
| Transformer, per-agent scaled | 6k / 27k | 0.94× / 2.7× | 0.76× / 2.2× | 6.9× / 4.5× |

Pedestrian speed sweep, MLP 8k, plain:

| Pedestrian speed | r | Pedestrian | Car |
|---|---|---|---|
| 0.003 | 10.4 | 1.9× | 0.83× |
| 0.006 | 5.2 | 1.6× | 0.87× |
| 0.012 | 2.5 | 1.3× | 1.2× |
| 0.024 | 1.2 | 1.2× | 3.2× |

- **D1 held weakly.** The pedestrian is 1.6–2.0× worse in 1 plain MLP and the car is about equal, but the penalty is well below r = 5.2.
- **D2 failed.** No tie at 90k. With per-agent scaling the shared MLP is worse on both agents (pedestrian 1.3×, car 1.75×). In this toy, two networks win at every size tried.
- **D3 failed.** The penalty is smaller near the car (1.0–1.7×) than overall. Most of the pedestrian's extra error is away from the car: walking and turning at the kerb, not stopping for the car.
- **D4: same direction, but the penalty is larger, not smaller.** The shared transformer gives the pedestrian 1.5× (6k) and 3.5× (27k) more error. At 2,000 steps it was 2.5× / 3.3× / 4.0× at 6k / 27k / 135k, though all transformers were then worse than constant velocity. One token per agent did not protect the slower agent.
- **D5 failed.** Per-agent scaling moves the cost onto the car: 4.5–6.9× worse in 1 transformer.
- **D6: the penalty grows with r, but much slower than r** (r 1.2 → 1.2×, 2.5 → 1.3×, 5.2 → 1.6×, 10.4 → 1.9×). At r = 1.2 the fast pedestrian is the harder agent, and the car becomes the one that pays (3.2×).
- **What carries over from the dot toy:** the slower-moving agent pays in one shared network, in both an MLP and a transformer, and the penalty grows with r.
- **What does not carry over:** "penalty ≈ r×" (here about r^0.3 for the MLP) and "scaled loss + big network ties" (here splitting still wins). The exact size of the law depends on the setup.
- Wall times: MLP 4–45 s, transformer 24–96 s, over the 10 s budget, because the machine's load average was 7–10 from other programs.

## Knob grid: why r^0.3 in driving, and which knobs change the effect. Predictions written before running

**First check, done before the grid (no training):** is the driving penalty small because most motion is straight-line?
- No. The straight-line share is 99.6% for both car and pedestrian.
- The "hard part" (change minus a constant-velocity guess) is 4.4× bigger for the car, yet the penalty was only 1.6–1.8×.
- **New candidate:** the pedestrian's hard part is *event-like*. 52% of it falls in the top 5% of steps (stops and turnarounds), against 20% for the car. Rare events may get less of the loss's tilt.

`experiments/drive.py` was rewritten to take more knobs. The default numbers moved a little because the random draws are now in a different order (medium MLP pedestrian penalty 1.8× instead of 1.6×), so every grid row is compared with fresh default runs.

Measured before training (rms per step, from the simulator):

| Setting | Slower thing | r (raw move) | Hard-part ratio | Slower thing's top-5% share |
|---|---|---|---|---|
| Default driving | pedestrian | 5.2 | 4.4 | 0.52 (event-like) |
| Weaving pedestrian, CURVE 0.5 | pedestrian | 4.9 | 1.7 | 0.14 (spread) |
| CURVE 1 | pedestrian | 4.3 | 0.93 | 0.13 |
| CURVE 2 | pedestrian | 3.0 | 0.47 | 0.13 |
| Arm robot, base vs tip, ARMGAIN 0.02 | base | 1.4 | 1.7 | 0.19 |
| ARMGAIN 0.06 | base | 3.0 | 4.5 | 0.19 |
| ARMGAIN 0.15 | base | 7.4 | 16.6 | 0.19 |
| Car + 2 pedestrians (speeds 0.003, 0.012) | slow pedestrian | 8.2 | 13 | 0.61 |

| # | Knob | Prediction |
|---|---|---|
| G1 | Weaving pedestrian (spread hard part) | The penalty follows the hard-part ratio: CURVE 0.5 about 1.7×; CURVE 1 and 2 about 1×, with the car now paying |
| G2 | Arm robot: slow base and fast tip of one robot, no events | Parts behave like separate things. Base penalty close to the hard-part ratio: 1.5× / 3–4.5× / ≥5× for gains 0.02 / 0.06 / 0.15 |
| G3 | Car + 2 and 3 pedestrians | The slowest pedestrian pays most, then the next slowest; the car about 1× |
| G4 | Random push on the pedestrian (PNOISE 0.0002), sizes s / m / l | The penalty falls toward 1 as size grows, because the pedestrian's learnable part shrinks toward its random floor |
| G5 | Absolute-error loss (L1) instead of squared | Much smaller penalty, ≤1.3×: the L1 gradient does not grow with the error's size, so there is no counting tilt |
| G6 | Recurrent network (GRU) | Same direction as MLP and transformer: pedestrian worse in 1 network, 1.5–3× |

**Results (finding 15).** Plain loss and matched weights unless noted. P = the error in 1 network ÷ the error in 2 networks, measured against a constant-velocity guess on held-out clips.

| Test | Size | Slower thing's P | Other thing(s) P | Prediction |
|---|---|---|---|---|
| Default driving (fresh) | 1.1k / 8k / 90k | pedestrian 2.0× / 1.8× / 2.0× | car 0.95× / 0.79× / 0.94× | reference |
| G1 weaving, CURVE 0.5 / 1 / 2 | 8k | 1.24× / 1.5× / 1.27× | car 1.25× / 1.3× / 1.35× | failed |
| G2 arm robot, gain 0.02 / 0.06 / 0.15 (r 1.4 / 3.0 / 7.4) | 8k | base 0.94× / 2.4× / 4.1× | tip 0.75× / 0.67× / 1.0× | mostly held |
| G3 car + 2 pedestrians (0.003, 0.012) | 9k | slow ped 1.28× | other ped 0.98×, car 1.7× | partly |
| G3 car + 3 pedestrians (0.003, 0.006, 0.012) | 11k | slowest 1.84×, middle 1.87× | fastest ped 1.35×, car 1.42× | partly |
| G4 random push 0.0002 | 1.1k / 8k / 90k | 1.06× / 1.01× / 1.07× | car 1.3× / 1.6× / 1.0× | partly |
| G5 absolute-error loss (L1) | 1.1k / 8k | 0.60× / 1.47× | car 1.08× / 0.98× | failed |
| G6 GRU | 0.9k / 5k / 48k | 0.48× / 0.94× / 1.40× | car 0.70× / 1.29× / 0.83× | failed at small sizes |

- **G1 failed, and with it both explanations of r^0.3.** A weaving pedestrian has a spread-out hard part, yet its penalty went *down* (1.8× → 1.2–1.5×), not up. With CURVE 1 and 2 its hard part is bigger than the car's, and it still pays about as much as the car (both 1.25–1.5×). Neither the hard-part ratio nor "events get less tilt" predicts the driving penalty. Why driving gives about r^0.3 is still open.
- **G2: parts of one robot behave like separate things.** The slow base pays 2.4× at r 3.0 and 4.1× at r 7.4, while the fast tip is equal or better in 1 network. The size is between the dot world and driving, about r^0.7–0.8. At r 1.4 there is no penalty (0.94×).
- **G3: with more things, the slowest ones pay most, and the car starts paying too** (1.4–1.7×). In the 2-pedestrian case the slowest pedestrian is worse than constant velocity in both setups (errors 0.87–1.96), so its numbers are weak.
- **G4: with a random push, P ≈ 1 at every size**, because the random floor dominates (errors 0.75–0.93 of the constant-velocity error). This matches "the penalty hits only the learnable part". The predicted fall with size was not visible, because the floor already dominated at the smallest size.
- **G5 failed: L1 does not remove the effect.** At 8k the pedestrian still pays 1.47× (plain 1.8×). At 1.1k the shared network wins (0.60×), but there the split pedestrian network was badly trained (0.61).
- **G6: the first breaks of the main rule.** With a GRU the shared network predicts the pedestrian *better* at 0.9k (0.48×) and slightly better at 5k (0.94×), and the rule returns only at 48k (1.40×). At 0.9k the split pedestrian GRU (width 8) is worse than constant velocity (1.07), so it is too small to learn its thing at all.
- **Corrected rule:** the slower thing is predicted worse in a shared network *once each separate network is big enough to learn its own thing*. When the separate networks are too small (GRU 0.9k–5k, L1 at 1.1k), sharing can win. This joins the earlier dot-world exception (the robot's half too small at 3k).
- Wall times: 3–60 s per run, load average 6–8 from other programs.

## Making P predictable: the capacity-split theory. Written before running

**Theory.** Treat the shared network as if it splits its resources (weights, or training effort) between the things so that the total loss is as small as possible. Each thing i has its own error curve E_i(w): its error when a separate network of w weights predicts it, in loss units. The shared network then picks w_fast + w_slow = W to minimise E_fast + E_slow, and the slower thing ends up with error E_slow(w_slow*). The separate-networks baseline gives it about W/2.

- If each curve is a power law, E_i = a_i · w^(−β), the best split gives w_fast / w_slow = (a_fast / a_slow)^(1/(1+β)), and
  **P = ((1 + (a_fast/a_slow)^(1/(1+β))) / 2)^β.**
- If a_fast / a_slow grows like r², then for large r, P ≈ r^(2β/(1+β)) / 2^β. **The exponent of r is 2β/(1+β)**, where β is how fast a thing's own error falls as its network grows.
  - β = 1 gives P ∝ r (dot world), β = 0.6 gives r^0.75 (arm), β = 0.18 gives r^0.3 (driving).
- What it explains for free: P does not depend on total size W (finding 9); scaling each thing's loss makes a_fast = a_slow, so P → 1 (finding 8); a random floor that no network can lower is not part of the a·w^(−β) part (finding 12: "r× on the learnable part"); L1 makes a_fast/a_slow grow like r, not r², so the exponent halves.
- What it cannot explain: features that help both things (the shared net can beat the split, like the arm tip at 0.67×).

**Predictions.**

| # | Test | Prediction |
|---|---|---|
| T1 | Driving: each thing's own error vs width (split only, widths 4–100) | The pedestrian's curve is shallow, β ≈ 0.1–0.3 (its split error was 0.161 at 1.1k and 0.166 at 8k weights) |
| T2 | Predict the driving single-net errors (s / m / l) from the T1 curves, no fitting to single runs | Predicted P within 25% of measured P (2.0 / 1.8 / 2.0) |
| T3 | Same for the arm world at three gains, and the dot world | Predicted P within 25% of measured; arm β ≈ 0.5–0.7, dot β ≈ 0.7–1 |
| T4 | Across all settings, predicted vs measured P | Correlation of log P ≥ 0.8 |

**Results (finding 16): the driving and arm penalties were mostly undertraining.**

- **T1 failed in an unexpected way.** At 6,000 steps with a constant learning rate, the driving pedestrian's own error did not fall with width at all. Across widths 4–200 it jumped between 0.15 and 0.38 of the constant-velocity error, with no trend. So the earlier driving networks were limited by training, not by size.
- **Long training with a decaying learning rate changes the driving numbers a lot.** Matched 8k nets with cosine decay to 0:

| Training | Car P | Pedestrian P | Pedestrian error (split / shared) |
|---|---|---|---|
| 6,000 steps, constant lr 1e-3 (all earlier runs) | 0.79 | 1.8 | 0.166 / 0.300 |
| 6,000 steps, decay from 1e-3 | 0.95 | 1.30 | 0.179 / 0.233 |
| 6,000 steps, decay from 3e-3 / 1e-2 | 1.32 / 1.11 | 1.18 / 1.18 | 0.111 / 0.131, 0.080 / 0.094 |
| 24,000 steps, decay from 1e-3 | 1.39 | 1.41 | 0.029 / 0.041 |
| 24,000 steps, decay from 3e-3 | 1.43 | 1.15 | 0.027 / 0.031 |

  Errors fall 6× with long training (0.17 → 0.03). With well-trained networks, the shared network is about 1.15–1.4× worse on *both* things, and the pedestrian-only penalty is mostly gone.

- **All three worlds, well trained** (24,000 steps, cosine decay from 1e-3, 8k weights; the dot world uses 16k):

| World | r | Slower thing's P | Faster thing's P | Before (6,000 steps, constant lr) |
|---|---|---|---|---|
| Dot world, chaser | 2.05 | 2.0 (0.14 / 0.07) | 0.75 | 1.92 |
| Arm, gain 0.06 | 3.0 | 1.06 | 0.85 | 2.4 |
| Arm, gain 0.15 | 7.4 | 1.34 | 0.74 | 4.1 |
| Driving, pedestrian 0.006 | 5.2 | 1.41 | 1.39 | 1.8 |
| Driving, pedestrian 0.003 | 10.4 | 1.89 | 1.48 | not run |
| Driving at 1.1k weights, pedestrian 0.006 | 5.2 | 1.41 | 1.18 | 2.0 |

  - The dot-world penalty is robust: 1.7× at 6,000 decayed steps and 2.0× at 24,000.
  - In the arm and driving worlds it shrinks a lot. It still grows with r within each world (arm 1.06 → 1.34, driving 1.4 → 1.9), and the slower thing still pays more than the faster one in 5 of 6 rows.
  - **So the exponents r^0.3 (driving) and r^0.7–0.8 (arm) from findings 14–15 were measured on undertrained networks. They are not properties of those worlds.** The weaving, multi-pedestrian, L1, GRU and transformer runs used the same 6,000-step constant-lr protocol and need rerunning before they can be trusted.

- **T2/T3: the capacity-split theory, tested directly in the dot world.** Separate networks with uneven widths at the same 16k total, 6,000 decayed steps:

| Robot / chaser widths | Chaser's share of weights | Robot miss | Chaser miss | Total loss |
|---|---|---|---|---|
| 49 / 49 (even) | 50% | 0.062 | 0.107 | 0.00907 |
| 63 / 32 | 30% | 0.045 | 0.132 | 0.00792 |
| 69 / 23 (loss-optimal) | 20% | 0.038 | 0.145 | **0.00752** |
| 76 / 12 | 10% | 0.035 | 0.190 | 0.00832 |
| 1 shared network (width 77) | — | 0.049 | 0.183 | 0.00960 |

  - The shared network's errors look like a split that gave the chaser **about 10%** of the weights (chaser 0.183 vs 0.190).
  - The loss-optimal split gives it about 20%, with a chaser error of 0.145. So the shared network neglects the slower thing **more than an optimal split would**. It also has a higher total loss than every split.
  - The capacity-split theory gets the direction right but underpredicts the size. Predicting from extrapolated width curves gave P 1.0–1.3 against a measured 2.0.

- **What is now predictable:**
  - The direction: slower thing worse, in 5 of 6 well-trained rows.
  - The growth with r within a world.
  - The dot-world size, P ≈ r.
- **What is not:**
  - The size of P across worlds. It depends on how well trained the networks are.
  - Why the dot world keeps a large penalty after long training while the arm and driving worlds do not.
- Wall times: 25–116 s per run at load average 5–6 (over the 10 s target).

## Does the rule survive long training? Every earlier setting rerun. Written before running

Long training = 24,000 steps, cosine decay from lr 1e-3 to 0 (transformer and dot world: 12,000 steps if 24,000 is too slow; noted per row).
**The rule holds in a row** if the slower thing's P ≥ 1.1 **and** it is larger than every faster thing's P.

| # | Setting | Prediction |
|---|---|---|
| L1 | Driving, sizes 1.1k / 8k / 90k | Holds at 1.1k (already 1.41 vs 1.18); at 8k it already failed narrowly (1.41 vs 1.39) and will stay a near-tie; 90k near-tie |
| L2 | Weaving pedestrian, CURVE 0.5 / 1 / 2 | Fails at CURVE 1 and 2: the weaving pedestrian's hard part is as big as the car's |
| L3 | Car + 2 pedestrians, car + 3 pedestrians | Holds for the slowest pedestrian |
| L4 | Random push on the pedestrian | Fails (P ≈ 1): the random floor dominates |
| L5 | Arm gain 0.02 (r 1.4) | Fails (P ≈ 1): almost no speed gap |
| L6 | Absolute-error loss, 1.1k / 8k | Holds at 8k, smaller than squared loss |
| L7 | GRU 0.9k / 5k / 48k | Holds at 48k only |
| L8 | Transformer 6k / 27k | Holds at both |
| L9 | Dot world: patrol, switch, flee, orbit strangers | Holds for the easy-to-learn ones (flee, orbit), weak for patrol and switch, as with short training |

**Which part of long training matters? (factor test, driving default, 8k MLP).** Long training differs from short in (a) the learning rate decaying to 0, which removes the end-of-training jitter in the weights, and (b) 4× more steps. Short-run stand-ins: a bigger batch (less jitter per step) and weight averaging over the last 25% of steps (removes jitter without changing training).

| # | Recipe | Prediction for pedestrian P / car P |
|---|---|---|
| F0 | 6k steps, constant lr (old) | known: 1.8 / 0.79 |
| F1 | 6k, decay | known: 1.30 / 0.95 |
| F2 | 24k, constant lr (more steps, jitter kept) | stays near short: ≥1.6 / ≤1.0 |
| F3 | 24k, decay | known: 1.41 / 1.39 |
| F4 | 6k, constant lr, batch 1024 | moves toward long: ≤1.5 |
| F5 | 6k, constant lr, weights averaged over last 25% | moves toward long: ≤1.5 |

Prediction: jitter is the main factor. The jitter adds about the same absolute error to every output, which is a large share of the slow pedestrian's small error and a tiny share of the car's.

Overall prediction: the rule holds in about two thirds of rows. Where it fails, the speed gap is small, the slower thing's motion is mostly unlearnable, or the separate networks are too small.

**Results (finding 17).**

Factor test (driving, 8k MLP):

| Recipe | Pedestrian P | Car P | Pedestrian own error |
|---|---|---|---|
| F0 6k constant lr (old) | **1.81** | 0.80 | 0.166 |
| F1 6k decay | 1.30 | 0.95 | 0.179 |
| F4 6k constant, batch 1024 | 1.19 | 1.08 | 0.214 |
| F5 6k constant, weights averaged over last 25% | 1.23 | 1.03 | 0.126 |
| F2 24k constant | 1.31 | 1.31 | 0.111 |
| F2b 24k constant, averaged | 1.74 | 1.27 | 0.043 |
| F3 24k decay | 1.41 | 1.39 | 0.029 |
| F3 24k decay from 3e-3 | 1.15 | 1.43 | 0.027 |

- **The old pedestrian-only penalty was mostly end-of-training jitter.** In the same 6,000 steps, anything that removes the jitter (decay, a bigger batch, weight averaging) drops the pedestrian penalty from 1.81 to 1.19–1.30 and raises the car's to about 1. A plausible mechanism: the car's large, noisy gradients shake the shared hidden features, and that shaking shows up in the pedestrian's small outputs. A separate pedestrian network is not shaken by the car.
- **After removing jitter, the driving penalty depends on recipe details.** Across 7 jitter-free recipes the pedestrian's P is 1.15–1.74 (median 1.30) and the car's is 0.95–1.43 (median 1.31). The pedestrian pays more in only 4 of 7. So in default driving there is **no reliable pedestrian-only penalty**. The shared network is about 1.3× worse on both things.
- In the dot world the penalty does not depend on jitter (2.0 at 24k decayed steps). There the capacity grab (finding 16: the chaser gets about 10% of the weights) is the mechanism.

Every setting with long training (24k decayed steps; transformer and dot world 12k; dot chase 24k). ✓ = slower thing's P ≥ 1.1 and larger than every faster thing's P.

| # | Setting | r | Slower thing's P | Faster thing's P | Holds? | Prediction right? |
|---|---|---|---|---|---|---|
| 1 | Driving MLP 1.1k | 5.2 | 1.41 | 1.18 | ✓ | yes |
| 2 | Driving MLP 8k | 5.2 | 1.41 | 1.39 | ✗ (tie) | yes |
| 3 | Driving MLP 90k | 5.2 | 1.03 | 1.52 | ✗ | yes (tie predicted, car pays) |
| 4 | Slow pedestrian 0.003, 8k | 10.4 | 1.89 | 1.48 | ✓ | — |
| 5 | Weaving 0.5 | 4.9 | 1.19 | 1.15 | ✓ (barely) | yes |
| 6 | Weaving 1 | 4.3 | 1.08 | 1.16 | ✗ | yes |
| 7 | Weaving 2 | 3.0 | 1.02 | 1.55 | ✗ | yes |
| 8 | Car + 2 pedestrians | 8.2 | 1.39 | car 1.36, ped 1.14 | ✓ (barely) | yes |
| 9 | Car + 3 pedestrians | 7.6 | 1.87 | 1.21–1.27 | ✓ | yes |
| 10 | Random push | 5.2 | 1.04 | 1.49 | ✗ | yes |
| 11 | L1, 1.1k | 5.2 | 1.66 | 2.11 | ✗ | yes (no rule at tiny size) |
| 12 | L1, 8k | 5.2 | 1.21 | 1.54 | ✗ | no |
| 13 | GRU 0.9k | 5.2 | 1.80 | 0.88 | ✓ | no (predicted fail) |
| 14 | GRU 5k | 5.2 | 1.70 | 1.13 | ✓ | no (predicted fail) |
| 15 | GRU 48k | 5.2 | 1.22 | 1.02 | ✓ | yes |
| 16 | Transformer 6k | 5.2 | 1.05 | 1.31 | ✗ | no |
| 17 | Transformer 27k | 5.2 | 3.20 | 1.90 | ✓ | yes |
| 18 | Arm gain 0.02 | 1.4 | 1.06 | 1.01 | ✗ | yes |
| 19 | Arm gain 0.06 | 3.0 | 1.06 | 0.85 | ✗ | no |
| 20 | Arm gain 0.15 | 7.4 | 1.34 | 0.74 | ✓ | yes |
| 21 | Dot, chaser | 2.05 | 2.0 | 0.75 | ✓ | yes |
| 22 | Dot, flee | 2.85 | 3.07 | 0.77 | ✓ | yes |
| 23 | Dot, orbit | 2.12 | 1.47 | 0.82 | ✓ | yes |
| 24 | Dot, switch | 1.60 | 1.18 | 1.02 | ✓ | no (predicted weak) |
| 25 | Dot, patrol | 1.81 | > 10 (0.000 → 0.005) | 0.68 | ✓ | yes |

- **The rule holds in 15 of 25 well-trained settings, so it does not always hold.**
- **It holds in all 5 dot-world settings** and whenever the speed gap is large (r ≥ 7.4: rows 4, 8, 9, 20).
- **It fails in four kinds of setting:**
  - small gaps (arm, r ≤ 3);
  - motion that cannot be learned (random push);
  - a slower thing whose hard part is as big as the faster one's (weaving 1 and 2);
  - default driving with an MLP or L1, where the shared network is about 1.3–1.5× worse on both things.
- Predictions were right in 18 of 24 rows (row 4 had no prediction).
- Wall times: 20–165 s per pair at load average 5–6.

## Where is the capacity threshold? Size sweep, long training. Written before running

Question: the abstract says the effect grows as the network gets too small to learn both things. At what size does it start, and does "too small" mean "own network still improving with size"?

**Capacity test:** a size is short of capacity when making the thing's own network 4× bigger still cuts its held-out error by more than 25%.

Runs, all long training (cosine decay to 0):
- **Driving MLP**, car + pedestrian, 24k steps, single widths 8, 20, 40, 75, 150, 284 (about 0.3k to 90k weights). Matched split widths.
- **Dot world chaser**, 12k steps, split widths 12, 24, 49, 98, 196 with matched single widths.

Predictions:
- C1. Driving: the pedestrian's P minus the car's P grows as the network shrinks. Below 1k weights the rule holds clearly (gap > 0.3). Above 20k it fails.
- C2. Dot world: P stays ≥ 1.5 for widths 12 to 49. At width 196 it falls below 1.3, because the chaser's own error stops falling there.
- C3. In both worlds the rule holds where the slower thing's own network is short of capacity by the test above, and fails where it is not.
- If the dot world keeps P ≈ 2 at width 196 while its own error has flattened, capacity is not the cause of the dot-world effect.

### Finding 18: the rule holds while the slower thing's own network still improves with size

Driving MLP, car + pedestrian (r 5.2), long training. Rows at 1.1k, 8k and 90k come from finding 17.

| Weights | Pedestrian's own error | Drop from previous size | Pedestrian P | Car P | Holds? |
|---|---|---|---|---|---|
| 0.4k | 0.091 | — | 3.97 | 0.56 | ✓ |
| 1.1k | 0.063 | 31% | 1.42 | 1.18 | ✓ |
| 3k | 0.042 | 33% | 1.49 | 1.28 | ✓ |
| 8k | 0.029 | 31% | 1.41 | 1.40 | tie |
| 28k | 0.028 | 3% | 1.19 | 1.72 | ✗ |
| 90k | 0.030 | none | 1.03 | 1.52 | ✗ |

Dot world, chaser (r 2.05), 12k steps, long training:

| Weights | Chaser's own error | Drop from previous size | Chaser P | Robot P | Holds? |
|---|---|---|---|---|---|
| 3k | 0.171 | — | 1.97 | 0.52 | ✓ |
| 7k | 0.129 | 25% | 1.84 | 0.56 | ✓ |
| 16k | 0.081 | 37% | 1.95 | 0.84 | ✓ |
| 42k | 0.055 | 32% | 2.00 | 1.05 | ✓ |
| 122k | 0.037 | 33% | 1.65 | 1.20 | ✓ |

- **Driving:** the rule holds up to 3k weights, ties at 8k, and fails from 28k. The flip comes exactly where the pedestrian's own error stops falling: it drops about 30% with each roughly 3× more weights up to 8k, then 3%, then none.
- **Dot world:** the chaser's own error keeps falling about 30% per 2× more weights at every size up to 122k. The network never stops being short of capacity, and the rule holds at every size. P first starts to fall at the largest size (2.00 → 1.65).
- **Threshold:** the slower thing loses while giving its own network about 3× more weights still cuts its error by roughly 30%. Once extra weights stop helping, the penalty disappears, and the faster thing pays more instead.
- **This explains the dot world vs driving puzzle.** The dot world holds even at a small speed gap (r 2) because its networks are short of capacity at every size we can train. Driving is fully learned from about 8k weights.
- Predictions: C1 right (0.4k gap 3.41; fails from 28k). C2 wrong: dot P stayed 1.65 at width 196, because the chaser's own error never flattened (the premise of C2 failed, not the rule). C3 right in all 11 rows.
- Wall times: driving pairs 28–83 s, dot runs 28–129 s, at load average 5–6. All are over the 10 s target.

## Phase 1 plan: understand the toy, one question at a time

| # | Question | Test | Prediction | Status |
|---|---|---|---|---|
| 1 | Is the measure sound? | A fixed "trained to the end" rule (held-out miss changes <1% over 2,000 steps); check that 1,000 clips is enough | Current numbers hold within ±0.02 | not started |
| 2 | Is the speed difference the cause? | Change only the stranger's speed: slower, same as the robot, faster | At equal speed the gap disappears; slower stranger, bigger gap | done (finding 10): plain penalty ≈ movement ratio (4.7× → 4.0×, 1.2× → 1.3×) |
| 3 | Does a simple loss fix work? | 1 network, each number's error divided by how much it moves | Closes most of the gap without extra width | done: closes about half (Chaser 0.17 → 0.13, 2 networks 0.09) |
| 4 | How much is shared width? | Width sweep for both setups | The gap shrinks slowly as networks grow | done (finding 9): plain-loss ratio stays 2× from 3k to 122k weights; scaled ratio falls 1.4× → 1.25× |
| 5 | Theory | A tiny linear model with two targets of different sizes and a shared middle layer | Predicts the equal-raw-error behaviour | not started |
| 6 | Why does the Switcher not benefit? | Give the predictor the Switcher's hidden mode | With the mode visible, 2 networks win as with the Chaser | not started |
| 7 | End-to-end training | Train the encoder and predictor together, long enough | Same direction, smaller gap | not started |
| 8 | Several steps ahead | Predict 5 to 10 steps ahead | The 1-network stranger error grows faster | 5 steps done (finding 10): same rule; the stranger net needs the robot's moves |
| 9 | Planning | Both models in the repo's planner; count crashes and goals | 2 networks crash less with the Chaser; tie with the Statue | not started |
| 10 | More agents | Three dots, one net per agent vs one shared | The agent that moves least suffers most in the shared net | not started |

## Phase 2 and 3

- **Phase 2:** write the draft from the toy alone.
- **Phase 3:** scale up with harder robots, more agents, a known benchmark and bigger models. Check only the claims that survived Phase 1.

## Literature

- **No direct test found.** We found no same-size test of a separate predictor per agent against one shared predictor.
- **Orthogonal JEPA** ([arXiv 2608.20065](https://arxiv.org/pdf/2608.20065)) argues that a shared target lets high-variance parts dominate, but does not measure it.
- **Loss-scale imbalance in multi-task learning is known** (GradNorm, [arXiv 1711.02257](https://arxiv.org/abs/1711.02257)). Our new part has to be world models of other agents, the case map, and the planning cost.

## Code and data

- **experiments/train_one.py** trains one full model in about 10 s and saves `tag.pt`. Example: `python3 experiments/train_one.py . enc_chase chase factored 600 0.001 per_dot=1`
- **experiments/two_vs_one.py** trains fresh 2-network and 1-network predictors on that frozen encoder.
  - Example: `python3 experiments/two_vs_one.py . enc_chase chase factored 0.001 6000 1000 49,49,77`
  - The env var `SW` / `RW` weights the stranger / robot error, and `NET=split|single` runs only one of the two.
- **experiments/pull.py** measures the robot's and stranger's gradient pull on a shared network.
- **experiments/two_vs_one.jsonl** holds every 2-vs-1 result so far.
- **experiments/drive.py** is the car + pedestrian toy, with MLP or transformer and split or single. Example: `NORM=agent PED=0.006 python3 experiments/drive.py split mlp m`. Results are in **experiments/drive.jsonl**. The first 3 plain rows with `"norm": false` are from before the shared-constant fix.

## Log

- **2026-11-12:**
  - Found the 600-step undertraining.
  - Built the frozen-encoder 2-vs-1 test and found the main result.
  - Found the raw-error mechanism for the Chaser and the weak Switcher result.
  - Added per-number error scaling (finding 8): it halves the Chaser gap, and the rest is shared capacity.
  - Size sweep from 3k to 122k weights (finding 9): the plain-loss gap stays at 2×, and the scaled gap fades to 0.01.
  - Ablations (finding 10): longer training, 250 clips, stranger speed, 5 steps ahead. The plain penalty ≈ movement ratio, and scaled + big held everywhere. Wall times were 2–3× the usual because another program's 5 Python workers loaded the CPU; run sizes were unchanged.
  - Other strangers (finding 11): scaled + big ties for all; the full R× plain penalty appears only for the easy-to-predict Chaser.
  - Knobs and formulas (finding 12): robot speed, random push, Flee and Orbit. "r× on the learnable part" fits best; r² is wrong. Code: `RSPEED=`, `NOISE=`, `MODE=flee|orbit` in experiments/two_vs_one.py; experiments/score.py scores the formulas.
  - Unpredictable part (finding 13): the oracle shows Instant's leftover is a plateau, not hidden information; Orbit's is partly the hidden angle. The SGD test showed Adam is not why the penalty is r. Code: `ORACLE=1`, `OPT=sgd`.
  - Paper plan written (three tiers, claims, exceptions).
  - Driving toy (finding 14), with car + pedestrian and MLP + transformer. In both architectures the slower pedestrian pays in a shared network, and the penalty grows with r. It is about r^0.3, not r×, and scaling + size does not give a tie: splitting wins here.
  - Capacity size sweep (finding 18). In driving, the pedestrian loses at 0.4k–3k weights, ties at 8k and the car pays more from 28k, exactly where the pedestrian's own error stops falling. In the dot world, the chaser's own error keeps falling up to 122k and the chaser loses at every size (P 1.65–2.0). The threshold is that the slower thing loses while 3× more weights still cut its own error by about a third. Code: sizes `xs`, `sm`, `ml` in experiments/drive.py.
  - Long-training reruns and factor test (finding 17). The rule holds in 15 of 25 well-trained settings: all dot-world ones and every case with r ≥ 7.4, but not default driving. The old driving pedestrian-only penalty was mostly end-of-training jitter: weight averaging or a bigger batch in the same 6k steps removes most of it. Code: `BATCH=`, `AVG=` in experiments/drive.py.
  - Capacity-split theory and long training (finding 16). Driving and arm networks at 6,000 constant-lr steps were undertrained. With 24,000 steps and cosine decay, the arm and driving penalties shrink to 1.06–1.9×, while the dot world stays 2×. The shared dot-world network acts like a split that gives the chaser 10% of the weights, against 20% for the loss-optimal split. Code: `DECAY=1`, `LR=`, a numeric size (exact width) and the `mse` / `cv_base` fields in experiments/drive.py; `DECAY=1` in experiments/two_vs_one.py; experiments/capsplit.py.
  - Knob grid (finding 15), with weaving pedestrians, an arm robot, 2–3 pedestrians, a random push, L1 and a GRU. Robot parts behave like separate things. L1 does not fix it. Both explanations of r^0.3 failed. The GRU broke the rule at small sizes, where the separate networks were too small to learn, so the rule now carries "once each separate network is big enough to learn its thing". Code: `WORLD=arm`, `PEDS=`, `CURVE=`, `PNOISE=`, `LOSS=l1`, `ARCH=gru` in experiments/drive.py.

## Finding 19: speed gap vs capacity (predictions written before the runs, 2026-11-12)

**Question.** Is there one rule linking the speed gap r and the network size at which the slower thing stops losing?

**Knob.** `CARSCALE=a` in experiments/drive.py measures the car in a× bigger units. The motion, the pedestrian and the interaction are identical; only r changes (r = 5.2a). Slowing the pedestrian instead (PED=0.003) made the pedestrian hard to learn even alone (own error 0.95 of the constant-velocity guess at 90k weights, 6,000 steps), which mixes r with task difficulty.

**Protocol.** Driving MLP, 6,000 steps, cosine decay, batch 256. a = 0.5, 1, 2, 4, 8 (r 2.6, 5.2, 10.4, 21, 42). Sizes xs, s, sm, m, ml, l (0.4k to 90k weights). P = pedestrian error shared ÷ own. Flip size N*(r) = the smallest size where the pedestrian's P ≤ the car's P.

**Predictions.**
1. The split pedestrian's error does not depend on a (its network never sees the car's error; Adam ignores the loss scale). Checked at one size.
2. At every size, the pedestrian's P grows with r.
3. N* grows with r. Guess: N* roughly doubles when r doubles (N* ∝ r); r² growth would follow loss share, earlier findings say penalties follow r or less.
4. At r 42, the pedestrian loses at every size up to 90k.
5. What would break it: P not falling with size at a fixed r, or N* not growing with r.

**Results (31 runs, 2–9 s each).** Pedestrian's own error (split): xs 0.453, s 0.278, sm 0.266, m 0.179, ml 0.186, l 0.211 (stops falling at about 8k). P = shared ÷ own:

| Size | Weights | r 2.6 | r 5.2 | r 10.4 | r 21 | r 42 |
|---|---|---|---|---|---|---|
| xs | 360 | 1.36 | 1.92 | 9.66 | 19.0 | 27.9 |
| s | 1.1k | 0.95 | 1.33 | 2.58 | 6.75 | 18.5 |
| sm | 3k | 0.97 | 1.22 | 2.03 | 5.25 | 8.59 |
| m | 8k | 1.18 | 1.30 | 1.81 | 3.39 | 6.49 |
| ml | 28k | 1.10 | 1.13 | 1.28 | 1.83 | 3.07 |
| l | 91k | 1.20 | 1.23 | 1.35 | 1.74 | 3.29 |

**Fit.** Extra pedestrian error (shared − own), sizes xs to ml: extra ∝ r^1.44 × N^−0.73 (R² 0.95). The exponent ratio is 1.97, so the extra error depends on r²/N. Points with equal r²/N match: m at r 10.4 (0.144) and ml at r 21 (0.155); m at r 21 (0.43) and ml at r 42 (0.38); sm at r 21 (1.13) and m at r 42 (0.98). With l included the ratio is 2.4 (R² 0.89); l is limited by 6,000 steps (its own error rose to 0.211).

**Rule.** The slower thing's extra error depends on r²/N: when the speed gap doubles, the network needs 4× the weights to keep the same penalty. That is the car's share of the squared-error loss (∝ r²). Here the penalty sinks to noise (extra ≤ 0.03) below r²/N ≈ 0.001, so about 1,000 × r² weights (27k at r 5.2, 108k at r 10.4).

**Against the predictions.** 1 held (own error 0.179 at a = 1 and 4). 2 held at every size beyond noise. 3 was wrong: N* grows with r², not r. 4 held (P ≥ 3 at r 42 up to 91k). At a fixed size the penalty grows like r^1.4, which fits finding 12 ("r² is wrong" at fixed size); r² is the trade between gap and size.

**Weak spots.** One world, 6,000 decayed steps, one run per point; P noise is about ±0.15 (s and sm read 0.95–0.97 at r 2.6). The constant 1,000 is specific to this world; only the r² trade is a candidate general rule. Next: the same grid in the dot world and the arm world, where the slower thing's own network needs a different size.

## Finding 20: does "extra error depends on r^k / N" hold everywhere? (predictions before the runs, 2026-11-12)

Same protocol as finding 19 (6,000 decayed steps, sizes xs–ml fitted, gaps ×0.5–×8 by `CARSCALE`, `TAG=` per scenario, fit with experiments/gapfit.py). k is the exponent of r that trades against N.

| Scenario | Knob | Prediction for k | Why |
|---|---|---|---|
| L1 loss | `LOSS=l1` | k ≈ 1 | absolute error makes the car's share of the loss grow with r, not r² |
| GRU | `ARCH=gru` | k ≈ 2 | the loss share does not depend on the network type |
| Transformer | `ARCH=tf` | k ≈ 2 | same |
| Arm world (fast tip scaled, slow base) | `WORLD=arm SCALETHING=1` | k ≈ 2 | different motion, same loss |
| Weaving pedestrian | `CURVE=1` | k ≈ 2 | harder slow motion moves the constant, not k |
| Two pedestrians | `PEDS=0.006,0.004` | k ≈ 2 for each | each pedestrian's share falls with r² |
| Random push | `PNOISE=0.002` | k ≈ 2, smaller constant | the push is unlearnable for both designs |

What would break the rule: k far from 2 (or from 1 under L1), or a poor fit (R² < 0.8).

**Results (2026-11-12, 14 depth-2 MLP scenarios plus GRU, depth 1–4, push, per-thing scaling; about 400 runs of 2–9 s).**

- **k is not universal.** The r-for-N trade of the absolute extra error: arm 1.0, L1 2.2, weave 2.7, GRU about 1.2 (poor fit). Prediction "k ≈ 2 everywhere" failed; the L1 prediction k ≈ 1 also failed (2.2).
- **The gap exponent of P − 1 at a fixed size is stable:** about 1.5 (1.2–1.7) in every depth-2 MLP scenario.
- **Per-thing loss scaling (`NORM=agent`) removes the gap effect:** P is flat across r at every size (gap exponent 0.00). The gap acts only through the loss scale.
- **One variable collapses the scenarios: the loss ratio Λ** = the faster thing's loss term ÷ the slower thing's, each computed from its own (split) network. Pooled over 308 points: **P − 1 = 0.149 · Λ^0.77** (R² 0.80, P = 1.5 at Λ ≈ 5), every scenario within ×1.8. With size: **P − 1 = 5.44 · Λ^0.71 · N^−0.42** (R² 0.90, worst scenario ×1.65). The leak ratio (extra ÷ car error) was not constant and was dropped.
- **Constants that move:** depth raises the constant (×1.5 at depth 3, ×2.3 at depth 4), the GRU lowers it (×0.25–0.30). Push: the unlearnable part dominates and too few points have a penalty. Transformer: 18–31 s per run, dropped.

## Finding 21: the loss-ratio law on new settings (predictions before the runs, 2026-11-12)

Law fitted on finding 20's 14 depth-2 MLP scenarios: **P − 1 = 0.149 · Λ^0.77**, Λ = the faster thing's loss term ÷ the slower thing's, each from its own (split) network. Scored here with no refit.

| Setting | Knob | Prediction |
|---|---|---|
| Car is the slower thing | `SCALETHING=1`, a = 16–128 (pedestrian 3–24× faster than the car), slow = car | measured/predicted within ×0.5–×2 |
| Constant learning rate | `DECAY=0` | within ×0.5–×2; P a bit higher (no annealing) |
| GRU at sizes it can learn | `gru` m, ml, l | constant ×0.25–×0.5 as in finding 20, slope ≈ 0.75 |
| Muon on the hidden matrix (rest Adam) | `OPT=muon MUONLR=0.1` (best of 0.005–0.2 at size m) | P smaller than Adam, since Muon enlarges small update directions. Written before the tuning runs; the first tuning point already disagrees (P 2.0 vs Adam 1.30 at r 5.3) |
| Dot world (frozen encoder, latent targets) | `LATSCALE=a` scales the robot's latent targets (inputs unchanged), split widths 8–64 | P rises with a at every size, slope on Λ 0.6–0.8; constant about ×2 above driving (21 older dot-world pairs: ×2.2, slope 0.61) |
| Dot world with 250 clips instead of 1000 | `ntraj=250`, widths 8–128, a = 1 and 4 | the stranger's own error stops falling with size sooner, so P now falls with size (rule: size cures the penalty only once the slower thing no longer needs the weights itself) |
| Dot world, Switch stranger (own error flat with size: 0.229 at width 49, 0.235 at 196) | `enc_switch`, widths 12–196, a = 1 and 4 | P falls with size (unlike Chase, whose own error keeps falling) |

**Results (2026-11-12).**

| Setting | Prediction | Result |
|---|---|---|
| Car is the slower thing (pedestrian 3–24× faster) | law within ×0.5–×2 | ×0.47, Λ 38–5,900 (beyond the fitted range); with the size term the pooled law still holds (R² 0.89 with this grid included) |
| Constant learning rate | within ×2, P a bit higher | ×1.50, held |
| GRU at sizes m–l | constant ×0.25–×0.5 | ×0.06–×0.6, slope on Λ 0.5 instead of 0.75 |
| Muon (rate 0.1, best of 0.005–0.2) | P smaller than Adam | **wrong**: P higher (law offset ×1.69); Muon cuts the pedestrian's own error 5× (0.038 vs 0.179) but the gap penalty stays |
| Dot world, Chase stranger, widths 8–64 | P rises with gap; constant ×2 above driving | P rises with gap (P − 1 ∝ r^0.96), law offset ×0.83; **P does not fall with size** (N^+0.07) |
| Dot world, 250 clips | P falls with size | precondition failed: the stranger's own error still fell with size (0.23 → 0.09); P flat at a = 1, mild fall at a = 4 |
| Dot world, Switch stranger (own error flat with size) | P falls with size | held: at gap ×4, P 3.63 → 2.17 → 1.27 over widths 12 → 196 (P − 1 ∝ N^−0.62), while Chase in the same world stays flat |

**Training length does not change P** (driving, sizes m and ml, cosine decay, P at 6k / 12k / 24k steps): r 5.3: 1.30 / 1.33 / 1.41; r 10.5: 1.81 / 1.81 / 1.72; r 21: 3.39 / 3.04 / 3.08 (m) and 1.83 / 2.02 / 2.03 (ml). Both errors fall about 6× over that range; the shared network learns the slower thing at a fixed fraction of the speed (about half at r 21, size m). The step curves also showed that the pedestrian alone reaches 0.06–0.10 at 18k steps even at 360 weights, so the earlier "own error stops falling at 8k weights" (finding 19) was a 6,000-step limit, not a capacity limit. Yet the bigger shared network still slows it less (m 3.08 vs ml 2.03 at 24k, same own error 0.029 vs 0.028).

**Direction checks over every grid** (experiments/monotone.py, a break must exceed 0.15 in P): P rises with the gap in 1,057 of 1,066 pairs (the 9 breaks are depth 1 and per-thing scaling, which is flat by design). P falls with size in 520 of 558 pairs; the breaks are mostly the smallest gap, where P ≈ 1, and per-thing scaling.

**Rules table** (experiments/rules.py; 6,000 decayed steps unless noted; law offset = measured ÷ 0.149 Λ^0.77, median over the setting):

| Setting | gap exponent (P − 1 ∝ r^x) | size exponent (P − 1 ∝ N^y) | own-error size slope of the slower thing | law offset |
|---|---|---|---|---|
| driving MLP, Adam | 1.45 | −0.52 | 0.21 | ×0.95 |
| driving, fine gap grid | 1.18 | −0.82 | −0.03 | ×0.55 |
| driving, 18k steps | 1.57 | −1.07 | 0.22 | ×1.23 |
| driving, constant LR | 1.06 | −0.40 | 0.00 | ×1.42 |
| driving, LR 3e-3 | 1.58 | −0.80 | 0.04 | ×0.88 |
| driving, LR 3e-4 | 1.10 | −0.48 | 0.26 | ×0.76 |
| driving, batch 64 | 1.53 | −0.50 | 0.25 | ×0.82 |
| driving, 250 clips | 1.36 | −0.61 | 0.08 | ×1.18 |
| driving, faster pedestrian | 1.31 | −0.62 | 0.12 | ×1.82 |
| driving, weaving pedestrian | 1.60 | −0.55 | 0.17 | ×1.65 |
| two pedestrians (1st / 2nd) | 1.37 / 1.21 | −0.53 / −0.58 | 0.16 / 0.15 | ×1.07 / ×0.67 |
| driving, car is the slower thing | 1.18 | −0.64 | 0.36 | ×0.35 |
| arm robot / faster arm | 1.38 / 1.24 | −1.50 / −1.48 | −0.02 / −0.03 | ×1.34 / ×1.57 |
| driving, L1 loss | 1.35 | −0.50 | 0.39 | ×0.93 |
| driving, Muon | 1.20 | −0.54 | 0.47 | ×1.69 |
| MLP depth 3 / depth 4 | 1.76 / 1.40 | −0.31 / −0.27 | 0.41 / 0.42 | ×1.82 / ×2.44 |
| GRU | 0.94 | −0.39 | 0.29 | ×0.12 |
| per-thing loss scaling | flat (P − 1 > 0.1 in 1 of 25 runs) | – | 0.21 | – |
| dot world, Chase | 0.96 | +0.07 | 0.36 | ×0.83 |
| dot world, Chase, 250 clips | 0.96 | −0.04 | 0.29 | ×0.85 |
| dot world, Switch | 1.22 | −0.46 | 0.05 | ×0.91 |

**Rules.**
1. **Gap (holds in all 23 settings without per-thing scaling):** P − 1 grows like r^1.3 (middle half 1.18–1.42, range 0.94–1.76), at every size, in two worlds, three network types and two optimizers.
2. **Loss ratio (22 of 23 within ×2.5):** P − 1 ≈ 0.15 · Λ^0.75, Λ = the faster thing's loss ÷ the slower thing's loss, each from its own network. Λ ≈ r² × (how much harder the fast thing is to predict, relative to its move). Exception: the GRU (×0.12).
3. **Size helps only when the slower thing has stopped needing weights:** where its own error no longer falls with size, P − 1 ∝ N^−0.4 to N^−1.5 (mean −0.87); where it still falls, N^+0.07 to N^−0.64 (mean −0.37). Correlation 0.60 over 23 settings, size exponent ≈ −0.87 + 1.40 × own-error slope. Direct test: Chase vs Switch in the dot world.
4. **Training length does not change P** while the slower thing is still learning (6k–24k steps, ±0.15).
5. **Per-thing loss scaling removes the gap effect.**

**Weak spots.** One run per point (runs are deterministic; P differs by about ±0.15 between neighbouring settings). The size rule is a correlation plus one direct test. The GRU constant is 8× lower and its slope 0.5; the transformer was not tested (too slow for 10 s runs). SGD diverged above rate 0.001 and learned almost nothing below it in driving. Code: experiments/allfit.py, experiments/formfit.py, experiments/effsteps.py, experiments/steptrend.py, experiments/dotgrid.py, experiments/rules.py, experiments/monotone.py; `OPT=muon`, `MUONLR=` in experiments/drive.py; `LATSCALE=`, `LATTHING=` in experiments/two_vs_one.py.

**Follow-up predictions (written before the runs, 2026-11-12).**

| Test | Prediction |
|---|---|
| Loss weight instead of units (dot world `RW`, driving `LW`) | weighting the fast thing's loss by a² gives the same P as measuring it in a× bigger units |
| Dot world, Instant stranger (own error falls 0.223 → 0.070 from width 49 to 196) | P flat with size (size exponent above −0.3) |
| Training length in the dot world (Chase, width 32, gap ×4) and arm world (size m, gap ×4) | P unchanged from 6k to 24k steps (±0.2) |

**Follow-up results (2026-11-12).**

| Test | Result |
|---|---|
| Loss weight instead of units | held. Dot world: robot weight 0.25 / 4 / 16 gives P 1.30 / 2.29 / 3.66, against units ×0.5 / 2 / 4 giving 1.31 / 2.43 / 3.78. Driving, size m: car weight 0.25 / 4 / 16 / 64 gives 1.06 / 2.22 / 4.25 / 7.03, against units ×0.5 / 2 / 4 / 8 giving 1.18 / 1.81 / 3.39 / 6.49. The effect is the loss weight; the units only set it. |
| Dot world, Instant stranger (own error still falling with size) | held: weak size effect, N^−0.14 to N^−0.24 |
| Training length, dot world (Chase, width 32, gap ×4) | held: P 3.78 / 3.82 / 3.70 at 6k / 12k / 24k |
| Training length, arm world (size m, gap ×4) | broken: P 2.75 / 2.20 / 1.67, because the base's own error stops falling (0.016 / 0.010 / 0.009). Driving size s at gap ×4 also falls at 24k (3.39 / 6.75 / 8.01 / 4.52 at 3k / 6k / 12k / 24k) once its own error nears its floor. |

**Fixed-slowdown model (experiments/slowdown.py, experiments/fslow.py).** The shared network learns the slower thing F× slower: its error after S steps equals the own network's error after S/F steps, with F constant over training. F measured at 6k predicts the shared error at 12k and 24k within ×0.80–1.15 (driving m), ×0.83–1.40 (driving s) and ×1.11–1.34 (arm). This explains rule 4: if the own error falls like S^−α, then P = F^α stays constant while the slower thing is learning. P → 1 once its own error reaches a floor that more steps cannot remove. Over 136 driving and arm runs (8 settings), F − 1 = 11 (Λ/N)^0.47 (R² 0.79), with offsets ×0.8–1.3, except the GRU at ×0.31. The direct P law on the same runs fits better: P − 1 = 20.7 (Λ/N)^0.52 (R² 0.86). In the teacher world (finding 22), F does not follow Λ (R² 0.05), so P, not F, is the variable to model. The step slope α does not replace size either (adding it leaves R² at 0.895).

## Finding 22: a third world with no physics, teacher and student (predictions before the grid, 2026-11-12)

`experiments/teacher.py`: input x ~ N(0, I₁₆), fresh samples every step. Fast target = a × (random 2-layer tanh teacher, 32 hidden). Slow target = a second random teacher with HS hidden units (std 1). The student is a depth-2 MLP, shared (8 outputs) or split (two students, equal total weights). 3,000 cosine-decayed steps, about 2 s per run. Gap r = a.

Precondition, split student, own error of the slow target at widths 8 → 128: easy teacher (HS = 2) 0.047 → 0.0024 (log size slope 0.68, still falling fast); hard teacher (HS = 256) 0.29 → 0.135 (slope 0.17). With noise 0.1 added to the easy target, the own error flattens at about 0.01 (slope near 0).

| Test | Prediction |
|---|---|
| Gap, all three teachers, a = 1, 2, 4, 8 | P rises with a at every width; P − 1 ∝ a^1.0–1.6 |
| Loss-ratio law, no refit | P − 1 within ×2.5 of 0.149 Λ^0.77 |
| Size, easy teacher (slope 0.68) | rule 3 formula −0.87 + 1.40 × 0.68 ≈ +0.1: P flat with size |
| Size, hard teacher (slope 0.17) | ≈ −0.6: P falls with size, although the hard target needs more weights |
| Size, easy teacher + noise floor (slope ≈ 0) | ≈ −0.9: P falls with size, fastest of the three |

**Grid result before the next test.** Size did not lower P in any of the three teachers (P − 1 ∝ N^+0.32, N^−0.08, N^+0.13). Once Λ is known, size adds nothing (pooled N^0.01). That contradicts driving, where size adds N^−0.47 beyond Λ.

**Prediction (written before the runs):** the difference is separate inputs. In driving each thing's future depends mostly on its own coordinates, while both teacher targets read all 16 inputs. With `SEP=1` (fast teacher reads x[:8], slow teacher x[8:]), P falls with size beyond Λ (size exponent ≤ −0.3 for both HS = 2 and HS = 256).

**Separate inputs, result:** wrong. P still did not fall with size (N^+0.12 for HS = 2, N^+0.34 for HS = 256). The loss-ratio law held: pooled over all 95 teacher points, P − 1 = 0.147 Λ^0.74, against driving's 0.149 Λ^0.77.

**Prediction (written before the runs):** the size effect comes from finite data. Driving trains on 1,000 fixed clips, and its size effect was strongest at 18k steps (N^−1.07). With `NTRAIN=2000` (fixed training set), P falls with size beyond Λ (exponent ≤ −0.3) for HS = 2 and HS = 256.

**Finite data, result:** wrong. With 2,000 fixed samples P still did not fall with size (N^+0.18 for HS = 2, N^−0.05 for HS = 256). The hard teacher with fixed data falls far below the law (×0.19); its own error barely moves with size (0.30 → 0.26).

**Prediction (written before the runs):** the misses are teachers whose own error improves slowly with training. Under the fixed-slowdown model P = F^α (α = step slope of the slower thing's own error), so a slow learner shows a small P even when its slowdown F is large. With own curves at 750 / 1,500 / 3,000 / 6,000 steps, F − 1 against Λ puts the hard teachers on the same line as the easy ones (offsets within ×2), while P misses them by ×3–5.

**Slowdown test, result:** wrong. F − 1 against Λ has R² 0.05 in the teacher world: the hard teacher's own curve is so flat that small P values map to huge, noisy F. P = F^α itself holds (median measured/predicted 1.00), but P, not F, is the variable that follows one law across worlds. Teacher-world P law over 126 points: P − 1 = 0.117 Λ^0.76; offsets ×0.9–2.1 except the hard teacher with fresh data (×0.40) and with fixed data (×0.21), whose own error is stuck (step slope α 0.0–0.3).

**Prediction (written before the runs):** driving's unit scale also multiplies the scaled thing's inputs; the dot and teacher worlds scale targets only. If input scaling causes driving's size effect, then loss weighting (`LW` = a², inputs unchanged) at sizes xs–ml gives P flat with size. Otherwise P falls with size as under `CARSCALE`.

**Loss weight across sizes, result:** P falls with size under loss weighting just as under unit scaling (gap ×4: LW 16 gives P 12.2 / 5.9 / 5.5 / 4.3 / 2.5 for xs–ml; CARSCALE 4 gives 19.0 / 6.8 / 5.3 / 3.4 / 1.8). Input scaling is not the cause. Prediction wrong.

**What differs:** in driving at gap ×4, Λ is flat from N 3,048 to 27,908 (both own errors stop improving with size: pedestrian 0.27 → 0.19, car 0.042 → 0.032), yet P falls 8.6 → 3.1. In the teacher world both own errors keep falling with size, and P follows Λ.

**Prediction (written before the runs):** size lowers P beyond Λ only where the own networks no longer gain from size. Teacher world at 300 steps (own errors limited by steps, not size), widths 16–256, HS = 32: own errors nearly flat with size, and P falls with size (exponent ≤ −0.3).

**300 steps, result:** held. Own error 0.40 / 0.30 / 0.28 / 0.26 / 0.23 for widths 16–256; P at gap ×8 falls 1.94 / 2.40 / 1.49 / 1.18 / 1.21, P − 1 ∝ N^−0.45. Control prediction (written before the run): the same teacher (HS = 32) at 3,000 steps, where the own error keeps falling with size, gives P flat or rising with size.

**3,000-step control, result:** held. Same teacher: own error 0.24 → 0.059 over widths 16–256, and P at gap ×8 rises 1.78 → 4.10 (N^+0.30). With only training length changed, the size effect flips sign: P falls with size when the own networks are step-limited (300 steps) and rises when they are size-limited (3,000 steps).

**Dot-world version (prediction written before the runs).** At 300 steps the dot world's own error still falls with size (0.40 / 0.31 / 0.22 at widths 8 / 32 / 128), so short training does not reach the flat regime there. Instead, a random push on the stranger (`NOISE=0.02`, Chase) gives its own error a floor: 0.63 at width 8, 0.52 at width 128. Prediction: at gap ×4, P falls with size (exponent ≤ −0.3), unlike noise-free Chase (N^+0.07).

**Dot world with a push on the stranger, result:** held at the edge. At gap ×4, P is 1.53 / 1.55 / 1.51 / 1.35 / 1.19 for widths 8–128 (N^−0.29), against N^+0.07 without the push. At gap ×1 it is 1.20 / 1.19 / 1.14 / 1.11 / 1.00.

**Size exponent at fixed Λ, every setting (experiments/regime.py).** Fit per setting: log(P − 1) = a + b log Λ + c log N. σ = how fast the slower thing's own error falls with size.

| Group | Settings | c (size exponent at fixed Λ) | b (Λ exponent) |
|---|---|---|---|
| driving and arm, all MLP settings | 16 | −0.22 to −0.73 (mean −0.48), at every σ from −0.03 to 0.47 | 0.57–0.86 |
| teacher, own error still falling with size (3,000 steps) | 7 | −0.08 to +0.38 (mean +0.12) | 0.46–0.94 |
| teacher, step-limited (300 steps) | 1 | −0.45 | 0.53 |
| teacher, slower target stuck (HS 256, fixed data) | 1 | −0.03 | 0.46 |

The c-against-σ correlation over all 25 settings is only 0.36, so σ alone does not set the size exponent across worlds. The direct within-world switches all point the same way (teacher 300 vs 3,000 steps; dot Switch vs Chase; dot with vs without the push): size lowers P at fixed Λ only where the own networks no longer gain much from size. Driving and arm sit in that regime at every setting tested.

**Prediction (written before the runs):** the depth and optimizer offsets found in driving carry over to the teacher world. Depth 3 and 4 raise the law offset (driving ×1.8 and ×2.4 against depth 2's ×0.95); Muon raises P (driving ×1.69). Easy teacher (HS = 2) and HS = 32, widths 16–128, gaps 1–8, 3,000 steps.

**Depth in the teacher world, result (easy teacher, widths 16–128; the Muon and HS = 32 half was stopped when Vuk returned).** Law offset ×0.92 / ×1.06 / ×1.38 at depth 2 / 3 / 4. Deeper networks raise the penalty in the same direction as driving (×0.95 / ×1.8 / ×2.4), but by less. All 46 points: P − 1 = 0.117 Λ^0.82 (R² 0.94).

**Rules after findings 21–22 (driving, arm, dot world, teacher world).**
1. P rises with the gap in every world: P − 1 ∝ r^0.9–1.9.
2. **P − 1 ≈ 0.15 · Λ^0.75** (Λ = the faster thing's loss ÷ the slower thing's loss, each from its own network). Fitted constants: driving 0.149 Λ^0.77; teacher 0.147 Λ^0.74 (fresh data and separate inputs, 95 points) and 0.117 Λ^0.82 (depth grid); dot world ×0.83–0.97 of the driving law. Within ×2.5 in 31 of 35 settings. All misses fall below the law: GRU ×0.12, hard teacher ×0.19–0.40, teacher at 300 steps ×0.16.
3. Size lowers P beyond Λ only where the own networks no longer gain much from size (step-limited or at a floor): driving and arm N^−0.2 to −0.7; teacher at 300 steps N^−0.45; dot Switch and dot with a push N^−0.3 to −0.6. Where own errors still fall with size, size acts only through Λ (teacher N^−0.1 to +0.4; dot Chase N^+0.07). Shown by three direct switches. There is no single cross-world threshold (correlation 0.36).
4. Training length leaves P unchanged while the slower thing is still learning, because the shared network learns it at a fixed fraction of the speed (P = F^α).
5. Per-thing loss scaling removes the effect, and the loss weight is the cause (weight a² = units ×a).

Rejected explanations for the size effect: separate inputs, finite data, input scaling, the step slope α, and a slowdown law in F.

## Finding 23 (planned): multi-token prediction in a small language model. Predictions written before any run, 2026-11-13

**Question:** a language model trained with multi-token prediction has extra heads that guess the token 2, 3 or 4 places ahead (DeepSeek-V3, Gloeckle et al. 2024). Future tokens are harder to guess, so those heads have a bigger loss. Does the loss-ratio law predict how much worse the model's next-token prediction gets, from one number measured on cheap runs?

**What is predicted, exactly.** P₁ = the next-token head's excess loss in the model trained with future heads ÷ its excess loss in the same model trained on next-token prediction only. Excess loss = cross-entropy − the true entropy, the part a model can still learn; the data come from a Markov chain, so the true entropy of every future token is known exactly. The input is Λ₁ = (sum of the future heads' loss terms) ÷ (the next-token term) = W × mean(L₂..L_K) ÷ L₁, each L from a model trained on that head alone at the same width.

**World and model:** order-2 Markov chain over 64 tokens with random Dirichlet rows (concentration 0.5, token frequencies 1/rank), fresh sequences every step. Transformer with 2 layers, widths 32 / 64 / 128, RoPE, QK-norm, SwiGLU, RMSNorm, Adam with cosine decay to zero, 1,500 steps. Loss = CE₁ + W · mean(CE₂..CE_K). Code: `experiments/mtp.py`, runs: `experiments/mtp_runs.sh`, scoring: `experiments/mtpfit.py` (fixed law, no refit).

| # | Prediction | Test |
|---|---|---|
| 1 | P₁ − 1 ≈ 0.15 Λ₁^0.75, within ×2.5 | stage 1: K = 4, W = 0.1 / 0.3 / 1 / 3, widths 32 / 64 / 128 |
| 2 | When W is small enough that Λ₁ < 1, next-token prediction is not hurt (P₁ < 1.1); the future heads are, by the same law with their own Λ | stage 1, W = 0.1 and 0.3 |
| 3 | Λ from excess loss fits; Λ from raw cross-entropy (entropy included) predicts too much harm, because the entropy part is a floor no network removes (finding 12: the penalty hits only the learnable part) | stage 1, both Λs scored |
| 4 | **Cheap prediction:** Λ₁ from own models trained for 1/8 of the steps is within ×1.5 of Λ₁ at full length, and P₁ at 1/8 of the steps is within ×1.3 of P₁ at full length (rule 4: P stays fixed while the thing is still learning) | stage 2 |
| 5 | P₁ falls with width only where the next-token model's own excess loss stops falling with width (rule 3) | stages 1 and 4 |
| 6 | P₁ > 1 at every width: future heads never help next-token prediction here, because the chain's next token depends only on the last 2 tokens, so future heads teach nothing the next-token head needs. A gain from future heads, as Gloeckle et al. saw in large models, would need data with longer structure | stages 1, 3 |
| 7 | Heads with their own transformer block (Gloeckle style) and K = 2 stay on the same line; the block shifts the constant by less than ×2 | stage 3 |
| 8 | A more predictable chain (ALPHA 0.1) and a less predictable one (ALPHA 2) stay on the same line once Λ uses excess loss | stage 4 |

**If 1 and 3 hold,** the law carries from squared error to cross-entropy and to a loss sum large language models really use, and prediction 4 makes it a cheap check before training. **If they fail,** the law is limited to squared error, and the repo stays a finished study of that case.
