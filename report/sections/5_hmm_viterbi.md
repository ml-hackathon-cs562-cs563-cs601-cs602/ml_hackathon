# 5. Hidden Markov Model & Viterbi Decoding

## 5.1 Formulation and Hidden States
Independent candidate classification predicts the most probable arc for each GPS observation in isolation, ignoring vehicle dynamics and road network topology. To resolve trajectory consistency, we formulate sequence decoding as a Hidden Markov Model (HMM):
- **Observations ($O_t$):** GPS points at timestamps $t_1, t_2, \dots, t_T$.
- **Hidden States ($S_t$):** The true road network arc $a \in \mathcal{A}$. At each timestamp $t$, the candidate set is restricted to the $K \le 10$ nearest arcs within search radius $R = 80$ m.

## 5.2 Emission and Transition Modeling
### Emission Probabilities
Candidate emission likelihoods are derived from the classifier's predicted probability $p_i(a)$ that arc $a$ matches observation $i$:
$$\log P(O_t \mid S_t = a) = w_e \cdot \log(\max(p_t(a), \epsilon)) + (1 - w_e) \cdot \log(1 - \min(p_t(a), 1 - \epsilon))$$
where $w_e = 0.357$ balances discriminative confidence against transition penalties, and $\epsilon = 10^{-6}$.

### Transition Costs
Transition penalties $C(a_{t-1} \to a_t)$ enforce physical network connectivity:
1. **Continuation ($a_t = a_{t-1}$):** Vehicle remains on the same arc; penalty = `stay` ($0.0$).
2. **Topological Reachability:** Evaluated using Breadth-First Search (BFS) on the directed road graph up to depth $H = 4$:
   $$C(a_{t-1} \to a_t) = \text{hop\_base} + (h - 1) \cdot \text{hop\_step}$$
   with $\text{hop\_base} = 1.089$ and $\text{hop\_step} = 1.962$.
3. **U-Turn Penalty:** If arc $a_t$ is the reverse directed edge of $a_{t-1}$ ($\text{dst}(a_t) = \text{src}(a_{t-1})$ and $\text{src}(a_t) = \text{dst}(a_{t-1})$), an additional penalty $\text{uturn} = 5.066$ is incurred.
4. **Disconnected Arcs ($h > H$):** If no topological path within $H$ hops exists, an `invalid` transition penalty of $17.302$ is applied.
5. **Sensor Gap Handling:** If the time delta $\Delta t = t_i - t_{i-1} > \text{gap\_s}$ ($5.0$ s), the transition penalty resets to zero to prevent penalizing vehicle path choices across sensor dropouts.

## 5.3 Classifier-Only vs Classifier + HMM Comparison
Evaluating across the 10 student-test trips demonstrates the critical impact of topological sequence decoding:

| Configuration | Top-1 Accuracy | Route Continuity | Sequence Arc F1 |
| :--- | :--- | :--- | :--- |
| **Untuned Classifier Only** | 85.57% | 75.39% | 85.13% |
| **Untuned Classifier + Default HMM** | 89.71% | 85.33% | -- |
| **Untuned Classifier + Tuned HMM** | 90.77% | 91.75% | 93.70% |
| **Tuned Classifier Only** | 85.40% | 75.30% | 85.03% |
| **Tuned Classifier + Default HMM** | 89.57% | 85.26% | -- |
| **Tuned Classifier + Tuned HMM** | **91.15%** | **92.15%** | **93.91%** |

![Tuning and HMM Progression](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/report/figures/tuning_and_hmm.png)
*Figure 5.1: Performance progression showing the impact of Bayesian parameter tuning and topological HMM Viterbi decoding.*

The tuned HMM achieves a **+5.75% absolute gain** in top-1 observation accuracy, a **+16.85% absolute gain** in route continuity (from 75.30% to 92.15%), and an overall sequence arc F1 score of **93.91%**.

## 5.4 Analysis of Failure Modes
Cases where the HMM struggled to recover the reference route:
1. **Parallel Service Roads & Flyovers:** When a highway runs parallel to a local access road within 15–30 meters for multiple consecutive observations, high emission confidence on an early noisy point can lock Viterbi into the wrong parallel corridor if topological continuity is valid on both roads.
2. **Dense Multi-Arm Intersections with GPS Jitter:** High speed deceleration near roundabouts creates dense clusters of noisy GPS points. If candidate generation misses the short junction connector arc, the BFS hop distance exceeds $H=4$, triggering the invalid penalty and causing fragmented transitions.
