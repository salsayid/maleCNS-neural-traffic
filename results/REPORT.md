# Lincoln traffic control experiment

The MaleCNS circuit was trained by supervised imitation to choose the signal phases recommended by an explicit downstream-aware queue heuristic. It was evaluated on synthetic traffic at three actual Lincoln signal locations: 9th & Q, 9th & P, and 9th & O.

## Evaluation

Accumulated queue seconds per arrival, including every arriving vehicle. Lower is better. Mean over 20 held-out traffic seeds; learned policies also average three training seeds. All runs drained fully within the 60-minute horizon.

| Model | Standard pulse | Higher event demand | Shuttle assumption | P Street restriction |
|---|---:|---:|---:|---:|
| Fixed schedule | 113.06 | 189.49 | 66.56 | 370.86 |
| Queue heuristic (teacher) | 93.26 | 148.52 | 65.76 | 286.73 |
| MaleCNS circuit | 93.20 | 147.34 | 64.50 | 285.65 |
| Rewired circuit | 92.84 | 146.09 | 64.04 | 284.09 |
| Conventional network | 92.82 | 147.49 | 64.11 | 283.93 |

The circuit reduced simulated queue time relative to the fixed schedule by **17.6%**, with a paired bootstrap 95% interval of **14.7–20.3%**. This interval only describes the modeled traffic distribution. It does not quantify uncertainty about Lincoln traffic.

The queue heuristic, rewired graph, and conventional network perform similarly. The comparison supplies no evidence for a special advantage of fly wiring. The MLP has 5,379 parameters versus 3,819 for each circuit; it is a practical reference, not a parameter-matched ablation. The rewired network is parameter-matched.

## Neural ablations

Mean standard-scenario queue seconds per arrival for the seed-11 circuit, over the same 20 traffic seeds. These are model interventions, not claims about living neurons.

| Intervention | Queue seconds per arrival |
|---|---:|
| None | 92.97 |
| Silence Mi1 | 91.37 |
| Silence Mi4 | 91.22 |
| Silence T4a | 91.02 |
| Silence all_T4 | 130.14 |

## Scope

Signal locations are city GIS data. Arrival rates, queue storage, service capacities, phase plans, and the 25% shuttle-related demand change are assumptions. Current game-day restrictions, turning geometry, pedestrian flows, bus operations, and actual signal timing are not modeled. This is an uncalibrated proof of concept, not a forecast or signal-setting recommendation.

The ten anatomical samples are real SWC neuron skeletons from the 196-node circuit. Their geometries are rendered for interpretation; the network dynamics use connectivity, not the shapes. No existing motion-classifier weights were transferred: the traffic network was trained from scratch on the same graph.

The webpage replays saved seed-5100 evaluation traces, with model seed 11. Scenario controls select already-computed runs. Training is not performed in the browser. Full traces, all checkpoints, source hashes, and scripts accompany the report.
