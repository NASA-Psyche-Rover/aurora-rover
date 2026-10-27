# Movement decision: existing sim vs v2

Same tests on both, headless, each with its own default site. Reproduce:

```
./vpy tools/movement_compare.py          # -> movement_results.json
./vpy tools/idle_same_gravity.py         # -> movement_idle_same_gravity.json
```
(`vpy` = `NASA Psyche/venv/bin/python`, MuJoCo 3.13.)

Labels: CAD / DATASHEET / PAPER / YOUR NUMBER / PLACEHOLDER for inputs. **SIM** = a result measured in this run.

## Results

| Test | existing `MuJoCo Sim/psyche_cubli.py` | v2 `psyche_cubli_v2.py` | Better |
|---|---|---|---|
| a) Geometry | 170 mm edge (YOUR NUMBER, "i think") | 152.0 mm (CAD, from the assembly STL) | v2 |
| a) Mass | 1.0 kg (PLACEHOLDER), uniform-cube inertia m·L²/6, wheels = 150 g rings at r 45 mm (PLACEHOLDER) | 900 g (YOUR NUMBER); motors 72 g (DATASHEET); per-part volume, centroid and inertia from the CAD split; CoM offset (−4.4, 1.1, 5.3) mm (CAD) | v2 |
| b) Motor model | straight torque–speed line, stall 0.05 N·m (PLACEHOLDER), no-load 3020 rpm (listing); no current, resistance or back-EMF; separate 0.5 N·m mechanical brake (PLACEHOLDER); no bearing friction | GM3506: 5.6 Ω (DATASHEET), 141 rpm/V (DATASHEET), 2 A limit (PLACEHOLDER), 11.1 V (YOUR NUMBER), back-EMF current model, electrical braking, Coulomb + viscous bearing friction from the no-load current (DATASHEET) with feed-forward, 200 Hz loop with IMU noise (PLACEHOLDER) | v2 |
| c) Idle drift, 5 s, own default gravity | 0.52 mm end, 0.90 mm worst (SIM) at 0.144 | 0.06 mm end, 0.18 mm worst (SIM) at 0.0966 | v2 |
| c) Idle drift at the **same** gravity, worst excursion (SIM) | 0.0885: 0.91 · 0.0966: **1.04 (fail)** · 0.1234: **1.13 (fail)** · 0.144: 0.90 mm | 0.14 · 0.18 · 0.12 · 0.10 mm | v2 |
| c) Ablation: v2 with its rounded hull swapped for a box (SIM) | – | 0.70 · 0.75 · 0.89 · **1.13 (fail)** mm: the box hull, not the rest of the sim, causes the jitter | v2's hull |
| d) Gravity | 0.144 m/s² hard-coded, straight down (old Wikipedia value) | 0.0885–0.1234 m/s² computed from the Shepard 2021 ellipsoid (PAPER: 278×238×171 km, 2.287e19 kg, 4.195948 h) + spin, per site, with geoid tilt | v2 |
| e) Rolls +x +y −x −y (SIM) | 4/4 OK, 16.8–17.1 cm, 5.5 s each (fixed torque profile) | 4/4 OK, 14.6–19.2 cm, 12.8–16.7 s each (spin-up, brake, tip) | both pass |
| e) Edge balance (LQR), 5 s hold (SIM) | not implemented | held 5 s | v2 |
| e) Release from balance (SIM) | not implemented | OK, settles on the next face | v2 |
| e) Hop (SIM) | 1.35 m, 5 s airborne, 46 cm apex, but only because of the 0.5 N·m mechanical brake (PLACEHOLDER) | 14–18 cm, with electrical braking only (the GM3506 cube has no brake) | existing hops further on a brake the hardware doesn't have |
| f) Speed vs real time (SIM, this Mac) | idle 94×, rolls 41×, hop 29× | idle 26×, rolls 21×, balance/hop 17× | both faster than real time |

## Decision: v2 wins. Use it entirely.

v2 is better or equal on every realism criterion:
- CAD geometry and mass layout, instead of placeholders.
- A physical motor model, instead of a torque line plus a brake that doesn't exist.
- The rounded hull, which is 5–10× steadier at the same gravity and is the only one that stays under 1 mm at every Psyche site.
- Gravity from Psyche's real shape and spin.
- The only one of the two that can balance on an edge and release.

The existing sim's longer hop comes from a 0.5 N·m mechanical brake that is a PLACEHOLDER and is not on the CAD cube. It is not evidence of better movement.

**Ported from the existing sim: nothing.** Candidates the winner lacks:
- A* dice-walk route planner (`dice_plan`)
- calibrated hop range table and hop planner (`calibrate`, `choose_hop`)
- click-on-map targeting
- in-flight attitude control

Each would add capability without reducing realism. They are not needed for this task, so I didn't port them. v2's `T` (go to the camera target) covers simple driving. The hop table and planner would have to be recalibrated on v2's motor model before they could be trusted.
