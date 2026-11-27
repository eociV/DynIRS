# Dynamic ω and repeat sampling of IR-only vehicles — design

Date: 2026-11-27
Status: approved in discussion (2026-11-27); builds on 2026-11-25-clean-ir-labels-design.md (variant C)

## 1. Motivation

Screening of the clean-label study (docs/results/2026-11-25-screening-clean.md, 3 seeds, val):

- **C** ("clean P"): registration, misaligned-image ignore, RGB-duplicate drop, the rest taught as in P.
  - C matches P in AP: −0.03 night RGB AP.
  - C raises IR-only night recall by +1.8 pt and faint-vehicle recall by +4.4 pt.
  - C adds 34 false alarms, which is not significant.
- Ignoring the unverified boxes (I, V, F) loses to P. The constant ω = 0.3 positive signal matters.
- Feature distillation from an IR teacher (TGD) is not pursued. The IRS-YOLO26n manuscript found that its contribution beyond the soft labels is not established. Lou et al. (AAAI 2026) cover IR-privileged feature hallucination.

**Remaining limits.**
- **Dilution.** The taught IR-only boxes are about 32 k with mean ω ≈ 0.34, against 283 k RGB labels at ω = 1. That is about 3–4 % of the supervision.
- **ω = 1 for every IR-only box over-teaches.** The IRS-YOLO26n manuscript reports that it raises false alarms without an RGB-label gain, because most IR-only vehicles are invisible in RGB.

**Idea.** Raise the share of the IR-only boxes that the student can actually learn, and only those:
- (a) a **dynamic ω**: the student re-scores the IR-only boxes during training;
- (b) **repeat sampling** of the images whose IR-only boxes are learnable.

**Research questions**

| RQ | Question | Comparison |
|---|---|---|
| RQ5 | Does repeat sampling of images with learnable IR-only boxes help? | CB − C* |
| RQ6 | Does the dynamic ω help? | D − C* |
| RQ7 | Do the two combine? | DB − max(CB, D) |

Metrics, seeds and the screening protocol are those of the clean-label study:
- YOLO26n, single class, fixed 20 % train subset, 15 epochs, 704 px, batch 32, seeds 0–2, scored on val;
- metrics: night RGB/IR AP50:95, IR-only night recall by type, false alarms.

A configuration counts as better when its night RGB or IR AP is higher with a seed-paired 95 % CI that excludes 0, without significantly more false alarms. Negative results are reported.

## 2. Dynamic ω (stage-wise re-scoring)

**Start.** The C weights: `proc/ironly_w/C/train`.

**Stages.** Training is split into 3 equal stages:
- screening (15 epochs): re-scoring after epochs 5 and 10;
- full scale (60 epochs): re-scoring after epochs 20 and 40.

**Re-scoring at the end of a stage:**
- **Input.** The current `last.pt` predicts on the run's train images that hold IR-only boxes (for screening, only those in the run's 20 % subset).
- **Score.** For each IR-only box at its (registered) position, s = the highest score of a student detection with IoU ≥ 0.5. This is `omega.teacher_conf`, the same rule as the teacher score.
- **Update.** The rules below are fixed in advance and not tuned on results. ω_old is the box's current weight; c is the cross-fitted RGB-teacher score from the clean-label build.

| Student score | New ω |
|---|---|
| s ≥ S_HI = 0.5 | max(ω_old, s) |
| S_LO = 0.05 ≤ s < S_HI | ω_old |
| s < S_LO and c < S_LO | −1 (ignore region) |
| s < S_LO and c ≥ S_LO | ω_old |

- **Boxes not updated:** ignore regions from the misaligned-image filter stay ignored, and dropped RGB duplicates stay dropped.
- **Storage.** The weights live in the run's own directory: `ws/omega_dyn/stage<k>/train/<id>.txt`, plus `stats_stage<k>.json` (counts of raised, kept and ignored boxes, and the mean ω). Two runs in parallel lanes never share a weight directory.
- **Why early.** With noisy labels a network fits the clean labels first and memorises the wrong ones later (small-loss criterion). Scoring at one and two thirds of training keeps s closer to learnability than to memorisation.

## 3. Repeat sampling

- **Rule.** In each stage, a train image with at least one IR-only box of ω ≥ 0.5 appears R = 2 times per epoch. Other images appear once.
  - For B, P and C nothing is repeated.
  - For CB the repeated set is fixed by the C weights.
  - For DB it is recomputed after every re-scoring.
- **Effect.** Epochs get longer. The number of extra images per epoch is logged, and the results tables report the optimizer steps of each config. CB is the control for the extra steps.
- **Implementation.** First check whether Ultralytics keeps duplicate entries of an image list (label cache keyed by file path). If it does not, repeat the indices in the dataset (`lowlight` dataset hook).

## 4. Configurations (screening)

| Code | Labels | ω update | Repeat |
|---|---|---|---|
| B, P, C | existing runs (local GPU) | – | – |
| C* | C, rerun on Kaggle | – | – |
| CB | C | – | yes |
| D | C | stage-wise | – |
| DB | C | stage-wise | yes |

- C* checks the GPU change (RTX 5060 Ti → T4). The new configs are compared with C*; C − C* is reported.
- All new configs use `ir_ignore=True`, since ignore regions can appear.
- Job names: `dyn_cl_<code>_s<seed>`. Queues: `t4_dyn_s<seed>` with C*, CB, D and DB, so 12 runs.

## 5. Kaggle execution

- **Account and token.** The user's Kaggle account, phone-verified for GPU access. The API token `kaggle.json` goes in `C:\Users\admin\.kaggle\`. It is created by the user, or by Claude in the user's logged-in Chrome when the user asks.
- **Datasets** (private), built with `scripts/pack_kaggle.py` and uploaded with the Kaggle CLI (large files):
  - `dynirs-dv`: RGB images and labels, IR labels, stats. IR images are left out; they are not needed and leaving them out halves the size.
  - `dynirs-omega`: the C weights and the per-box teacher score c.
  - `dynirs-code`: the code.
- **Notebook.** Created and run by Claude through Claude in Chrome in the user's logged-in browser:
  - attach the 3 datasets, choose accelerator GPU T4×2, internet on (Ultralytics weights);
  - run `scripts/kaggle_session.py` with 2 lanes;
  - commit with "Save & Run All", so the run continues server-side without the tab.
- **Results.** Downloaded from the notebook output into `runs/remote/` and scored locally with `src/screen_tables.py`.
- **Estimate (unmeasured on T4).** About 25–30 min per run including two re-scorings, so 12 runs on 2 lanes take about 3 h. That is one session and about 3 of the 30 weekly GPU hours.

## 6. Code changes

- `src/dyn_omega.py` (new): `rescore(run_dir, weights_in, preds, teacher_c) -> (weights_out, stats)` with the rule of §2, and `repeat_ids(weights) -> ids` with the rule of §3. CLI for one run directory.
- `src/job_main.py`: a job option `dyn_omega: {"stages": [5, 10], "repeat": true|false}` (or `"repeat"` alone for CB).
  - At each stage boundary (a chunk boundary), predict on the IR-only train images, call `rescore`, and point `ir_soft_w_dir` at the new stage directory.
  - Chunks are set to the stage boundaries.
- `src/lowlight.py`: repeat sampling, if Ultralytics drops duplicate list entries.
- `src/clean_labels.py`: write the per-box teacher score c of variant C (`proc/stats/clean_C_conf.json`).
- `scripts/make_dyn_queues.py`: queues `t4_dyn_s{0,1,2}`.
- `scripts/pack_kaggle.py`:
  - `pack_data` gains an option to leave out the IR images;
  - `pack_omega` adds the C teacher scores.
- `src/screen_tables.py`: codes C*, CB, D and DB, plus a column for optimizer steps.

## 7. Testing

- **Unit tests:**
  - every row of the update table, and frozen boxes stay frozen;
  - the repeat rule;
  - per-run weight directories, so two runs never share one;
  - a chunk boundary triggers re-scoring and the next chunk reads the new weights;
  - the queue contents.
- **Local smoke run** (1 + 1 + 1 epochs, few images) on the local GPU before uploading. It must show:
  - two re-scorings with plausible stats;
  - the repeated images counted in the epoch length;
  - a full prediction file.
- **Before the Kaggle runs:** `scripts/check_wsoft_loader.py` on a stage-2 weight directory.

## 8. Risks

- **Memorisation.** If the student memorises taught boxes by epoch 5, s rises for wrong boxes and D over-teaches. Watch the false alarms and the share of raised boxes per stage.
- **Small effect.** C already matches P. The gains may stay within seed noise at 20 % / 15 epochs. A negative RQ5–RQ7 is reported as such.
- **Kaggle quirks.**
  - The accelerator choice and the dataset attachment are UI steps. Claude in Chrome needs the user's logged-in session.
  - A failed upload or a session limit can delay the runs. The local GPU stays the fallback.
- **Compute comparability.** Repeat sampling adds steps. CB isolates that effect.
