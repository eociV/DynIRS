# Dynamic per-box IR soft-label weights (DynIRS) — design

Date: 2026-11-23
Status: draft, awaiting user review

## 1. Goal

Extend IRS-YOLO26n for a follow-up paper. The original recipe adds vehicles annotated only in the
IR frame (IR-only boxes) to the RGB detection loss with a **constant** weight ω = 0.3. This work
replaces the constant with a **per-box weight** computed from a strong IR teacher, and tests whether
the effect holds across several nano YOLO architectures.

**Research questions** (configuration codes are defined in §4)

| RQ | Question | Comparison | Metric |
|---|---|---|---|
| RQ1 | Does a per-box ω beat the constant ω = 0.3 when the mean weight is held equal, and is the gain due to the teacher's information rather than to weight variance? | **T − C**, and **T − S** | RGB AP50:95, overall and night |
| RQ2 | Does adding darkness to ω (trusting the teacher more where RGB is dark) help beyond teacher confidence alone? | **TL − T** (and TL − C) | As RQ1, with emphasis on night images and IR-only vehicle recall |
| RQ3 | Do the gains transfer from YOLO26n to YOLOv8n, YOLOv10n, YOLO11n and YOLO12n? | Sign and size of T − C and TL − C per model | As RQ1 |

C − B is also reported, to show that the original paper's result reproduces under the new protocol
and on the other models.

**Success criteria**

- Main result: on YOLO26n, T − C or TL − C is positive, with a Holm-corrected 95 % CI that excludes
  0, at least for night RGB AP50:95.
- Generalisation: T − C or TL − C is positive on at least 3 of the 4 other models.
- Mechanism: T > S on YOLO26n. If T ≈ S, the gain is regularisation from weight variance rather
  than teacher information, and the paper's claim must be reframed.

**Out of scope:** thermal-guided feature distillation (TGD), larger students such as 26s, training
at 60 epochs / 704 px, and learned (trainable) weights.

## 2. Weight functions

All signals are computed **offline, once**, on the train split.

- `c_i` — IR teacher confidence for IR-only box *i*: run the teacher on the IR frame and take the
  highest-scoring prediction with IoU ≥ 0.5 to box *i*. If no prediction matches, `c_i = 0`.
- `v_i` — RGB visibility of box *i*: mean luminance of the RGB pixels inside the box, scaled to
  [0, 1]. Use the same luminance definition as `src/stage0_manifest.py`, so values are comparable
  with the night / twilight / day illumination bands.

The two variants:

| Variant | Raw weight | Meaning |
|---|---|---|
| ω-T | `c_i` | Trust the teacher alone |
| ω-TL | `c_i · (1 − v_i)^γ`, γ = 1 | Trust the teacher more where the RGB image is dark and cannot confirm the vehicle. A bright IR-only box is more likely a registration error or teacher false positive. |

Post-processing for both variants, in this order:

1. Scale by a single constant *s* so the train-set mean equals 0.3.
2. Clip to [0.05, 1].
3. Find *s* by bisection so that the mean **after clipping** is still 0.3 ± 0.001.

This mean-matching is the key control: the total soft-label mass equals the constant-ω baseline, so
any difference comes only from how the weight is redistributed across boxes. Hyperparameters
(γ = 1, floor 0.05, IoU 0.5) are fixed in advance and not tuned.

## 3. IR teacher

- Model: YOLO26l, initialised from the COCO checkpoint `yolo26l.pt`, trained on the IR frames of the
  DroneVehicle train split.
- Settings: 640 px, batch 16, 50 epochs. Reduce to 30 epochs if the measured epoch time × 50
  exceeds 12 h.
- Hardware: the user's RTX 5060 Ti (16 GB). Use `cache: false` or `"disk"`, because the machine has
  16 GB of RAM.
- Acceptance check: IR val AP50:95 must be clearly higher than the YOLO26n IR teacher (`t_ir`)
  reported in the original paper (640 px, 60 epochs). The old teacher is not retrained. If the new
  teacher is not better, stop and review with the user.
- The teacher is used only to compute `c_i`. It is never loaded during student training.

## 4. Experiment matrix

**Shared protocol:** 40 epochs, 640 px, batch 32 (lower it only if a model runs out of memory, and
record the change), the train / val / test splits under `splits/`, and soft-label-only training:
`paired: true`, `tgd_lambda: 0`, `ir_only_dir: auto`.

**Four configurations per model and seed:**

| Code | Configuration |
|---|---|
| B | Baseline, RGB labels only. Floor: shows whether IR labels help at all. |
| C | IR soft labels with constant ω = 0.3 (job `m2_soft` in the original paper). **Main competitor.** |
| T | IR soft labels with ω-T |
| TL | IR soft labels with ω-TL |
| S | Shuffled control, YOLO26n only: the ω-T values randomly permuted across IR-only boxes (fixed permutation seed 0). Same mean and distribution as T, no link between ω and the box. |

The union configuration (ω = 1) is not rerun; the original paper showed it is worse than ω = 0.3.

**Hardware assignment.** Each model is trained entirely on one GPU type, so the B / C / T / TL
comparison for that model is not confounded by hardware.

| Model | Seeds | Runs | Hardware |
|---|---|---|---|
| YOLO26n | 0, 1, 2 | 15 (B, C, T, TL, S) | user's RTX 5060 Ti |
| YOLOv8n | 0 | 4 | T4 (Kaggle / Colab) |
| YOLO11n | 0 | 4 | T4 |
| YOLOv10n | 0 | 4 | T4 |
| YOLO12n | 0 | 4 | T4 |
| **Total** | | **31 students + 1 teacher** | |

**Order, with gates:**

0. Benchmark: time one epoch of 26n, 12n and 26l on the local GPU, and of 26n on a Kaggle T4. Lock
   the epoch count and the matrix from these numbers.
1. Local GPU: teacher 26l. In parallel on T4: B and C for v8n and 11n, which need no teacher.
2. Local GPU: compute ω, then build and upload the Kaggle dataset with the weight files.
3. Local GPU: 26n B / C / T / TL / S with seed 0. **Gate:** if neither T nor TL beats C on overall RGB
   AP50:95 or night RGB AP50:95, stop and review with the user before spending more compute.
4. T4: T and TL for v8n and 11n, then all four configurations for v10n and 12n.
5. Local GPU: 26n seeds 1 and 2.

**If the budget runs short, cut in this order:** 12n, then v10n, then 26n seed 2.

## 5. Evaluation

- Reuse `src/evaluate.py`: COCO AP50:95 against both the RGB and the IR annotations, overall and on
  the night band, on the test split.
- Also report recall of IR-only vehicles (`analysis_*.py`).
- YOLO26n (3 seeds): seed-paired differences T − C, TL − C, TL − T, T − S and C − B, with t confidence intervals
  and Holm correction, as in `make_rev1_tables.py`.
- Single-seed models: image-level paired bootstrap (`rev1_boot.py`). The paper must state that
  these models have one seed.
- Latency of the deployed models is unchanged, because ω exists only at training time. No new
  latency measurement is needed.

## 6. Code changes

### 6.1 `src/omega.py` (new)

- Inputs: the teacher weights, the processed data dir, and the variant (`T`, `TL` or `S`; `S`
  permutes the final ω-T values across all IR-only boxes of the train split with seed 0).
- Writes `proc/ironly_w/<variant>/train/<id>.txt` with one line `cls x y w h ω` per IR-only box,
  in the same box order as `proc/ironly_cls/train/<id>.txt`.
- Writes `proc/stats/omega_<variant>.json` with the scale *s*, the mean, a histogram, and the
  fraction of boxes clipped at each bound.
- Pure functions for matching, luminance and mean-matching, kept separate from file I/O so they
  can be unit-tested.

### 6.2 `src/lowlight.py`

- New config key `ir_soft_w_dir`.
- When it is set, `update_labels_info` reads the 6-column files and stores ω for each box. Its
  value travels alongside `cls` and is carried through augmentation.
- `preprocess_batch` uses the per-box ω for soft boxes instead of the constant `ir_soft`.
- When `ir_soft_w_dir` is unset, behaviour is unchanged.
- Implementation note: Ultralytics augmentations (mosaic, random perspective) filter and reorder
  boxes, so ω cannot be kept in a separate array. Encode it into the class value. For example,
  `cls = SOFT_OFFSET + k`, where `k` indexes a quantised table of 1000 levels, so ω has a
  resolution of 0.001. `preprocess_batch` then decodes it. This reuses the existing offset trick
  and keeps the change local to `lowlight.py`.

### 6.3 `src/job_main.py`

- Resolve `ir_soft_w_dir: "auto:<variant>"` to `proc/ironly_w/<variant>/train`.
- Add a `teacher` training kind that trains on the IR frames, if the existing `t_ir` path does not
  already cover YOLO26l.

### 6.4 Model support

- Check that YOLOv8n and YOLO12n train through the patched `v8DetectionLoss` (with
  `box_w ≠ 1`). YOLO26n and YOLOv10n already work in the original paper.
- YOLOv10n uses `v10DetectLoss`, which wraps two `v8DetectionLoss` instances, so the patch applies
  to both branches. Confirm this with a 1-epoch smoke run.

### 6.5 Queues and notebooks

- `configs/queues/dyn/*.json`: one file per (model, hardware) group.
- `notebooks/kaggle_runner.ipynb`:
  - installs `requirements.txt`,
  - mounts the private Kaggle dataset with the prepared data and the ω files,
  - runs a queue with resume across 12-hour sessions, using the existing `chunks` mechanism,
  - writes outputs to `/kaggle/working`, to be downloaded after each session.
- Colab uses the same notebook, with Google Drive as storage.

### 6.6 Environment

- Use a separate Python 3.11 virtual environment with `ultralytics==8.4.173`. The machine's default
  Python is 3.13.
- The RTX 5060 Ti (Blackwell) needs a CUDA 12.8 PyTorch build.

## 7. Testing

- Unit tests for `omega.py`:
  - box matching on synthetic boxes,
  - the luminance of a constant image,
  - the mean equals 0.3 within 0.001 after clipping,
  - every weight lies in [0.05, 1].
- Encode / decode round-trip test for the ω-in-class encoding, including after mosaic and
  perspective augmentation on a sample batch.
- **Regression test (most important):** with every per-box ω set to 0.3, the loss on a fixed batch
  must equal the loss of the constant `ir_soft: 0.3` path to within 1e-6.
- Smoke run: 1 epoch on a 200-image subset for every model (26n, v8n, v10n, 11n, 12n) with
  configuration TL.

## 8. Risks

| Risk | Mitigation |
|---|---|
| The teacher's confidence on IR-only boxes is low, so most `c_i` are 0 and the weights collapse to the floor. | Check the histogram in `omega_<variant>.json` before step 3. If more than half of the boxes sit at the floor, review IoU and floor with the user. |
| ω-TL gives high weight to boxes with no RGB signal and adds noise. | That is the hypothesis under test; a negative result is still reported. |
| Kaggle throughput is limited by its 4 CPU cores. | Measure in step 0. Use `cache: "ram"` on Kaggle (about 30 GB of RAM). |
| The results are not directly comparable with the original paper (40 epochs / 640 px versus 60 / 704). | All four configurations are retrained under the same protocol. State this in the paper. |
