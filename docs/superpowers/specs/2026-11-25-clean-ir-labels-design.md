# Clean IR-only labels for single-class RGB vehicle detection — design

Date: 2026-11-25
Status: approved in discussion (2026-11-25); replaces the ω variants of the 2026-11-23 spec and its amendments

## 1. Goal

IRS-YOLO26n trains an RGB detector with extra boxes from the IR annotation (IR-only boxes: vehicles
annotated in the IR frame with no matching RGB box, Hungarian matching at IoU ≥ 0.3). All of them enter
the loss with a constant weight ω = 0.3. The analyses of 2026-11-23..25 showed:

- About 9 % of IR-only boxes are displaced or empty in RGB. 2,284 have an RGB-teacher detection beside
  them (IoU 0.1–0.5) but not on them, and 886 lie on daytime RGB where the scene is clearly visible and
  empty. The constant-ω recipe teaches these as vehicles.
- RGB–IR misregistration is mostly a per-image translation. Shifting the IR-only boxes by the image's
  median pair offset raises the cross-fitted RGB-teacher confidence on the 6,025 moved boxes from 0.17
  to 0.24 (boxes with c ≥ 0.3: 1,522 → 2,072). The teacher played no part in estimating the shift.
- The IR teacher's confidence is uninformative (about 0.75 on every box type). A cross-fitted RGB
  teacher separates the types. On val at night, its detections at score ≥ 0.5 are 92.8 % on a labelled
  vehicle, 2.5 % displaced and 4.7 % on nothing. The "on nothing" rate is no higher in dark regions
  (3.8 %) than in visible ones (5.5 %).
- RGB and IR class labels agree on 99.9 % of matched pairs.

This work cleans the IR-only labels (registration, filtering, verification by a cross-fitted RGB
teacher) and asks what an RGB detector gains from them, on a single-class "vehicle" task.

**Research questions**

| RQ | Question | Comparison (§4) |
|---|---|---|
| RQ0 | Does the original result reproduce on the single-class task? | P − B |
| RQ1 | Is the gain from *teaching* IR-only vehicles, or only from *no longer penalising* detections on them? | I − P |
| RQ2 | Does teaching only the IR-only vehicles the RGB teacher can see, and ignoring the rest, beat ignoring all? | V − I |
| RQ3 | Do registration and filtering of misaligned images add to verification? | F − V |

**Metrics.** Single-class COCO AP50:95 against the RGB and the IR annotation, over all images and over
night images. Night recall of IR-only vehicles. Night false alarms: detections at score ≥ 0.25 that match
(IoU ≥ 0.5) neither annotation. This is the measure of the student "guessing".

**Success.** Screening (3 seeds) picks configurations by the sign and size of the seed-paired
differences. The full-scale runs (§5) give the paper's claims. A configuration counts as better when its
night RGB AP50:95 is higher with a 95 % CI that excludes 0, without more night false alarms. A negative
RQ2 or RQ3 is also a reportable result.

**Out of scope:** thermal-guided feature distillation and any other use of the IR teacher in training,
multi-class detection, other datasets, and other student architectures.

## 2. Label pipeline (offline, train split)

Inputs: proc/{rgb,ir}/train/labels, proc/ironly_cls/train (IR-only boxes),
proc/stats/pairs_train.json (matched-pair centre offsets), and the RGB-teacher predictions (§3). All
classes are mapped to 0 ("vehicle").

| Step | Rule |
|---|---|
| ① Registration | For an image with ≥ 3 matched pairs, d = median over pairs of (IR centre − RGB centre). The IR-only boxes are moved by −d and clipped to the image. Images with < 3 pairs are not moved. |
| ② Misaligned images | Residual r = median over pairs of ‖offset − d‖. If r > 4 px, every IR-only box of the image becomes an ignore region (1,093 images exceed 4 px; 319 of them have IR-only boxes, 1,621 boxes in all). |
| ③ Verification | For each remaining IR-only box (at its moved position), c = highest score of an RGB-teacher detection with IoU ≥ 0.5 to it. The detections come from the fold model that did not train on the image. |
| ④ Decision | c ≥ 0.5: taught, with ω = c. c < 0.5: ignore region. |
| ⑤ RGB duplicates (V and F) | A box with IoU ≥ 0.3 to an RGB label (at its final position) is that labelled vehicle, usually a misregistered duplicate: it is dropped (not written). With the 5-class teacher this drops 805 boxes in F (572 of the "dup" type), and taught duplicates fall from 249 to 17. |

Fixed constants: MIN_PAIRS = 3, MAX_RESIDUAL = 4 px, TAU = 0.5, DUP_IOU = 0.3 (the pair-matching IoU). They are not tuned on results.

Output per variant: proc/ironly_w/<variant>/train/<id>.txt with rows "0 cx cy w h ω". ω ∈ (0, 1]
is taught with weight ω, and ω = −1 is an ignore region. A stats JSON records counts per step and the
distribution of ω.

**Ignore region**: inside the box, background anchors get no classification loss. Anchors assigned to a
real box keep their loss. Already implemented (`lowlight.ignore_anchor_mask`, `ir_ignore`).

## 3. RGB teacher (retrained, single class)

- YOLO26l, single class, RGB images with RGB labels only (no IR labels), 640 px, batch 16, 30 epochs,
  seed 0, local GPU.
- Cross-fitted over 2 folds (job_main.fold_of, seed 0): fold model i trains on the train images outside
  fold i and predicts train and val. A train image's score comes only from the model that did not see it.
- Jobs: dyn_teacher_rgb1c_f0, dyn_teacher_rgb1c_f1. The 5-class teachers (dyn_teacher_rgb26l_f*) are
  kept only for the analyses already reported.

## 4. Student configurations

All student runs use YOLO26n, single class, RGB labels with ω = 1, plus the IR-only boxes as below.

| Code | Name | IR-only boxes |
|---|---|---|
| B | Baseline | none |
| P | Paper | all, ω = 0.3, not moved |
| I | Ignore | all as ignore regions, not moved |
| V | Verify | steps ③–④ on unmoved boxes, without ② |
| F | Full | steps ①–④ |

Each row from P to F changes one thing relative to the row above it.

## 5. Protocol and compute

**Screening**: B, P, I, V, F × seeds 0, 1, 2 = 15 runs. Seed 0 (5 runs) runs first and is reviewed with the user; seeds 1 and 2 and the full scale follow only if the user decides to continue. YOLO26n, fixed 20 % train subset (proxy seed
0), 15 epochs, 704 px, batch 32, scored on val. This is the proxy protocol of the original paper. Runs on
Kaggle T4 ×2, two runs per session.

**Full scale**: B, P and the best screened configuration (or two if close) × 3 seeds. 100 % train, 60
epochs, 704 px, batch 32, best.pt by validation fitness, scored on test (the original paper's protocol).

**Estimated time.** Only the teacher figure is measured. The others are extrapolated from it and from the
1-epoch local smoke run, assuming a T4 is about 2.5× slower than the local RTX 5060 Ti.

| Stage | Where | Estimate |
|---|---|---|
| Label pipeline | local CPU | < 30 min |
| RGB teacher, 2 folds | local GPU | about 4 h (measured: about 1 h 50 min per 5-class fold) |
| Screening, 15 runs | Kaggle T4 ×2 | about 25 min per run, about 3–4 h per session pair, about 6 h GPU in total |
| Full scale, 9 runs | local GPU | about 2.5–3 h per run, about 24–27 h |
| (alternative) Full scale on Kaggle | T4 ×2 | about 7–8 h per run, about 35 h wall time over 2 accounts |

## 6. Code changes

- `src/clean_labels.py` (new): steps ①–④ and the variants P, I, V, F. It reuses
  `omega.image_shifts`, `omega.shift_boxes` and `omega.teacher_conf`, and adds the residual and the
  fold-aware teacher scores. CLI: `python src/clean_labels.py --proc <proc> --variants P I V F`.
- Single class:
  - Every student and teacher job sets the Ultralytics override `single_cls: True`.
  - The injected IR-only rows are already class 0 (clean_labels.py writes class 0), so the loader needs
    no change; Ultralytics' single_cls maps the RGB labels to 0.
  - `evaluate.Matched` gains a class-agnostic mode (all GT and prediction categories set to 0), which the
    new score table uses.
- `scripts/make_dyn_queues.py`:
  - Emits only `teacher_rgb1c`, `t4_clean_s{0,1,2}` (screening) and `full_clean_s{0,1,2}` (full scale;
    its configuration list is set after screening).
  - Job names: `dyn_cl_<code>_s<seed>` (screening), `dyn_full_<code>_s<seed>` (full).
  - Old queue files are deleted from `configs/queues/dyn` (they stay in git history). The old ω variant
    code (T, TL, S, rank, type, SH/IG/SI) stays in the code but is no longer queued.
- `src/screen_tables.py` is generalised to the `dyn_cl_` runs, the codes B/P/I/V/F, the references B
  and P, and class-agnostic AP.
- `scripts/pack_kaggle.py` (omega package): includes `ironly_w/{P,I,V,F}`.

## 7. Testing

- Unit tests, each on small fixtures:
  - residual and misalignment filter;
  - registration direction (a box moves into the RGB frame);
  - verification uses only the held-out fold's predictions;
  - decision rule at c = 0.5;
  - output rows for taught and ignored boxes;
  - single-class class mapping in the loader;
  - class-agnostic evaluation;
  - queue contents (names, `single_cls`, config per code).
- Real-data checks:
  - `scripts/check_wsoft_loader.py` on V and F, which must show both taught and ignore boxes;
  - pipeline stats compared with the figures in §1 and §2 (319 images and 1,621 boxes ignored by the residual, 6,025 moved boxes).
- Sanity sheet: 24 random crops each of taught, ignored-by-teacher and ignored-by-residual boxes, for a
  visual check before training.

## 8. Risks

- **Fewer IR-only boxes taught.** Only about 10 % of IR-only boxes reach c ≥ 0.5, so V and F may differ
  little from I. That is still an answer to RQ2.
- **Teacher bias.** The teacher has the same architecture family as the student and learned from the
  same RGB labels. ω then favours what RGB detectors already find. Taught boxes may add little new
  signal, while ignore regions still remove the false penalty.
- **Class change.** The single-class change breaks direct comparison with the original paper's 5-class
  numbers. B and P are rerun under the new task, so the comparison is internal.
- **Unmoved images.** Images with < 3 pairs (36 % of IR-only boxes, mostly dark night) keep their
  misregistration. Verification (step ③) is their only safeguard.
