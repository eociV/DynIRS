# DynCV: cross-fitted IR teacher (k = 5)

Separate from the in-sample teacher (`configs/queues/dyn/teacher.json`): different Kaggle code dataset,
kernel names, state file, local runs folder and ω folder. A teammate's `dynirs-*` sessions are never touched.

Teacher f trains on the IR train images of the other 4 folds (same YOLO26l recipe as
`make_dyn_queues.teacher`: 50 epochs, 640, batch 16, chunks of 10 epochs) and scores **only** the IR-only boxes
of fold f. A fold is made of blocks of 50 consecutive ids (neighbouring frames never cross the split),
balanced by IR-only box count.

## Steps (from the repository root)

With uv, prefix every command with `uv run` (it uses the repo's `.venv`): `uv run python -m dyncv run ...`.

```bash
pip install -r requirements-dev.txt                # mlflow, kaggle

python -m dyncv folds                             # configs/cv/folds_k5_b50_s0.json (already generated)
python -m dyncv queues                            # configs/queues/cv5/cv5_{all,f01,f23,f4}.json
python -m dyncv upload code                       # <user>/dynirs-cv-code; again after every code change
python -m dyncv prep                              # once: CPU kernel <user>/dyncv-prep downloads DroneVehicle from
                                                  # Hugging Face, prepares it, checks label hashes + fold ids
                                                  # against the local copy, saves dv_proc.tar (no upload from home)

# one queue = one chain of Kaggle sessions (2 folds per session, one per T4); can run in parallel on several accounts
python -m dyncv run configs/queues/cv5/cv5_f01.json     # data: dyncv-prep output (or --data-owner X for X/dynirs-dv)
python -m dyncv run configs/queues/cv5/cv5_f23.json --user <account 2> --code-owner <you>
python -m dyncv run configs/queues/cv5/cv5_f4.json  ...

python -m dyncv ui                                # MLflow at http://127.0.0.1:5000
python -m dyncv merge                             # proc/ironly_w/cv5_{T,TL,S} + comparison with in-sample c_i
python -m dyncv upload omega                      # <user>/dynirs-cv-omega for the student runs
```

`run` loops: push session → every `--poll` s read `kaggle kernels logs` (DYNCV_METRIC lines → live MLflow
metrics) → when it ends, download `runs/remote` into `runs/cv/remote/`, log results.csv, scoring stats and
best/last.pt to MLflow → if any fold is still training or unscored, push the next session (it mounts the
previous session's output and resumes from last.pt). Interrupted? Run the same command again.

Single steps: `push`, `status`, `fetch`, `sync`; `score --fold F` scores on the local GPU instead.
Students: config `ir_soft_w_dir: "auto:cv5_T"` (or `cv5_TL`, `cv5_S`).

## Adding a backend (e.g. Modal)

Write `dyncv/backends/<name>.py` with the methods of `backends/base.Backend` (`submit`, `status`, `logs`,
`fetch`, `active`, `mark`), register it in `backends/__init__.get`, and inside the container run
`python -m dyncv.session run --queue ... --data-in ... --code-in ... --work ...`. Folds, queues, scoring,
merging and tracking do not change.
