"""DynCV: cross-fitted IR teacher for the DynIRS per-box weights.

The in-sample teacher (configs/queues/dyn/teacher.json) scores the IR-only boxes of the very images
it was trained on. DynCV splits the IR train images into k folds; teacher f is trained on every fold
except f and scores the IR-only boxes of fold f only, so every c_i comes from a teacher that never
saw that image.

Modules (each one job, no module imports a backend except cli / orchestrator):
  config        all settings and paths (k, folds, names, slugs, MLflow location)
  folds         train ids -> k disjoint folds of consecutive-id blocks
  queues        one teacher job per fold, same recipe as scripts/make_dyn_queues.teacher
  score         teacher of fold f -> signals (c, v) for the IR-only boxes of fold f
  merge         k fold signals -> one signals file -> per-box weights (variants cv5_T, cv5_TL, cv5_S)
  progress      in the cloud session: tail results.csv and print one DYNCV_METRIC line per epoch
  tracking      local MLflow: epochs, scoring stats, checkpoints
  session       entry point inside a cloud session (any backend)
  orchestrator  backend-agnostic loop: submit, watch, fetch, track, repeat until the queue is done
  backends/     kaggle (now); add modal.py with the same four methods to scale out
  cli           python -m dyncv <command>
"""
