# Figure-8 Diverse Expert Dataset Artifacts

This folder contains the versioned audit artifacts for the diverse figure-8 expert dataset used before ACT + CVAE + language-intent retraining.

## Dataset

Local dataset path:

```text
/workspace/turbopi_isaac/data/act_figure8_diverse_128
```

Sessions:

```text
figure8_diverse_train_64
figure8_diverse_val_64
```

Accepted episodes:

```text
128 total
64 go_left
64 go_right
```

Rejected off-track attempts were excluded from training:

```text
train session: 19 rejected
validation session: 6 rejected
```

## Artifacts

- `figure8_expert_paths_on_topdown_map.png`: main top-down map overlay for slides.
- `figure8_expert_paths_overlay.png`: split left/right pose traces against the reference route.
- `figure8_expert_progress_projection.png`: route-progress coverage diagnostic.
- `figure8_expert_paths_summary.csv`: per-episode frame counts, duration, final progress, mean error, and max error.

## Slide Notes

Use `figure8_expert_paths_on_topdown_map.png` to explain that the accepted expert trajectories stay inside the road while still covering diverse lanes and offsets.

Use `figure8_expert_paths_overlay.png` when comparing left-intent and right-intent behavior separately.

The dataset writes actual `pose_x`, `pose_y`, and `pose_yaw` columns for audit plots. The trainer still uses onboard camera frames plus the saved action chunks and language intent labels.
