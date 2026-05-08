# TurboPi Figure-8 Training Pipeline

This document is the running reference for the figure-8 map work. Keep it updated whenever the data collection, training, inference, or video export flow changes.

## Current Scope

The current target is instruction-conditioned driving on the `figure8` map:

- `go_left`: drive the shared center arm, take the left loop, return to the shared start, and repeat.
- `go_right`: drive the shared center arm, take the right loop, return to the shared start, and repeat.
- One episode defaults to `3` complete loops for a single intent.
- The model input image is the robot-mounted forward camera.
- The language intent is saved as the instruction string, currently `go left` or `go right`.
- The action is the autonomous teacher command normalized for ACT/CVAE training.

The original cloned mountain route is preserved as the `original` map. The figure-8 route is an alternate map named `figure8`.

## Files

- `scripts/mountain_cliff_scene.py`
  - Defines the `original` and `figure8` maps.
  - `MountainCliffSceneCfg(map_name="figure8")` selects the new map.
  - `route_waypoints(scene_cfg, task_name)` returns the left or right route for collection.
  - Guard rails and end caps are disabled for `figure8`, so the car has an unobstructed path.

- `scripts/record_turbopi_mountain_act.py`
  - Records ACT episodes from the robot-facing camera.
  - Supports `--map figure8`, `--task go_left|go_right|mix`, and `--laps 3`.
  - Adds per-episode diversity through start jitter, yaw jitter, speed jitter, action noise, and camera-pose jitter.

- `scripts/collect_figure8_act_parallel.sh`
  - Launches separate headless Isaac workers for left and right data collection.
  - This is the fast path for generating balanced left/right data without manual interaction.

- `scripts/record_turbopi_figure8_act_vec.py`
  - Runs vectorized figure-8 ACT collection in one Isaac Lab process.
  - Clones many figure-8 arenas with one TurboPi and one robot camera per environment.
  - Uses batched route following and per-environment resets.
  - This is the preferred fast path for future collection.

- `scripts/act_mountain_dataset.py`
  - Writes each episode as `video.mp4`, `data.parquet`, and `episode_info.json`.
  - Session metadata records the map name and laps per episode.

## What One Episode Contains

For each episode:

1. The robot is reset near the start of the shared straight segment.
2. A single intent is chosen: `go_left` or `go_right`.
3. The waypoint route is repeated `--laps` times.
4. The autonomous teacher drives the car along the route.
5. At each control step, the recorder saves:
   - RGB image from the robot-mounted forward camera.
   - Previous action as state.
   - Current normalized action.
   - Raw velocity command `[vx, vy, wz, stop]`.
   - Track error.
   - Route progress.
   - Task name, task index, and language instruction.
6. The episode is accepted only if the route is completed successfully.

With the default `--laps 3`, a left episode contains three left loops, and a right episode contains three right loops.

## Diversity Sources

The collector is intentionally deterministic enough to stay on the track, but varied enough to avoid a brittle dataset.

- `--speed_jitter`
  - Randomizes teacher speed per episode.
  - Default: `0.15`, meaning plus or minus 15 percent around `--target_speed`.

- `--start_xy_jitter`
  - Randomizes the initial x/y position near the route start.
  - Default: `0.025` meters.

- `--start_yaw_jitter`
  - Randomizes initial heading.
  - Default: `0.08` radians.

- `--camera_xyz_jitter`
  - Randomizes the forward camera eye and target offsets per episode.
  - Default: `0.015` meters.

- `--action_noise_std`
  - Optional command noise.
  - Default: `0.0`; increase carefully if the teacher remains stable.

## Fast Collection

There are two levels of collection speed:

1. Current committed path: process-level parallelism.
2. Required two-minute path: Isaac Lab vectorized collection.

The current committed path is useful and stable, but it is not enough for a two-minute large dataset target because each worker still launches a full Isaac process with one robot and one camera.

### Current Process-Level Parallelism

```bash
cd /workspace/turbopi_isaac
EPISODES_PER_WORKER=100 WORKERS_PER_INTENT=1 ./scripts/collect_figure8_act_parallel.sh
```

This launches two headless Isaac jobs:

- one worker for `go_left`
- one worker for `go_right`

Each worker writes a separate session under:

```text
data/act_figure8/
```

For more parallelism on a large GPU:

```bash
cd /workspace/turbopi_isaac
EPISODES_PER_WORKER=100 WORKERS_PER_INTENT=2 ./scripts/collect_figure8_act_parallel.sh
```

That launches four jobs total: two left workers and two right workers. Watch GPU memory before increasing worker count.

### Two-Minute Target

The two-minute target requires a vectorized collector, not just more shell workers.

#### What Vectorized Means In Isaac Lab

In a normal standalone Isaac script, the simulator contains one world and one robot. The loop looks like this:

```text
reset one robot
drive one episode
save one episode
reset the same robot
drive the next episode
...
```

That is simple, but it wastes time because every episode waits for the previous episode to finish.

In Isaac Lab, a vectorized setup creates many cloned environments inside one simulator process. Each clone has its own local origin, its own robot, its own camera, and its own episode state. The simulation loop then advances all environments together:

```text
env_000: TurboPi + camera + route state
env_001: TurboPi + camera + route state
env_002: TurboPi + camera + route state
...
env_063: TurboPi + camera + route state

single sim.step() advances all 64 environments
single batched teacher computes actions for all 64 robots
single camera update returns a batch of 64 images
finished envs reset individually while unfinished envs continue
```

This is the key idea: `num_envs=64` does not mean launching 64 Isaac applications. It means one Isaac application with 64 copies of the task inside it.

The benefit is that GPU work is batched. Instead of paying startup, scene setup, and Python overhead for each robot separately, we pay those costs once and collect many episodes at the same time.

In code, Isaac Lab usually expresses this through `InteractiveScene` and `InteractiveSceneCfg`:

```text
InteractiveSceneCfg.num_envs = 64
InteractiveSceneCfg.env_spacing = enough distance between copies
robot prim path = "{ENV_REGEX_NS}/TurboPi"
camera prim path = "{ENV_REGEX_NS}/TurboPi/camera_link/RobotCamera"
```

`{ENV_REGEX_NS}` is the placeholder Isaac Lab uses so every clone gets its own namespace, for example:

```text
/World/envs/env_0/TurboPi
/World/envs/env_1/TurboPi
/World/envs/env_2/TurboPi
...
```

The teacher logic should also be vectorized. Instead of looping through robots and calling one controller at a time, the collector should hold batched tensors:

```text
positions:      [num_envs, 2]
yaws:           [num_envs]
task_ids:       [num_envs]
segment_ids:    [num_envs]
lap_progress:   [num_envs]
commands:       [num_envs, 3]
images:         [num_envs, H, W, 3]
```

Then each control step does:

1. Read all robot poses as one batch.
2. Compute nearest route segment and lookahead target for all environments.
3. Compute all velocity commands as one batch.
4. Save one frame per active environment.
5. Step the simulator.
6. Mark finished environments.
7. Reset only the finished environments.

This is why vectorized collection can be much faster than sequential collection.

#### Why Not Just Launch Many Processes?

Launching many independent Isaac processes is crude parallelism. It can help a little, but it is inefficient:

- Each process loads Isaac Sim separately.
- Each process loads the robot asset separately.
- Each process owns its own renderer and camera pipeline.
- Startup time is repeated.
- GPU memory is duplicated.
- Too many processes can fight each other and become slower than one well-batched process.

Vectorized Isaac Lab collection avoids most of that overhead by keeping all environments inside one application.

#### What Makes Vectorized Collection Harder?

The main complication is that every visual object the robot camera should see must be cloned into each environment. For the figure-8 task, that means the road geometry, start shelf, terrain pieces, TurboPi, and robot camera need to live under the environment namespace. The route math must use local environment coordinates, while Isaac world poses include each environment's origin offset.

Another bottleneck is camera rendering. Physics and kinematic teacher control are cheap, but rendering 64 RGB cameras every control step can be expensive. This is why the vectorized collector should start with lower image resolution, such as `64x48` or `96x72`, then increase after measuring throughput.

The existing `scripts/record_turbopi_square_vec.py` is the local example of this pattern for a square-loop task. The figure-8 vectorized collector should follow the same architecture, adapted to the figure-8 route and ACT language dataset format.

The intended design is:

- One Isaac process.
- `N` cloned figure-8 environments inside one `InteractiveScene`.
- One TurboPi per environment.
- One robot-facing camera per environment.
- Half the environments collect `go_left`; half collect `go_right`.
- The teacher, route progress, resets, and action computation run batched over all environments.
- When one environment finishes an episode, only that environment resets; the rest keep collecting.

The existing process-level collector does not do this yet. It is standalone-style and handles one robot per Isaac process. Launching many copies of that script wastes GPU memory and startup time. The vectorized collector is the correct Isaac Lab implementation for collecting a large balanced dataset in roughly two minutes.

Expected scale:

- `num_envs=32`: about 32 episodes are collected concurrently.
- `num_envs=64`: about 64 episodes are collected concurrently if GPU memory allows the cameras.
- With three-loop episodes at about 50 seconds of simulated driving each, a vectorized 64-env collector can collect around 100 accepted episodes in roughly a few episode durations plus startup, instead of waiting for 100 episodes sequentially.

Rendering is the main bottleneck. Physics is cheap because the teacher uses kinematic root updates; the expensive part is generating many camera images. For the fastest run, use lower camera resolution such as `64x48` or `96x72` first, then increase to `128x128` once throughput is confirmed.

Implemented vectorized command:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_figure8_act_vec.py \
  --headless \
  --num_envs 64 \
  --num_episodes 64 \
  --laps 3 \
  --image_width 96 \
  --image_height 72 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8_vec_64 \
  --session_name figure8_vec_64 \
  --max_episode_time 90
```

Observed result from the first 64-episode vectorized run:

```text
output: /workspace/turbopi_isaac/data/act_figure8_vec_64/figure8_vec_64
episodes: 64
left intent: 32
right intent: 32
failures: 0
total frames: 23,474
mean frames per episode: 366.8
image size: 96x72
laps per episode: 3
dataset size on disk: about 12 MB
```

For larger training data, consider disabling per-episode MP4 writing or storing image chunks directly. MP4 encoding can become a CPU bottleneck when many environments finish at the same time. The current implementation still writes `video.mp4` per episode because the existing ACT dataset format stores robot-camera images there.

## Single-Worker Commands

Left-only:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_mountain_act.py \
  --headless \
  --map figure8 \
  --task go_left \
  --laps 3 \
  --num_episodes 50 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8 \
  --session_name figure8_left_debug \
  --no_rollers
```

Right-only:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_mountain_act.py \
  --headless \
  --map figure8 \
  --task go_right \
  --laps 3 \
  --num_episodes 50 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8 \
  --session_name figure8_right_debug \
  --no_rollers
```

Balanced mixed collection in one process:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_mountain_act.py \
  --headless \
  --map figure8 \
  --task mix \
  --laps 3 \
  --num_episodes 100 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8 \
  --session_name figure8_mix_debug \
  --no_rollers
```

## Why This Is Fast

- Headless Isaac avoids viewer overhead.
- `--no_rollers` skips procedural mecanum roller generation; the policy camera does not need those details.
- The route is kinematically followed by the teacher, so collection is stable and does not waste many failed attempts.
- Left and right intents can be collected in parallel workers.
- Each episode contains three loops, so simulator startup cost is amortized over more useful frames.
- Small domain randomization is applied per episode instead of restarting or rebuilding the scene for every variation.

For the two-minute target, the important additional speedup is vectorization: one simulator step advances all cloned environments together.

## Dataset Layout

Each session contains:

```text
task_mapping.json
tasks.json
session_info.json
episode_00000/
  video.mp4
  data.parquet
  episode_info.json
episode_00001/
  ...
```

`data.parquet` contains one row per control step. The main columns are:

- `state`
- `action`
- `command`
- `track_error`
- `route_progress`
- `task`
- `task_index`
- `instruction`

The RGB frames are stored in `video.mp4`.

## Diverse Expert Dataset Audit

Before retraining, a new diverse expert dataset was collected and audited:

```text
dataset root: /workspace/turbopi_isaac/data/act_figure8_diverse_128
train session: figure8_diverse_train_64
validation session: figure8_diverse_val_64
accepted episodes: 128
left intent episodes: 64
right intent episodes: 64
laps per episode: 3
control rate: 10 Hz
camera source: robot_forward
stored camera resolution: 96x72
```

The collector now stores the actual expert pose at every saved frame:

```text
pose_x
pose_y
pose_yaw
```

These columns are for auditing and plotting. The ACT trainer still reads the same training fields as before:

```text
video.mp4
data.parquet["action"]
data.parquet["task"]
data.parquet["task_index"]
```

The new diversity sources are:

- Per-episode target-speed jitter.
- Start position jitter.
- Start yaw jitter.
- Bounded lateral expert offset within the road.
- Gentle sinusoidal lateral variation within the road.
- Small action noise.

The collector rejects attempts whose centerline error exceeds the safety gate:

```text
--off_track_abort_distance 0.22
```

Rejected attempts are not written as training episodes. In the collected dataset:

```text
train session: 64 accepted, 19 rejected
validation session: 64 accepted, 6 rejected
```

The final accepted dataset audit is:

```text
go_left:
  episodes: 64
  frames: 24,156
  mean duration: 37.7437 s
  mean track error: 0.0384 m
  max track error: 0.1555 m
  minimum final progress: 1.0000

go_right:
  episodes: 64
  frames: 24,021
  mean duration: 37.5328 s
  mean track error: 0.0379 m
  max track error: 0.1806 m
  minimum final progress: 1.0000
```

The pose-based audit plots are:

```text
/workspace/turbopi_isaac/outputs/figure8_diverse_expert_path_audit/figure8_expert_paths_overlay.png
/workspace/turbopi_isaac/outputs/figure8_diverse_expert_path_audit/figure8_expert_paths_on_topdown_map.png
/workspace/turbopi_isaac/outputs/figure8_diverse_expert_path_audit/figure8_expert_paths_summary.csv
```

The same audit artifacts are also versioned in Git for later slide/report generation:

```text
docs/experiment_artifacts/figure8_diverse_experts/README.md
docs/experiment_artifacts/figure8_diverse_experts/figure8_expert_paths_on_topdown_map.png
docs/experiment_artifacts/figure8_diverse_experts/figure8_expert_paths_overlay.png
docs/experiment_artifacts/figure8_diverse_experts/figure8_expert_progress_projection.png
docs/experiment_artifacts/figure8_diverse_experts/figure8_expert_paths_summary.csv
```

Important note: an earlier command-integration plot from the old dataset did not line up with the route because it reconstructed pose by integrating only the saved frame commands. Segment-transition control steps were applied in simulation but were not saved as frames, so command-only reconstruction was not a reliable pose audit. The new diverse dataset writes actual pose columns and the final audit plots use those saved poses directly.

## ACT + CVAE + Language Intent Training

The training script is:

```text
train_turbopi_mountain_act.py
```

It calls:

```text
act_policy.train.main()
```

The current figure-8 training command is:

```bash
cd /workspace/turbopi_isaac
/workspace/isaaclab/_isaac_sim/python.sh train_turbopi_mountain_act.py \
  --episodes-dir /workspace/turbopi_isaac/data/act_figure8_vec_64/figure8_vec_64 \
  --run-dir runs/figure8_act_cvae \
  --epochs 40 \
  --batch-size 128 \
  --num-workers 0 \
  --device cuda \
  --no-progress
```

`--num-workers 0` is intentional in this container. The shared-memory mount is small, and PyTorch video-decoding workers can otherwise fail with shared-memory bus errors.

### Dataset Used For This Training Run

The first vectorized figure-8 dataset has:

```text
dataset path: /workspace/turbopi_isaac/data/act_figure8_vec_64/figure8_vec_64
episodes: 64
left intent episodes: 32
right intent episodes: 32
failures during collection: 0
total frames: 23,474
mean frames per episode: 366.8
camera source: robot_forward
stored camera resolution: 96x72
model input resolution: 128x128
control rate: 10 Hz
laps per episode: 3
```

Each training sample is built from one timestep:

- Input image: robot-forward RGB frame from `video.mp4`.
- Intent: task id from `data.parquet`, mapped from `go_left` or `go_right`.
- Target: future action chunk from `data.parquet["action"]`.

The action chunk length is `5`, so the model predicts five future actions from the current image and intent. At `10 Hz`, that chunk covers about `0.5 s` of future control.

### Architecture

The model is `LanguageConditionedACTCVAE` in `act_policy/model.py`.

High-level structure:

```text
robot camera image
  -> CNN image encoder
  -> 64 spatial tokens, each 64-dim

language intent id
  -> learned embedding
  -> 1 language token, 64-dim

future action chunk during training
  -> CVAE encoder
  -> latent z distribution
  -> sampled z token, 64-dim

spatial tokens + language token + z token
  -> transformer encoder
  -> transformer decoder with 5 learned action queries
  -> 5 predicted actions
```

The language conditioning is currently task-id based:

```text
go_left  -> task index 0
go_right -> task index 1
```

Those ids are embedded into learned 64-dimensional vectors. This is sufficient for the current two-intent setup. If we later want open-vocabulary natural language, the language-token block can be replaced with a text encoder while keeping the ACT/CVAE action decoder structure.

### CVAE Loss

During training, the model sees the ground-truth future action chunk. The CVAE encoder maps:

```text
ground-truth action chunk + language token -> latent distribution
```

It predicts:

```text
mu, logvar
```

Then the model samples:

```text
z = mu + eps * std
```

The total loss is:

```text
SmoothL1(predicted_action_chunk, target_action_chunk)
+ kl_weight * KL(q(z | action, language) || N(0, I))
```

Current training uses:

```text
kl_weight = 0.01
optimizer = AdamW
learning rate = 3e-4
weight decay = 1e-4
epochs = 40
batch size = 128
```

At inference time, the future action chunk is not available. The runtime uses `z = 0`, plus the image and language intent, to predict the next action chunk.

### Checkpoints

Training writes:

```text
runs/figure8_act_cvae/run_<timestamp>/
  checkpoints/
    last.pt
    best.pt
  training_summary.json
```

`best.pt` is selected by validation loss if a validation split exists; otherwise it follows the training loss. Because the current 64-episode dataset is one session, the current loader treats all episodes as training records and has no separate validation session. For better validation, future collection should write at least two sessions or the dataset splitter should support episode-level splitting within one session.

### Current Training Result

The completed training run is:

```text
runs/figure8_act_cvae/run_20260508_070640
```

It finished all `40` epochs. The final logged loss was:

```text
epoch 040
train_loss = 0.0132
val_loss = nan
best_epoch = 40
```

The `val_loss` is `nan` only because there was no validation split for this one-session dataset. The checkpoint used for inference is:

```text
runs/figure8_act_cvae/run_20260508_070640/checkpoints/best.pt
```

## Inference Video Rendering

The inference script is:

```text
scripts/drive_turbopi_mountain_act.py
```

It loads the trained checkpoint, builds the default `figure8` map, attaches the policy camera for the model input, and can attach separate high-resolution cameras for rendering. For the requested outputs, only the chase camera was recorded by passing:

```text
--video_views chase
```

This keeps the render focused on the deliverable videos instead of also writing robot and isometric views. The script default remains unchanged: without `--video_views`, it records `robot`, `chase`, and `isometric`.

The video runs used kinematic control:

```text
--control_mode kinematic
```

That matches the vectorized collector, where the expert trajectory directly integrates commanded planar velocity. The model still receives the onboard policy camera image and language intent; the chase camera is only for human-readable video output.

### Left-Intent Video Command

```bash
cd /workspace/turbopi_isaac
TERM=xterm /workspace/isaaclab/isaaclab.sh -p /workspace/turbopi_isaac/scripts/drive_turbopi_mountain_act.py \
  --headless \
  --checkpoint /workspace/turbopi_isaac/runs/figure8_act_cvae/run_20260508_070640/checkpoints/best.pt \
  --task go_left \
  --view chase \
  --duration 30 \
  --control_mode kinematic \
  --no_rollers \
  --video_output_dir /workspace/turbopi_isaac/inference_videos/figure8_act_1080 \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 10 \
  --video_views chase
```

### Right-Intent Video Command

```bash
cd /workspace/turbopi_isaac
TERM=xterm /workspace/isaaclab/isaaclab.sh -p /workspace/turbopi_isaac/scripts/drive_turbopi_mountain_act.py \
  --headless \
  --checkpoint /workspace/turbopi_isaac/runs/figure8_act_cvae/run_20260508_070640/checkpoints/best.pt \
  --task go_right \
  --view chase \
  --duration 30 \
  --control_mode kinematic \
  --no_rollers \
  --video_output_dir /workspace/turbopi_isaac/inference_videos/figure8_act_1080 \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 10 \
  --video_views chase
```

The videos are:

```text
/workspace/turbopi_isaac/inference_videos/figure8_act_1080/mountain_act_inference_go_left_chase_1920x1080.mp4
/workspace/turbopi_isaac/inference_videos/figure8_act_1080/mountain_act_inference_go_right_chase_1920x1080.mp4
```

Both videos were verified as:

```text
resolution: 1920x1080
frames: 300
fps: 10
duration: 30.00 s
```

`--video_fps 10` is deliberate because the inference loop writes one frame per `10 Hz` control step. Using `10 fps` makes the MP4 duration match the 30 seconds of simulated driving. If the control loop is later changed to write frames at render rate, this value should be revisited.

## Update Log

- Added the `figure8` alternate map while preserving `original`.
- Removed guard rails and end caps from `figure8`.
- Added `--laps`, defaulting to three loops per episode.
- Added per-episode speed, start pose, and camera jitter.
- Added `scripts/collect_figure8_act_parallel.sh` for balanced parallel left/right collection.
- Added a two-minute target design note explaining why the fast path should be a vectorized `InteractiveScene` collector, not more standalone Isaac processes.
- Added `scripts/record_turbopi_figure8_act_vec.py` and collected a 64-episode vectorized dataset with 32 left and 32 right episodes.
- Completed ACT + CVAE + language-intent training on the 64-episode vectorized figure-8 dataset.
- Rendered 30-second 1920x1080 chase-camera inference videos for `go_left` and `go_right`.
- Added pose columns to the vectorized collector and collected a 128-episode diverse expert dataset with pose-based top-down audit plots.
- Versioned the diverse expert audit plots and summary CSV under `docs/experiment_artifacts/figure8_diverse_experts/` for later slides.

## Next Sections To Add

When implemented, extend this document with:

- Recommended figures for reports or slides.
