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

## Update Log

- Added the `figure8` alternate map while preserving `original`.
- Removed guard rails and end caps from `figure8`.
- Added `--laps`, defaulting to three loops per episode.
- Added per-episode speed, start pose, and camera jitter.
- Added `scripts/collect_figure8_act_parallel.sh` for balanced parallel left/right collection.
- Added a two-minute target design note explaining why the fast path should be a vectorized `InteractiveScene` collector, not more standalone Isaac processes.
- Added `scripts/record_turbopi_figure8_act_vec.py` and collected a 64-episode vectorized dataset with 32 left and 32 right episodes.

## Next Sections To Add

When implemented, extend this document with:

- Training commands and dataset selection.
- Model checkpoint layout.
- Inference commands on the figure-8 map.
- How to export and download inference videos.
- Recommended figures for reports or slides.
