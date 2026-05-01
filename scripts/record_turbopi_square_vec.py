"""Vectorized TurboPi square-loop recorder.

Spawns ``--num_envs`` cloned arenas inside one Isaac Sim process, runs the
deterministic kinematic teacher in all of them simultaneously, and writes each
completed episode to disk in the same on-disk format as
``record_turbopi_square_simple.py`` so the dataset/training loaders need no
changes.

Per-env independence:
- Each env has its own square arena cloned via ``InteractiveScene``.
- Each env has its own start direction, randomized start phase + jitter.
- An env that finishes (success or failure) is reset in-place with a fresh
  start while the others keep running. Collection ends when the total number
  of accepted episodes reaches ``--num_episodes``.

Mecanum rollers are skipped because kinematic teleport does not need them, and
omitting them avoids InteractiveScene's spawn-then-clone-without-injection
constraint for stage-level prim additions.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cnn_square_vec"

parser = argparse.ArgumentParser(description="Vectorized TurboPi square-loop recorder.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel cloned arenas in one sim.")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--session_name", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="turbopi_square_vec_cnn")
parser.add_argument("--num_episodes", type=int, default=50, help="Total accepted episodes across all envs.")
parser.add_argument("--mix_directions", action="store_true", help="Alternate ccw/cw across envs.")
parser.add_argument("--randomize_start", action="store_true", help="Randomize start phase + jitter per episode.")
parser.add_argument("--start_lateral_jitter", type=float, default=0.03)
parser.add_argument("--start_yaw_jitter_deg", type=float, default=10.0)
parser.add_argument("--action_noise_std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--image_width", type=int, default=64)
parser.add_argument("--image_height", type=int, default=48)
parser.add_argument("--square_half_extent", type=float, default=0.45)
parser.add_argument("--floor_half_extent", type=float, default=0.7)
parser.add_argument("--wall_height", type=float, default=0.55)
parser.add_argument("--wall_thickness", type=float, default=0.04)
parser.add_argument("--tape_width", type=float, default=0.08)
parser.add_argument("--target_speed", type=float, default=0.30)
parser.add_argument("--min_forward_speed", type=float, default=0.08)
parser.add_argument("--square_max_wz", type=float, default=1.10)
parser.add_argument("--square_heading_gain", type=float, default=2.7)
parser.add_argument("--square_cross_track_gain", type=float, default=1.2)
parser.add_argument("--lookahead_distance", type=float, default=0.12)
parser.add_argument("--corner_slowdown_distance", type=float, default=0.24)
parser.add_argument("--heading_slowdown_angle", type=float, default=1.20)
parser.add_argument("--lap_completion_threshold", type=float, default=0.97)
parser.add_argument("--start_phase_tolerance", type=float, default=0.06)
parser.add_argument("--start_yaw_tolerance", type=float, default=0.35)
parser.add_argument("--min_square_distance_ratio", type=float, default=0.85)
parser.add_argument("--off_track_abort_distance", type=float, default=0.30)
parser.add_argument("--max_episode_time", type=float, default=30.0)
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--camera_warmup_steps", type=int, default=6)
parser.add_argument("--min_image_std", type=float, default=8.0)
parser.add_argument("--no_onboard_video", action="store_true", help="Skip writing per-episode video.mp4.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz

from cnn_dataset import CNNSessionWriter, EpisodeFrame, EpisodeResult
from common import (
    CAMERA_LINK_TO_SENSOR_POS,
    CAMERA_LINK_TO_SENSOR_ROT,
    build_turbopi_cfg,
)
from square_loop import (
    SquareTrackSceneCfg,
    compute_square_track_frame,
    direction_sign,
    phase_to_segment_and_progress,
    segment_tangent_clockwise,
    square_phase_to_point_and_tangent,
)


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------


def _scene_cfg() -> SquareTrackSceneCfg:
    return SquareTrackSceneCfg(
        square_half_extent=args_cli.square_half_extent,
        floor_half_extent=args_cli.floor_half_extent,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )


def _build_scene_cfg_class(track: SquareTrackSceneCfg, num_envs: int):
    """Build the InteractiveSceneCfg dynamically using the requested geometry."""

    floor_size = 2.0 * track.floor_half_extent
    tape_span = 2.0 * track.square_half_extent
    wall_half = track.floor_half_extent + 0.5 * track.wall_thickness
    wall_z = 0.5 * track.wall_height
    env_spacing = float(2.0 * track.floor_half_extent + 0.6)

    floor_cfg = sim_utils.CuboidCfg(
        size=(floor_size, floor_size, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=track.floor_color, roughness=0.95),
    )
    tape_h_cfg = sim_utils.CuboidCfg(
        size=(tape_span + track.tape_width, track.tape_width, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=track.tape_color, roughness=0.65),
    )
    tape_v_cfg = sim_utils.CuboidCfg(
        size=(track.tape_width, tape_span + track.tape_width, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=track.tape_color, roughness=0.65),
    )
    wall_x_cfg = sim_utils.CuboidCfg(
        size=(floor_size + track.wall_thickness, track.wall_thickness, track.wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=track.wall_color, roughness=0.90),
    )
    wall_y_cfg = sim_utils.CuboidCfg(
        size=(track.wall_thickness, floor_size + track.wall_thickness, track.wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=track.wall_color, roughness=0.90),
    )

    robot_cfg = build_turbopi_cfg(asset_usd=args_cli.asset_usd, prim_path="{ENV_REGEX_NS}/TurboPi", add_rollers=False)

    camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TurboPi/camera_link/RobotCamera",
        update_period=0.0,
        height=args_cli.image_height,
        width=args_cli.image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.5,
            focus_distance=400.0,
            horizontal_aperture=10.0,
            vertical_aperture=7.5,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(CAMERA_LINK_TO_SENSOR_POS),
            rot=tuple(CAMERA_LINK_TO_SENSOR_ROT),
            convention="opengl",
        ),
    )

    _num_envs_default = int(num_envs)
    _env_spacing_default = float(env_spacing)

    @configclass
    class _Cfg(InteractiveSceneCfg):
        num_envs: int = _num_envs_default
        env_spacing: float = _env_spacing_default

        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            ),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.95, 0.95, 0.95)),
        )
        sun = AssetBaseCfg(
            prim_path="/World/SunLight",
            spawn=sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0), angle=0.5),
        )

        floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            spawn=floor_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, track.floor_z)),
        )
        tape_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TapeTop",
            spawn=tape_h_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, track.square_half_extent, track.tape_z)),
        )
        tape_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TapeBottom",
            spawn=tape_h_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -track.square_half_extent, track.tape_z)),
        )
        tape_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TapeLeft",
            spawn=tape_v_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-track.square_half_extent, 0.0, track.tape_z)),
        )
        tape_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TapeRight",
            spawn=tape_v_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(track.square_half_extent, 0.0, track.tape_z)),
        )
        wall_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/WallTop",
            spawn=wall_x_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, wall_half, wall_z)),
        )
        wall_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/WallBottom",
            spawn=wall_x_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -wall_half, wall_z)),
        )
        wall_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/WallLeft",
            spawn=wall_y_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-wall_half, 0.0, wall_z)),
        )
        wall_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/WallRight",
            spawn=wall_y_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(wall_half, 0.0, wall_z)),
        )

        robot = robot_cfg
        camera = camera_cfg

    return _Cfg


# ---------------------------------------------------------------------------
# Per-env state
# ---------------------------------------------------------------------------


@dataclass
class EnvState:
    direction: str = "counterclockwise"
    start_phase: float = 0.0
    start_yaw: float = 0.0
    previous_phase: float = 0.0
    lap_progress: float = 0.0
    commanded_forward_distance: float = 0.0
    elapsed_steps: int = 0
    frames: list[EpisodeFrame] = field(default_factory=list)
    track_errors: list[float] = field(default_factory=list)
    image_stds: list[float] = field(default_factory=list)
    body_speeds: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    finished: bool = False
    success: bool = False
    terminal_reason: str = "active"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_phase_error(a: float, b: float) -> float:
    delta = (a - b + 0.5) % 1.0 - 0.5
    return abs(delta)


def signed_phase_delta(current: float, previous: float, direction: str) -> float:
    delta = current - previous
    if delta < -0.5:
        delta += 1.0
    if delta > 0.5:
        delta -= 1.0
    sign = 1.0 if direction == "clockwise" else -1.0
    return max(0.0, sign * delta)


def random_start_pose(
    track: SquareTrackSceneCfg,
    direction: str,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Return (x_local, y_local, yaw, start_phase) for a randomized start pose."""
    sign = direction_sign(direction)
    if args_cli.randomize_start:
        phase = float(rng.uniform(0.0, 1.0))
    else:
        phase = 0.0  # default: middle of left edge
    pt, tg = square_phase_to_point_and_tangent(
        torch.tensor([phase], dtype=torch.float32), track.square_half_extent
    )
    px, py = float(pt[0, 0].item()), float(pt[0, 1].item())
    tx = float(tg[0, 0].item()) * sign
    ty = float(tg[0, 1].item()) * sign
    nx, ny = -ty, tx
    if (px + nx) ** 2 + (py + ny) ** 2 > px ** 2 + py ** 2:
        nx, ny = -nx, -ny
    if args_cli.randomize_start:
        lat = float(rng.uniform(-args_cli.start_lateral_jitter, args_cli.start_lateral_jitter))
        yaw_jitter = math.radians(float(rng.uniform(-args_cli.start_yaw_jitter_deg, args_cli.start_yaw_jitter_deg)))
    else:
        lat = 0.0
        yaw_jitter = 0.0
    x = px + nx * lat
    y = py + ny * lat
    yaw = wrap_to_pi(math.atan2(ty, tx) + yaw_jitter)
    return x, y, yaw, phase


def reset_envs(
    scene: InteractiveScene,
    states: list[EnvState],
    env_ids: list[int],
    track: SquareTrackSceneCfg,
    rng: np.random.Generator,
    *,
    next_episode_index: list[int],
    accepted_so_far: int,
):
    """Re-seed the given envs with a fresh start pose and clear their buffers."""
    if not env_ids:
        return
    robot = scene["robot"]
    device = robot.device
    env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    root_state = robot.data.default_root_state[env_ids_t].clone()
    yaws = torch.zeros(len(env_ids), dtype=torch.float32, device=device)
    for i, env_idx in enumerate(env_ids):
        if args_cli.mix_directions:
            states[env_idx].direction = (
                "counterclockwise" if (next_episode_index[0] + i) % 2 == 0 else "clockwise"
            )
        x, y, yaw, phase = random_start_pose(track, states[env_idx].direction, rng)
        states[env_idx] = EnvState(direction=states[env_idx].direction)
        states[env_idx].start_phase = phase
        states[env_idx].previous_phase = phase
        states[env_idx].start_yaw = yaw
        root_state[i, 0] = x
        root_state[i, 1] = y
        root_state[i, 2] = track.start_height
        yaws[i] = yaw
    next_episode_index[0] += len(env_ids)
    zeros = torch.zeros_like(yaws)
    root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaws)
    root_state[:, :3] += scene.env_origins[env_ids_t]
    root_state[:, 7:] = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids_t)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids_t)
    joint_pos = robot.data.default_joint_pos[env_ids_t].clone()
    joint_vel = robot.data.default_joint_vel[env_ids_t].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)


# ---------------------------------------------------------------------------
# Batched teacher
# ---------------------------------------------------------------------------


def compute_batched_command(
    local_pos_xy: torch.Tensor,
    quat_w: torch.Tensor,
    directions_sign: torch.Tensor,
    track: SquareTrackSceneCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized version of the kinematic full-square teacher."""
    nearest_xy, _, track_err, phase = compute_square_track_frame(local_pos_xy, track.square_half_extent)
    segment, segment_progress = phase_to_segment_and_progress(phase)

    cw_tangent = segment_tangent_clockwise(segment)
    current_tangent_xy = cw_tangent * directions_sign.unsqueeze(-1)
    direction_step = torch.where(directions_sign > 0, torch.ones_like(segment), -torch.ones_like(segment))
    next_segment = torch.remainder(segment + direction_step, 4)
    next_tangent_xy = segment_tangent_clockwise(next_segment) * directions_sign.unsqueeze(-1)

    seg_len = 2.0 * track.square_half_extent
    distance_to_corner = torch.where(
        directions_sign > 0,
        (1.0 - segment_progress) * seg_len,
        segment_progress * seg_len,
    )
    blend_dist = max(args_cli.lookahead_distance, args_cli.corner_slowdown_distance)
    corner_blend = torch.clamp(1.0 - distance_to_corner / max(blend_dist, 1e-4), min=0.0, max=1.0)
    blended = (1.0 - corner_blend).unsqueeze(-1) * current_tangent_xy + corner_blend.unsqueeze(-1) * next_tangent_xy
    bn = torch.linalg.norm(blended, dim=-1, keepdim=True)
    target_tangent = torch.where(bn > 1e-6, blended / bn, current_tangent_xy)

    lookahead_phase = phase + directions_sign * (
        args_cli.lookahead_distance / max(8.0 * track.square_half_extent, 1e-6)
    )
    target_pt, _ = square_phase_to_point_and_tangent(lookahead_phase, track.square_half_extent)

    n = local_pos_xy.shape[0]
    target_w = torch.zeros((n, 3), dtype=torch.float32, device=local_pos_xy.device)
    target_w[:, :2] = target_pt
    pos_w_3 = torch.zeros((n, 3), dtype=torch.float32, device=local_pos_xy.device)
    pos_w_3[:, :2] = local_pos_xy
    lookahead_b = quat_apply_inverse(quat_w, target_w - pos_w_3)[:, :2]
    nearest_w = torch.zeros_like(target_w)
    nearest_w[:, :2] = nearest_xy
    nearest_b = quat_apply_inverse(quat_w, nearest_w - pos_w_3)[:, :2]

    _, _, yaw = euler_xyz_from_quat(quat_w)
    path_yaw = torch.atan2(target_tangent[:, 1], target_tangent[:, 0])
    yaw_err = torch.atan2(torch.sin(path_yaw - yaw), torch.cos(path_yaw - yaw))
    point_heading_err = torch.atan2(lookahead_b[:, 1], torch.clamp(lookahead_b[:, 0], min=0.04))

    corner_speed_scale = torch.clamp(
        distance_to_corner / max(args_cli.corner_slowdown_distance, 1e-4),
        min=0.55,
        max=1.0,
    )
    heading_speed_scale = torch.clamp(
        1.0 - torch.abs(yaw_err) / max(args_cli.heading_slowdown_angle, 1e-4),
        min=0.45,
        max=1.0,
    )
    speed = args_cli.target_speed * torch.minimum(corner_speed_scale, heading_speed_scale)
    vx = torch.clamp(
        speed * torch.clamp(torch.cos(yaw_err), min=0.35, max=1.0),
        min=args_cli.min_forward_speed,
        max=args_cli.target_speed,
    )
    desired_wz = (
        args_cli.square_heading_gain * yaw_err
        + args_cli.square_cross_track_gain * point_heading_err
        + 0.35 * nearest_b[:, 1]
    )
    wz = torch.clamp(desired_wz, min=-args_cli.square_max_wz, max=args_cli.square_max_wz)
    command = torch.stack((vx, torch.zeros_like(vx), wz), dim=-1)
    return command, track_err, phase, yaw_err


def integrate_kinematic(
    pos_xy: torch.Tensor,
    yaw: torch.Tensor,
    command: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vx = command[:, 0]
    vy = command[:, 1]
    wz = command[:, 2]
    yaw_mid = yaw + 0.5 * wz * dt
    new_x = pos_xy[:, 0] + (vx * torch.cos(yaw_mid) - vy * torch.sin(yaw_mid)) * dt
    new_y = pos_xy[:, 1] + (vx * torch.sin(yaw_mid) + vy * torch.cos(yaw_mid)) * dt
    new_yaw = (yaw + wz * dt + math.pi) % (2.0 * math.pi) - math.pi
    return torch.stack((new_x, new_y), dim=-1), new_yaw


# ---------------------------------------------------------------------------
# Stop-on-signal
# ---------------------------------------------------------------------------


class StopFlag:
    def __init__(self):
        self.requested = False

    def request(self, signum, _):
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_vec_%Y%m%d_%H%M%S")


def make_episode_result(state: EnvState, control_dt: float, route_length: float) -> EpisodeResult:
    frames = state.frames
    track_errors = state.track_errors
    image_stds = state.image_stds
    body_speeds = state.body_speeds
    action_history = state.actions
    duration_s = len(frames) * control_dt
    mean_track_error = float(np.mean(track_errors)) if track_errors else float("inf")
    p90_track_error = float(np.quantile(track_errors, 0.9)) if track_errors else float("inf")
    max_track_error = float(np.max(track_errors)) if track_errors else float("inf")
    over_010 = float(np.mean(np.asarray(track_errors) > 0.10)) if track_errors else 1.0
    over_015 = float(np.mean(np.asarray(track_errors) > 0.15)) if track_errors else 1.0
    mean_image_std = float(np.mean(image_stds)) if image_stds else 0.0
    min_image_std = float(np.min(image_stds)) if image_stds else 0.0
    mean_speed = float(np.mean(body_speeds)) if body_speeds else 0.0
    action_arr = np.asarray(action_history, dtype=np.float32) if action_history else np.zeros((0, 3), dtype=np.float32)
    mean_abs = np.mean(np.abs(action_arr), axis=0) if len(action_arr) > 0 else np.zeros(3, dtype=np.float32)
    final_progress = 1.0 if state.success else float(min(1.0, state.lap_progress))
    return EpisodeResult(
        direction=state.direction,
        task_name=state.direction,
        task_index=0 if state.direction == "clockwise" else 1,
        frames=frames,
        success=state.success,
        terminal_reason=state.terminal_reason,
        final_lap_progress=final_progress,
        mean_track_error=mean_track_error,
        p90_track_error=p90_track_error,
        max_track_error=max_track_error,
        frames_over_010_ratio=over_010,
        frames_over_015_ratio=over_015,
        mean_image_std=mean_image_std,
        min_image_std=min_image_std,
        mean_abs_action_vx=float(mean_abs[0]),
        mean_abs_action_vy=float(mean_abs[1]),
        mean_abs_action_wz=float(mean_abs[2]),
        mean_action_vy_vx_ratio=float(mean_abs[1] / max(float(mean_abs[0]), 1e-6)),
        mean_speed=mean_speed,
        duration_s=duration_s,
    )


def main() -> None:
    track = _scene_cfg()
    SceneCfgClass = _build_scene_cfg_class(track, args_cli.num_envs)
    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    physics_dt = min(control_dt, max(physics_dt, 1.0 / 30.0))
    substeps = steps_per_control(physics_dt, args_cli.control_hz)

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, render_interval=substeps, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(SceneCfgClass())
    sim.reset()

    robot = scene["robot"]
    camera = scene["camera"]
    device = robot.device
    n_envs = args_cli.num_envs
    states = [EnvState() for _ in range(n_envs)]
    rng = np.random.default_rng(int(args_cli.seed))
    next_episode_index = [0]

    # Initial reset for all envs.
    reset_envs(scene, states, list(range(n_envs)), track, rng,
               next_episode_index=next_episode_index, accepted_so_far=0)
    scene.write_data_to_sim()
    for _ in range(args_cli.settle_steps):
        sim.step()
        scene.update(physics_dt)
    for _ in range(args_cli.camera_warmup_steps):
        sim.step()
        scene.update(physics_dt)
    camera.update(physics_dt)

    writer = CNNSessionWriter(
        output_root=Path(args_cli.output_dir),
        session_name=build_session_name(),
        dataset_name=args_cli.dataset_name,
        fps=args_cli.control_hz,
        image_width=args_cli.image_width,
        image_height=args_cli.image_height,
        episode_time_s=args_cli.max_episode_time,
        control_hz=args_cli.control_hz,
        physics_dt=physics_dt,
        tasks=("clockwise", "counterclockwise") if args_cli.mix_directions else ("counterclockwise",),
        track_layout="square_full_loop",
        episode_definition="one_autonomous_full_square_lap",
    )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print()
    print("=" * 60)
    print("  TurboPi Vectorized Recorder")
    print("=" * 60)
    print(f"  Output session : {writer.session_dir}")
    print(f"  N envs         : {n_envs}")
    print(f"  Episode target : {args_cli.num_episodes}")
    print(f"  Control rate   : {args_cli.control_hz:.1f} Hz")
    print(f"  Sim dt         : {physics_dt:.4f} s ({substeps} substeps/control)")
    print(f"  Image          : {args_cli.image_width}x{args_cli.image_height}")
    print(f"  Target speed   : {args_cli.target_speed:.2f} m/s")
    print(f"  Randomize start: {args_cli.randomize_start}, action_noise={args_cli.action_noise_std}")
    print()

    accepted = 0
    failed = 0
    max_steps_per_episode = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))
    perimeter = 8.0 * track.square_half_extent
    sign_table = {"counterclockwise": -1.0, "clockwise": 1.0}
    max_command = np.array([0.45, 0.35, 2.0], dtype=np.float32)

    try:
        while accepted < args_cli.num_episodes and simulation_app.is_running() and not stop_flag.requested:
            # Build per-env direction sign tensor.
            dirs_sign = torch.tensor(
                [sign_table[states[i].direction] for i in range(n_envs)],
                dtype=torch.float32, device=device,
            )

            # Get local positions (subtract env origins).
            pos_w = robot.data.root_pos_w[:, :2]
            local_pos = pos_w - scene.env_origins[:, :2]
            quat_w = robot.data.root_quat_w
            command, track_err, phase, yaw_err = compute_batched_command(local_pos, quat_w, dirs_sign, track)
            if args_cli.action_noise_std > 0.0:
                noise = torch.from_numpy(
                    rng.normal(0.0, args_cli.action_noise_std, size=(n_envs, 3)).astype(np.float32)
                ).to(device) * torch.tensor(max_command, device=device)
                command = command + noise
                command[:, 0].clamp_(args_cli.min_forward_speed, args_cli.target_speed)
                command[:, 1].clamp_(-0.20, 0.20)
                command[:, 2].clamp_(-args_cli.square_max_wz, args_cli.square_max_wz)

            # Read current yaw from quat for kinematic integration.
            _, _, yaw = euler_xyz_from_quat(quat_w)

            # Build new poses by kinematic integration of one control step.
            new_xy_local, new_yaw = integrate_kinematic(local_pos, yaw, command, control_dt)

            # Capture frames BEFORE writing the new pose (frame corresponds to current obs + chosen action).
            rgb_batch = camera.data.output["rgb"]
            if rgb_batch is None or rgb_batch.numel() == 0:
                rgb_np = np.zeros((n_envs, args_cli.image_height, args_cli.image_width, 3), dtype=np.uint8)
            else:
                rgb = rgb_batch[..., :3]
                if rgb.dtype != torch.uint8:
                    rgb = torch.clamp(rgb, 0, 255).to(torch.uint8)
                rgb_np = rgb.detach().cpu().numpy()

            cmd_np = command.detach().cpu().numpy()
            track_err_np = track_err.detach().cpu().numpy()
            phase_np = phase.detach().cpu().numpy()

            for i in range(n_envs):
                state = states[i]
                if state.finished:
                    continue
                # update lap_progress
                state.lap_progress = min(
                    1.5, state.lap_progress + signed_phase_delta(float(phase_np[i]), state.previous_phase, state.direction)
                )
                state.previous_phase = float(phase_np[i])
                state.commanded_forward_distance += max(0.0, float(cmd_np[i, 0])) * control_dt

                if float(track_err_np[i]) > args_cli.off_track_abort_distance:
                    state.finished = True
                    state.success = False
                    state.terminal_reason = "off_track"
                    continue
                if state.elapsed_steps >= max_steps_per_episode:
                    state.finished = True
                    state.success = False
                    state.terminal_reason = "timeout"
                    continue

                # Success check: lap closed, near start phase + yaw, enough cmd distance
                cur_yaw = float(yaw[i].item())
                if (
                    state.lap_progress >= args_cli.lap_completion_threshold
                    and wrap_phase_error(float(phase_np[i]), state.start_phase) <= args_cli.start_phase_tolerance
                    and abs(wrap_to_pi(cur_yaw - state.start_yaw)) <= args_cli.start_yaw_tolerance
                    and state.commanded_forward_distance >= perimeter * args_cli.min_square_distance_ratio
                ):
                    state.finished = True
                    state.success = True
                    state.terminal_reason = "goal_reached"

                action_np = np.clip(cmd_np[i] / max_command, -1.0, 1.0).astype(np.float32)
                state.actions.append(action_np)
                state.frames.append(
                    EpisodeFrame(
                        image_rgb=rgb_np[i],
                        timestamp=float(state.elapsed_steps * control_dt),
                        state=state.prev_action.copy(),
                        action=action_np.copy(),
                        command=cmd_np[i].astype(np.float32),
                        body_velocity=cmd_np[i].astype(np.float32),
                        track_error=float(track_err_np[i]),
                        lap_progress=float(min(1.0, state.lap_progress)),
                    )
                )
                state.prev_action = action_np
                state.track_errors.append(float(track_err_np[i]))
                state.image_stds.append(float(np.asarray(rgb_np[i], dtype=np.float32).std()))
                state.body_speeds.append(abs(float(cmd_np[i, 0])))
                state.elapsed_steps += 1

            # Apply teleport: new world positions
            world_xy = new_xy_local + scene.env_origins[:, :2]
            root_pose = torch.zeros((n_envs, 7), dtype=torch.float32, device=device)
            root_pose[:, :2] = world_xy
            root_pose[:, 2] = track.start_height
            zeros = torch.zeros_like(new_yaw)
            root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, new_yaw)
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(torch.zeros((n_envs, 6), dtype=torch.float32, device=device))

            scene.write_data_to_sim()
            for _ in range(substeps):
                sim.step()
            scene.update(physics_dt)

            # Collect finished envs, save, and reset them.
            finished_ids = [i for i, s in enumerate(states) if s.finished]
            if finished_ids:
                to_reset = []
                for env_idx in finished_ids:
                    state = states[env_idx]
                    if state.success and state.frames and accepted < args_cli.num_episodes:
                        result = make_episode_result(state, control_dt, perimeter)
                        if args_cli.no_onboard_video:
                            # CNNSessionWriter always writes video; tolerate by overriding _write_video.
                            writer._write_video = (lambda *a, **k: None)  # type: ignore
                        episode_dir = writer.save_episode(accepted, result)
                        accepted += 1
                        print(
                            f"[INFO] env={env_idx} saved ep_{accepted - 1:05d} dir={state.direction} "
                            f"frames={len(state.frames)} mean_err={result.mean_track_error:.3f} -> {episode_dir}",
                            flush=True,
                        )
                    else:
                        if not state.success:
                            failed += 1
                            writer.record_failure()
                            print(
                                f"[WARN] env={env_idx} failed reason={state.terminal_reason} "
                                f"frames={len(state.frames)} lap={state.lap_progress:.2f}",
                                flush=True,
                            )
                    to_reset.append(env_idx)
                if accepted >= args_cli.num_episodes:
                    break
                reset_envs(scene, states, to_reset, track, rng,
                           next_episode_index=next_episode_index, accepted_so_far=accepted)
                scene.write_data_to_sim()
                for _ in range(args_cli.settle_steps):
                    sim.step()
                scene.update(physics_dt)
                camera.update(physics_dt)

    finally:
        print()
        print(f"[INFO] Session complete : {writer.session_dir}", flush=True)
        print(f"[INFO] Saved episodes   : {accepted}", flush=True)
        print(f"[INFO] Failed attempts  : {failed}", flush=True)


def close_app_and_exit(exit_code: int = 0) -> None:
    def force_exit() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)

    timer = threading.Timer(5.0, force_exit)
    timer.daemon = True
    timer.start()
    try:
        simulation_app.close()
    except Exception:
        pass
    finally:
        timer.cancel()
        force_exit()


if __name__ == "__main__":
    main()
    close_app_and_exit(0)
