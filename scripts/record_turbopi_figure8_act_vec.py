"""Vectorized ACT recorder for the TurboPi figure-8 map.

This script is the fast data path: it runs many cloned figure-8 arenas inside
one Isaac Lab process, drives all TurboPis with a batched kinematic teacher,
and writes accepted language-conditioned ACT episodes.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = ("go_left", "go_right")
TASK_INSTRUCTIONS = {"go_left": "go left", "go_right": "go right"}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "act_figure8_vec"

parser = argparse.ArgumentParser(description="Vectorized TurboPi figure-8 ACT recorder.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--session_name", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="turbopi_figure8_act_cvae")
parser.add_argument("--num_episodes", type=int, default=64)
parser.add_argument("--laps", type=int, default=3)
parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--image_width", type=int, default=96)
parser.add_argument("--image_height", type=int, default=72)
parser.add_argument("--target_speed", type=float, default=0.42)
parser.add_argument("--min_forward_speed", type=float, default=0.09)
parser.add_argument("--max_wz", type=float, default=1.35)
parser.add_argument("--position_tolerance", type=float, default=0.075)
parser.add_argument("--switch_distance", type=float, default=0.095)
parser.add_argument("--finish_progress", type=float, default=0.985)
parser.add_argument("--lookahead_distance", type=float, default=0.18)
parser.add_argument("--approach_distance", type=float, default=0.22)
parser.add_argument("--heading_gain", type=float, default=2.4)
parser.add_argument("--lookahead_heading_gain", type=float, default=1.1)
parser.add_argument("--heading_slowdown_angle", type=float, default=1.25)
parser.add_argument("--turn_in_place_angle", type=float, default=1.15)
parser.add_argument("--off_track_abort_distance", type=float, default=0.34)
parser.add_argument("--stuck_timeout", type=float, default=8.0)
parser.add_argument("--progress_epsilon", type=float, default=0.025)
parser.add_argument("--max_episode_time", type=float, default=90.0)
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--camera_warmup_steps", type=int, default=6)
parser.add_argument("--speed_jitter", type=float, default=0.15)
parser.add_argument("--start_xy_jitter", type=float, default=0.025)
parser.add_argument("--start_yaw_jitter", type=float, default=0.08)
parser.add_argument("--action_noise_std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
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
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from act_mountain_dataset import ACTEpisodeFrame, ACTEpisodeResult, ACTMountainSessionWriter
from common import (
    CAMERA_LINK_TO_SENSOR_POS,
    CAMERA_LINK_TO_SENSOR_ROT,
    build_turbopi_cfg,
    get_wheel_joint_ids,
    twist_to_wheel_targets,
)
from mountain_cliff_scene import MountainCliffSceneCfg, route_waypoints


ROAD_Z = 0.02
START_HEIGHT = 0.055
ROAD_WIDTH = 0.48
SHOULDER_WIDTH = 0.10
ROAD_THICKNESS = 0.035
MAX_COMMAND = np.array([0.45, 0.35, 2.0, 1.0], dtype=np.float32)


@dataclass
class EnvState:
    task_name: str = "go_left"
    task_index: int = 0
    segment_index: int = 0
    best_progress: float = 0.0
    last_progress_time: float = 0.0
    elapsed_steps: int = 0
    target_speed: float = 0.42
    frames: list[ACTEpisodeFrame] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    finished: bool = False
    success: bool = False
    terminal_reason: str = "active"


class StopFlag:
    requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


def yaw_to_quat_tuple(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def segment_geometry(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float, float, float]:
    cx = 0.5 * (start[0] + end[0])
    cy = 0.5 * (start[1] + end[1])
    length = math.dist(start, end)
    yaw = math.atan2(end[1] - start[1], end[0] - start[0]) - 0.5 * math.pi
    return cx, cy, length, yaw


def repeated_waypoints(task_name: str) -> tuple[tuple[float, float], ...]:
    cfg = MountainCliffSceneCfg(map_name="figure8")
    base = route_waypoints(cfg, task_name)
    points = list(base)
    for _ in range(max(1, int(args_cli.laps)) - 1):
        points.extend(base[1:])
    return tuple(points)


LEFT_POINTS = repeated_waypoints("go_left")
RIGHT_POINTS = repeated_waypoints("go_right")
LEFT_BASE_VISUAL = route_waypoints(MountainCliffSceneCfg(map_name="figure8"), "go_left")
RIGHT_BASE_VISUAL = route_waypoints(MountainCliffSceneCfg(map_name="figure8"), "go_right")


def build_route_tensors(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    routes_np = np.asarray([LEFT_POINTS, RIGHT_POINTS], dtype=np.float32)
    segments = routes_np[:, 1:, :] - routes_np[:, :-1, :]
    lengths = np.linalg.norm(segments, axis=-1).astype(np.float32)
    headings = np.arctan2(segments[..., 1], segments[..., 0]).astype(np.float32)
    cumulative = np.concatenate([np.zeros((2, 1), dtype=np.float32), np.cumsum(lengths, axis=1)], axis=1)
    return (
        torch.tensor(routes_np, dtype=torch.float32, device=device),
        torch.tensor(lengths, dtype=torch.float32, device=device),
        torch.tensor(headings, dtype=torch.float32, device=device),
        torch.tensor(cumulative, dtype=torch.float32, device=device),
        torch.tensor(np.sum(lengths, axis=1), dtype=torch.float32, device=device),
    )


def make_asset_cfg(
    prim_path: str,
    spawn,
    pos: tuple[float, float, float],
    yaw: float = 0.0,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=yaw_to_quat_tuple(yaw)),
    )


def build_scene_cfg_class(num_envs: int):
    env_spacing = 5.2
    total_width = ROAD_WIDTH + 2.0 * SHOULDER_WIDTH
    road_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.13, 0.085), roughness=0.96)
    shoulder_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.25, 0.16), roughness=0.98)
    mark_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.74, 0.46), roughness=0.68)
    floor_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.25, 0.20), roughness=0.98)

    attrs: dict[str, object] = {
        "num_envs": int(num_envs),
        "env_spacing": float(env_spacing),
        "ground": AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        ),
        "dome_light": AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=950.0, color=(0.95, 0.95, 0.95)),
        ),
        "floor": make_asset_cfg(
            "{ENV_REGEX_NS}/Floor",
            sim_utils.CuboidCfg(size=(4.8, 4.8, 0.006), collision_props=None, visual_material=floor_mat),
            (0.0, -0.25, ROAD_Z - 0.026),
        ),
        "robot": build_turbopi_cfg(
            asset_usd=args_cli.asset_usd,
            prim_path="{ENV_REGEX_NS}/TurboPi",
            add_rollers=False,
        ),
        "camera": CameraCfg(
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
        ),
    }

    for route_name, points in (("left", LEFT_BASE_VISUAL), ("right", RIGHT_BASE_VISUAL)):
        for idx, (start, end) in enumerate(zip(points[:-1], points[1:], strict=False)):
            cx, cy, length, yaw = segment_geometry(start, end)
            attrs[f"{route_name}_deck_{idx:02d}"] = make_asset_cfg(
                f"{{ENV_REGEX_NS}}/{route_name.capitalize()}RoadDeck{idx:02d}",
                sim_utils.CuboidCfg(size=(total_width, length + 0.04, ROAD_THICKNESS), collision_props=None, visual_material=shoulder_mat),
                (cx, cy, ROAD_Z - 0.5 * ROAD_THICKNESS),
                yaw,
            )
            attrs[f"{route_name}_surface_{idx:02d}"] = make_asset_cfg(
                f"{{ENV_REGEX_NS}}/{route_name.capitalize()}RoadSurface{idx:02d}",
                sim_utils.CuboidCfg(size=(ROAD_WIDTH, length + 0.055, 0.006), collision_props=None, visual_material=road_mat),
                (cx, cy, ROAD_Z + 0.004),
                yaw,
            )
            if idx % 2 == 0:
                attrs[f"{route_name}_mark_{idx:02d}"] = make_asset_cfg(
                    f"{{ENV_REGEX_NS}}/{route_name.capitalize()}CenterMark{idx:02d}",
                    sim_utils.CuboidCfg(size=(0.026, min(0.22, length * 0.55), 0.004), collision_props=None, visual_material=mark_mat),
                    (cx, cy, ROAD_Z + 0.012),
                    yaw,
                )

    _Cfg = configclass(type("_Figure8VecSceneCfg", (InteractiveSceneCfg,), attrs))
    return _Cfg


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_figure8_vec_%Y%m%d_%H%M%S")


def sample_task(env_idx: int) -> tuple[str, int]:
    value = env_idx % 2
    return TASKS[value], value


def reset_envs(
    scene: InteractiveScene,
    states: list[EnvState],
    env_ids: list[int],
    rng: np.random.Generator,
    next_episode_index: list[int],
) -> None:
    if not env_ids:
        return
    robot = scene["robot"]
    device = robot.device
    env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=device)
    root_state = robot.data.default_root_state[env_ids_t].clone()
    joint_pos = robot.data.default_joint_pos[env_ids_t].clone()
    joint_vel = robot.data.default_joint_vel[env_ids_t].clone()
    yaws = torch.zeros(len(env_ids), dtype=torch.float32, device=device)
    for local_i, env_idx in enumerate(env_ids):
        task_name, task_index = sample_task(env_idx)
        route = LEFT_POINTS if task_name == "go_left" else RIGHT_POINTS
        start = route[0]
        first = route[1]
        yaw = math.atan2(first[1] - start[1], first[0] - start[0])
        xy_jitter = float(args_cli.start_xy_jitter)
        yaw_jitter = float(args_cli.start_yaw_jitter)
        speed_jitter = float(args_cli.speed_jitter)
        x = start[0] + float(rng.uniform(-xy_jitter, xy_jitter))
        y = start[1] + float(rng.uniform(-xy_jitter, xy_jitter))
        yaw = ((yaw + float(rng.uniform(-yaw_jitter, yaw_jitter)) + math.pi) % (2.0 * math.pi)) - math.pi
        speed_scale = 1.0 + float(rng.uniform(-speed_jitter, speed_jitter))
        target_speed = max(args_cli.min_forward_speed, min(float(MAX_COMMAND[0]), args_cli.target_speed * speed_scale))

        states[env_idx] = EnvState(task_name=task_name, task_index=task_index, target_speed=target_speed)
        root_state[local_i, 0] = x
        root_state[local_i, 1] = y
        root_state[local_i, 2] = ROAD_Z + START_HEIGHT
        yaws[local_i] = yaw
    next_episode_index[0] += len(env_ids)
    root_state[:, :3] += scene.env_origins[env_ids_t]
    root_state[:, 3:7] = yaw_to_quat(yaws)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids_t)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids_t)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)


def batched_project(
    pos_xy: torch.Tensor,
    routes: torch.Tensor,
    lengths: torch.Tensor,
    headings: torch.Tensor,
    cumulative: torch.Tensor,
    total_lengths: torch.Tensor,
    states: list[EnvState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pos_xy.device
    task_ids = torch.tensor([state.task_index for state in states], dtype=torch.long, device=device)
    seg_ids = torch.tensor([state.segment_index for state in states], dtype=torch.long, device=device)
    starts = routes[task_ids, seg_ids]
    goals = routes[task_ids, seg_ids + 1]
    seg = goals - starts
    denom = torch.clamp(torch.sum(seg * seg, dim=-1), min=1e-9)
    t = torch.clamp(torch.sum((pos_xy - starts) * seg, dim=-1) / denom, 0.0, 1.0)
    nearest = starts + t.unsqueeze(-1) * seg
    error = torch.linalg.norm(pos_xy - nearest, dim=-1)
    progress_m = cumulative[task_ids, seg_ids] + t * lengths[task_ids, seg_ids]
    progress_ratio = torch.clamp(progress_m / total_lengths[task_ids], 0.0, 1.0)
    heading = headings[task_ids, seg_ids]
    dist_to_goal = torch.linalg.norm(goals - pos_xy, dim=-1)
    return starts, goals, t, error, progress_m, progress_ratio, heading, dist_to_goal


def compute_commands(
    local_pos: torch.Tensor,
    yaw: torch.Tensor,
    routes: torch.Tensor,
    lengths: torch.Tensor,
    headings: torch.Tensor,
    cumulative: torch.Tensor,
    total_lengths: torch.Tensor,
    states: list[EnvState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    starts, goals, t, error, progress_m, progress_ratio, heading, dist = batched_project(
        local_pos, routes, lengths, headings, cumulative, total_lengths, states
    )
    device = local_pos.device
    task_ids = torch.tensor([state.task_index for state in states], dtype=torch.long, device=device)
    seg_ids = torch.tensor([state.segment_index for state in states], dtype=torch.long, device=device)
    lookahead_t = torch.clamp(t + args_cli.lookahead_distance / torch.clamp(lengths[task_ids, seg_ids], min=1e-6), 0.0, 1.0)
    target = starts + lookahead_t.unsqueeze(-1) * (goals - starts)
    delta = target - local_pos
    target_bx = torch.cos(yaw) * delta[:, 0] + torch.sin(yaw) * delta[:, 1]
    target_by = -torch.sin(yaw) * delta[:, 0] + torch.cos(yaw) * delta[:, 1]
    point_error = torch.atan2(target_by, torch.clamp(target_bx, min=0.04))
    yaw_error = torch.atan2(torch.sin(heading - yaw), torch.cos(heading - yaw))
    approach = torch.clamp(dist / max(args_cli.approach_distance, 1e-6), min=0.35, max=1.0)
    heading_scale = torch.clamp(1.0 - torch.abs(yaw_error) / max(args_cli.heading_slowdown_angle, 1e-6), min=0.10, max=1.0)
    target_speed = torch.tensor([state.target_speed for state in states], dtype=torch.float32, device=device)
    vx = torch.clamp(target_speed * torch.minimum(approach, heading_scale), min=args_cli.min_forward_speed)
    vx = torch.minimum(vx, target_speed)
    vx = torch.where(torch.abs(yaw_error) >= args_cli.turn_in_place_angle, torch.zeros_like(vx), vx)
    wz = args_cli.heading_gain * yaw_error + args_cli.lookahead_heading_gain * point_error
    wz = torch.clamp(wz, min=-args_cli.max_wz, max=args_cli.max_wz)
    command = torch.stack((vx, torch.zeros_like(vx), wz), dim=-1)
    return command, error, progress_m, progress_ratio, dist


def integrate(pos_xy: torch.Tensor, yaw: torch.Tensor, command: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    vx = command[:, 0]
    vy = command[:, 1]
    wz = command[:, 2]
    yaw_mid = yaw + 0.5 * wz * dt
    new_x = pos_xy[:, 0] + (vx * torch.cos(yaw_mid) - vy * torch.sin(yaw_mid)) * dt
    new_y = pos_xy[:, 1] + (vx * torch.sin(yaw_mid) + vy * torch.cos(yaw_mid)) * dt
    new_yaw = wrap_to_pi(yaw + wz * dt)
    return torch.stack((new_x, new_y), dim=-1), new_yaw


def make_result(state: EnvState, control_dt: float) -> ACTEpisodeResult:
    return ACTEpisodeResult(
        task_name=state.task_name,
        task_index=state.task_index,
        instruction=TASK_INSTRUCTIONS[state.task_name],
        frames=state.frames,
        success=state.success,
        terminal_reason=state.terminal_reason,
        final_route_progress=1.0 if state.success else state.best_progress,
        mean_track_error=float(np.mean(state.errors)) if state.errors else float("inf"),
        duration_s=len(state.frames) * control_dt,
    )


def main() -> None:
    SceneCfgClass = build_scene_cfg_class(args_cli.num_envs)
    physics_dt = min(float(args_cli.physics_dt), 1.0 / max(args_cli.control_hz, 1e-6))
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = max(1, int(round(control_dt / physics_dt)))
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=physics_dt, render_interval=substeps, device=args_cli.device)
    )
    scene = InteractiveScene(SceneCfgClass())
    sim.reset()
    robot = scene["robot"]
    camera = scene["camera"]
    device = robot.device
    wheel_joint_ids = get_wheel_joint_ids(robot)
    routes, lengths, headings, cumulative, total_lengths = build_route_tensors(device)
    states = [EnvState() for _ in range(args_cli.num_envs)]
    rng = np.random.default_rng(args_cli.seed)
    next_episode_index = [0]
    reset_envs(scene, states, list(range(args_cli.num_envs)), rng, next_episode_index)
    scene.write_data_to_sim()
    for _ in range(args_cli.settle_steps + args_cli.camera_warmup_steps):
        sim.step()
        scene.update(physics_dt)
    camera.update(physics_dt)

    writer = ACTMountainSessionWriter(
        output_root=args_cli.output_dir,
        session_name=args_cli.session_name or datetime.utcnow().strftime("figure8_vec_%Y%m%d_%H%M%S"),
        dataset_name=args_cli.dataset_name,
        fps=args_cli.control_hz,
        image_width=args_cli.image_width,
        image_height=args_cli.image_height,
        control_hz=args_cli.control_hz,
        physics_dt=physics_dt,
        tasks=TASKS,
        task_instructions=TASK_INSTRUCTIONS,
        record_camera="robot_forward",
        map_name="figure8_vectorized",
        laps=args_cli.laps,
    )
    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print(f"[vec-record] Output session: {writer.session_dir}", flush=True)
    print(f"[vec-record] num_envs={args_cli.num_envs} target_episodes={args_cli.num_episodes}", flush=True)

    accepted = 0
    accepted_by_task = {task: 0 for task in TASKS}
    target_by_task = {
        "go_left": args_cli.num_episodes // 2,
        "go_right": args_cli.num_episodes - args_cli.num_episodes // 2,
    }
    failed = 0
    max_steps = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))
    wheel_targets_zero = torch.zeros((args_cli.num_envs, 4), dtype=torch.float32, device=device)
    try:
        while accepted < args_cli.num_episodes and simulation_app.is_running() and not stop_flag.requested:
            pos_w = robot.data.root_pos_w[:, :2]
            local_pos = pos_w - scene.env_origins[:, :2]
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
            command, error, progress_m, progress_ratio, dist = compute_commands(
                local_pos, yaw, routes, lengths, headings, cumulative, total_lengths, states
            )
            if args_cli.action_noise_std > 0.0:
                noise = torch.tensor(
                    rng.normal(0.0, args_cli.action_noise_std, size=(args_cli.num_envs, 3)).astype(np.float32),
                    dtype=torch.float32,
                    device=device,
                ) * torch.tensor(MAX_COMMAND[:3], dtype=torch.float32, device=device)
                command = command + noise
                command[:, 0].clamp_(args_cli.min_forward_speed, args_cli.target_speed)
                command[:, 1].clamp_(-0.18, 0.18)
                command[:, 2].clamp_(-args_cli.max_wz, args_cli.max_wz)

            rgb_batch = camera.data.output.get("rgb")
            if rgb_batch is None or rgb_batch.numel() == 0:
                rgb_np = np.zeros((args_cli.num_envs, args_cli.image_height, args_cli.image_width, 3), dtype=np.uint8)
            else:
                rgb = rgb_batch[..., :3]
                if rgb.dtype != torch.uint8:
                    rgb = torch.clamp(rgb, 0, 255).to(torch.uint8)
                rgb_np = rgb.detach().cpu().numpy()

            command_np = command.detach().cpu().numpy()
            action_np = np.clip(
                np.concatenate([command_np, np.zeros((args_cli.num_envs, 1), dtype=np.float32)], axis=1) / MAX_COMMAND,
                -1.0,
                1.0,
            ).astype(np.float32)
            error_np = error.detach().cpu().numpy()
            progress_np = progress_ratio.detach().cpu().numpy()
            dist_np = dist.detach().cpu().numpy()

            finished_ids: list[int] = []
            for env_idx, state in enumerate(states):
                if state.finished:
                    continue
                state.elapsed_steps += 1
                is_final = state.segment_index >= len(LEFT_POINTS) - 2
                if is_final and (dist_np[env_idx] <= args_cli.switch_distance or progress_np[env_idx] >= args_cli.finish_progress):
                    state.finished = True
                    state.success = True
                    state.terminal_reason = "goal_reached"
                    finished_ids.append(env_idx)
                    continue
                if dist_np[env_idx] <= args_cli.switch_distance and not is_final:
                    state.segment_index += 1
                    continue
                if error_np[env_idx] > args_cli.off_track_abort_distance:
                    state.finished = True
                    state.success = False
                    state.terminal_reason = "off_track"
                    finished_ids.append(env_idx)
                    continue
                if state.elapsed_steps >= max_steps:
                    state.finished = True
                    state.success = False
                    state.terminal_reason = "timeout"
                    finished_ids.append(env_idx)
                    continue
                if progress_np[env_idx] >= state.best_progress + args_cli.progress_epsilon:
                    state.best_progress = float(progress_np[env_idx])
                    state.last_progress_time = state.elapsed_steps * control_dt
                if state.elapsed_steps * control_dt - state.last_progress_time >= args_cli.stuck_timeout:
                    state.finished = True
                    state.success = False
                    state.terminal_reason = "stuck"
                    finished_ids.append(env_idx)
                    continue

                stop_value = 1.0 if progress_np[env_idx] >= 0.97 else 0.0
                command4 = np.asarray(
                    [command_np[env_idx, 0], command_np[env_idx, 1], command_np[env_idx, 2], stop_value],
                    dtype=np.float32,
                )
                action4 = action_np[env_idx].copy()
                action4[3] = stop_value
                state.frames.append(
                    ACTEpisodeFrame(
                        image_rgb=rgb_np[env_idx],
                        timestamp=float((state.elapsed_steps - 1) * control_dt),
                        state=state.prev_action.copy(),
                        action=action4.copy(),
                        command=command4.copy(),
                        track_error=float(error_np[env_idx]),
                        route_progress=float(progress_np[env_idx]),
                    )
                )
                state.prev_action = action4
                state.errors.append(float(error_np[env_idx]))

            new_xy, new_yaw = integrate(local_pos, yaw, command, control_dt)
            root_pose = torch.zeros((args_cli.num_envs, 7), dtype=torch.float32, device=device)
            root_pose[:, :2] = new_xy + scene.env_origins[:, :2]
            root_pose[:, 2] = ROAD_Z + START_HEIGHT
            root_pose[:, 3:7] = yaw_to_quat(new_yaw)
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(torch.zeros((args_cli.num_envs, 6), dtype=torch.float32, device=device))
            robot.set_joint_velocity_target(twist_to_wheel_targets(command, device), joint_ids=wheel_joint_ids)
            scene.write_data_to_sim()
            for _ in range(substeps):
                sim.step()
            scene.update(physics_dt)
            camera.update(control_dt)

            if finished_ids:
                reset_ids = []
                for env_idx in finished_ids:
                    state = states[env_idx]
                    if (
                        state.success
                        and state.frames
                        and accepted < args_cli.num_episodes
                        and accepted_by_task[state.task_name] < target_by_task[state.task_name]
                    ):
                        episode_dir = writer.save_episode(accepted, make_result(state, control_dt))
                        accepted_by_task[state.task_name] += 1
                        accepted += 1
                        print(
                            f"[vec-record] saved episode_{accepted - 1:05d} env={env_idx} "
                            f"{state.task_name} frames={len(state.frames)} "
                            f"left={accepted_by_task['go_left']}/{target_by_task['go_left']} "
                            f"right={accepted_by_task['go_right']}/{target_by_task['go_right']} -> {episode_dir}",
                            flush=True,
                        )
                    elif not state.success:
                        failed += 1
                        writer.record_failure()
                        print(
                            f"[vec-record] failed env={env_idx} reason={state.terminal_reason} "
                            f"progress={state.best_progress:.2f}",
                            flush=True,
                        )
                    if accepted < args_cli.num_episodes:
                        reset_ids.append(env_idx)
                if reset_ids:
                    reset_envs(scene, states, reset_ids, rng, next_episode_index)
                    scene.write_data_to_sim()
                    for _ in range(args_cli.settle_steps):
                        sim.step()
                    scene.update(physics_dt)
                    camera.update(physics_dt)
    finally:
        robot.set_joint_velocity_target(wheel_targets_zero, joint_ids=wheel_joint_ids)
        print(f"[vec-record] complete: saved={accepted} failed={failed} session={writer.session_dir}", flush=True)


def close_app_and_exit(code: int = 0) -> None:
    timer = threading.Timer(5.0, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    try:
        simulation_app.close()
    finally:
        timer.cancel()
        os._exit(code)


if __name__ == "__main__":
    main()
    close_app_and_exit(0)
