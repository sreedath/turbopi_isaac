"""Record simple TurboPi runs inside the square arena.

This script supports two modes:

- `--route segment`: reset the robot to a chosen start point, face a chosen
  goal point, drive there with `[vx, 0.0, wz]`, and record that one segment.
- `--route square_ccw` or `--route square_cw`: reset the robot at one corner,
  then drive a full four-edge square lap as one episode.

Examples:

    cd /workspace
    ./isaaclab/isaaclab.sh -p \
        /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py \
        --livestream 2 --view chase --route square_ccw --num_episodes 10
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cnn_square_loop"
NAMED_POINTS = ("bl", "br", "tr", "tl", "lm", "rm", "tm", "bm", "center")

parser = argparse.ArgumentParser(description="Record simple TurboPi segment runs or full square laps.")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", type=str, choices=("overview", "chase", "robot"), default="chase")
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--session_name", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="turbopi_square_point_to_point_cnn")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument(
    "--route",
    type=str,
    choices=("segment", "square_ccw", "square_cw"),
    default="segment",
    help="`segment` records one start->goal edge. `square_ccw`/`square_cw` record one full square lap per episode.",
)
parser.add_argument(
    "--direction",
    type=str,
    choices=("clockwise", "counterclockwise"),
    default="counterclockwise",
    help="Only used to choose the default start/goal edge when --start/--goal are omitted.",
)
parser.add_argument("--start", type=str, choices=NAMED_POINTS, default=None, help="Named start point inside the square.")
parser.add_argument("--goal", type=str, choices=NAMED_POINTS, default=None, help="Named goal point inside the square.")
parser.add_argument("--start_x", type=float, default=None, help="Optional explicit start x in meters.")
parser.add_argument("--start_y", type=float, default=None, help="Optional explicit start y in meters.")
parser.add_argument("--goal_x", type=float, default=None, help="Optional explicit goal x in meters.")
parser.add_argument("--goal_y", type=float, default=None, help="Optional explicit goal y in meters.")
parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--image_width", type=int, default=64)
parser.add_argument("--image_height", type=int, default=48)
parser.add_argument("--square_half_extent", type=float, default=0.45)
parser.add_argument("--floor_half_extent", type=float, default=1.40)
parser.add_argument("--wall_height", type=float, default=0.55)
parser.add_argument("--wall_thickness", type=float, default=0.04)
parser.add_argument("--tape_width", type=float, default=0.08)
parser.add_argument("--target_speed", type=float, default=0.30, help="Cruise forward speed in m/s.")
parser.add_argument("--min_speed_scale", type=float, default=0.25, help="Minimum forward-speed scale during slowdowns.")
parser.add_argument("--min_forward_speed", type=float, default=0.08, help="Minimum forward speed used by the full-square teacher.")
parser.add_argument("--approach_distance", type=float, default=0.16, help="Distance from goal where forward speed ramps down.")
parser.add_argument("--position_tolerance", type=float, default=0.05, help="Goal tolerance in meters.")
parser.add_argument("--turn_in_place_angle", type=float, default=0.75, help="If |yaw error| exceeds this, rotate before driving.")
parser.add_argument("--drive_heading_gain", type=float, default=0.8, help="P gain on yaw error while driving.")
parser.add_argument("--square_heading_gain", type=float, default=2.7, help="Heading gain used by the full-square teacher.")
parser.add_argument("--square_cross_track_gain", type=float, default=1.2, help="Extra yaw correction toward the lookahead point on the square.")
parser.add_argument("--square_max_wz", type=float, default=1.10, help="Yaw-rate cap used by the full-square teacher.")
parser.add_argument("--lookahead_distance", type=float, default=0.12, help="Lookahead distance along the square loop in meters.")
parser.add_argument("--lap_completion_threshold", type=float, default=0.97, help="Lap progress threshold required for a full-square success.")
parser.add_argument("--start_phase_tolerance", type=float, default=0.06, help="How close the robot must return to its start phase to close a full square lap.")
parser.add_argument("--start_yaw_tolerance", type=float, default=0.35, help="How close the robot must return to its start yaw to close a full square lap.")
parser.add_argument("--min_square_distance_ratio", type=float, default=0.85, help="Minimum commanded forward distance, as a fraction of the square perimeter, before a full lap can be accepted.")
parser.add_argument("--corner_slowdown_distance", type=float, default=0.24, help="Distance from a corner where the full-square teacher starts slowing down.")
parser.add_argument("--heading_slowdown_angle", type=float, default=1.20, help="Heading error angle where the full-square teacher strongly reduces forward speed.")
parser.add_argument("--enable_omega_feedback", dest="enable_omega_feedback", action="store_true", default=True, help="Enable closed-loop yaw-rate compensation during autonomous recording (default: on).")
parser.add_argument("--disable_omega_feedback", dest="enable_omega_feedback", action="store_false", help="Disable closed-loop yaw-rate compensation during autonomous recording.")
parser.add_argument("--omega_feedback_gain", type=float, default=2.0, help="Closed-loop yaw-rate feedback gain.")
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured yaw rate in the compensator.")
parser.add_argument("--max_wz", type=float, default=0.20, help="Yaw-rate cap in rad/s.")
parser.add_argument("--off_track_abort_distance", type=float, default=0.25, help="Abort if distance from the start-goal line gets too large.")
parser.add_argument("--stuck_timeout", type=float, default=8.0, help="Abort if path progress stalls for too long.")
parser.add_argument("--progress_epsilon", type=float, default=0.02, help="Meters of forward progress needed to reset the stuck timer.")
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--cooldown_steps", type=int, default=2)
parser.add_argument("--camera_warmup_steps", type=int, default=6)
parser.add_argument(
    "--randomize_start",
    action="store_true",
    help="Randomize start phase along the square plus small lateral/yaw jitter for variety.",
)
parser.add_argument("--start_lateral_jitter", type=float, default=0.03, help="Max lateral start offset (m) when randomizing.")
parser.add_argument("--start_yaw_jitter_deg", type=float, default=10.0, help="Max yaw jitter (deg) at start when randomizing.")
parser.add_argument(
    "--mix_directions",
    action="store_true",
    help="Alternate ccw/cw episodes (overrides --route direction for full_square mode).",
)
parser.add_argument("--action_noise_std", type=float, default=0.0, help="Stddev of Gaussian noise added to commanded action (in normalized [-1,1] units).")
parser.add_argument(
    "--record_external",
    action="store_true",
    help="Also record an overhead spectator camera as external.mp4 per episode.",
)
parser.add_argument("--external_width", type=int, default=480)
parser.add_argument("--external_height", type=int, default=480)
parser.add_argument("--external_z", type=float, default=2.5, help="Height (m) of overhead spectator camera (overhead view only).")
parser.add_argument(
    "--external_view",
    type=str,
    choices=("overhead", "chase"),
    default="overhead",
    help="`overhead` = fixed top-down camera. `chase` = third-person camera that follows the robot.",
)
parser.add_argument("--chase_back", type=float, default=0.45, help="Chase camera distance behind the robot (m).")
parser.add_argument("--chase_up", type=float, default=0.30, help="Chase camera height above the robot (m).")
parser.add_argument("--chase_target_forward", type=float, default=0.30, help="Chase look-at point forward of robot (m).")
parser.add_argument("--chase_target_up", type=float, default=0.05, help="Chase look-at point above floor (m).")
parser.add_argument("--no_onboard_video", action="store_true", help="Skip writing the per-episode on-board video.mp4 (faster).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min_image_std", type=float, default=8.0)
parser.add_argument("--max_episode_time", type=float, default=45.0)
parser.add_argument("--no_rollers", action="store_true")
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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_from_euler_xyz

from cnn_dataset import CNNSessionWriter, EpisodeFrame, EpisodeResult
from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot_pose,
    resolve_asset_usd,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from square_loop import (
    SquareTrackSceneCfg,
    build_overhead_camera_sensor,
    build_robot_camera_sensor,
    compute_square_track_frame,
    design_square_loop_scene,
    direction_sign,
    phase_to_segment_and_progress,
    segment_tangent_clockwise,
    square_phase_to_point_and_tangent,
    start_pose_for_direction,
)


@dataclass(frozen=True)
class SegmentPath:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    label: str
    length: float
    tangent_xy: tuple[float, float]
    start_yaw: float


@dataclass(frozen=True)
class RoutePath:
    label: str
    segments: tuple[SegmentPath, ...]
    length: float
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw: float
    mode: str
    direction_name: str | None = None


class StopFlag:
    def __init__(self):
        self.requested = False

    def request(self, signum, _frame):
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_phase_error(phase_a: float, phase_b: float) -> float:
    delta = (phase_a - phase_b + 0.5) % 1.0 - 0.5
    return abs(delta)


def pick_random_start_pose(
    scene_cfg: SquareTrackSceneCfg,
    direction: str,
    rng: np.random.Generator,
    *,
    lateral_jitter: float,
    yaw_jitter_deg: float,
) -> tuple[tuple[float, float], float, float]:
    """Sample a random start pose along the taped square boundary.

    Returns ((x, y), yaw, start_phase). The start phase is in the cw-parametrized
    [0, 1) used by ``square_phase_to_point_and_tangent``.
    """
    sign = direction_sign(direction)
    phase = float(rng.uniform(0.0, 1.0))
    point_t, tangent_t = square_phase_to_point_and_tangent(
        torch.tensor([phase], dtype=torch.float32), scene_cfg.square_half_extent
    )
    px = float(point_t[0, 0].item())
    py = float(point_t[0, 1].item())
    tx = float(tangent_t[0, 0].item()) * sign
    ty = float(tangent_t[0, 1].item()) * sign

    # Inward normal = rotate tangent 90deg toward arena interior.
    # For a clockwise tangent, the inward normal is (ty, -tx); when sign flips,
    # it inverts too — we want the normal pointing inside, so use the corrected
    # tangent's left-hand perpendicular.
    nx = -ty
    ny = tx
    # If this normal happens to point outward for the cw segments (sign>0), flip.
    # Easy check: nearest interior point should be closer to origin than the wall.
    if (px + nx) ** 2 + (py + ny) ** 2 > px ** 2 + py ** 2:
        nx, ny = -nx, -ny
    lat = float(rng.uniform(-lateral_jitter, lateral_jitter))
    x = px + nx * lat
    y = py + ny * lat

    yaw_t = math.atan2(ty, tx)
    yaw = wrap_to_pi(yaw_t + math.radians(float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))))
    return (x, y), yaw, phase


def signed_phase_delta(current_phase: float, previous_phase: float, direction: str) -> float:
    delta = current_phase - previous_phase
    if delta < -0.5:
        delta += 1.0
    if delta > 0.5:
        delta -= 1.0
    sign = 1.0 if direction == "clockwise" else -1.0
    return max(0.0, sign * delta)


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_simple_%Y%m%d_%H%M%S")


def named_point_xy(name: str, half_extent: float) -> tuple[float, float]:
    h = half_extent
    table = {
        "bl": (-h, -h),
        "br": (h, -h),
        "tr": (h, h),
        "tl": (-h, h),
        "lm": (-h, 0.0),
        "rm": (h, 0.0),
        "tm": (0.0, h),
        "bm": (0.0, -h),
        "center": (0.0, 0.0),
    }
    return table[name]


def build_segment(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    start_label: str,
    goal_label: str,
) -> SegmentPath:
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        raise ValueError("Start and goal must be different points.")

    tangent_xy = (dx / length, dy / length)
    start_yaw = math.atan2(dy, dx)
    return SegmentPath(
        start_xy=start_xy,
        goal_xy=goal_xy,
        label=f"{start_label}_to_{goal_label}",
        length=length,
        tangent_xy=tangent_xy,
        start_yaw=start_yaw,
    )


def resolve_segment_path(half_extent: float) -> SegmentPath:
    if args_cli.start_x is not None or args_cli.start_y is not None:
        if args_cli.start_x is None or args_cli.start_y is None:
            raise ValueError("Specify both --start_x and --start_y.")
        start_xy = (float(args_cli.start_x), float(args_cli.start_y))
        start_label = f"({start_xy[0]:+.2f},{start_xy[1]:+.2f})"
    elif args_cli.start is not None:
        start_xy = named_point_xy(args_cli.start, half_extent)
        start_label = args_cli.start
    else:
        default_start = "bl"
        start_xy = named_point_xy(default_start, half_extent)
        start_label = default_start

    if args_cli.goal_x is not None or args_cli.goal_y is not None:
        if args_cli.goal_x is None or args_cli.goal_y is None:
            raise ValueError("Specify both --goal_x and --goal_y.")
        goal_xy = (float(args_cli.goal_x), float(args_cli.goal_y))
        goal_label = f"({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})"
    elif args_cli.goal is not None:
        goal_xy = named_point_xy(args_cli.goal, half_extent)
        goal_label = args_cli.goal
    else:
        default_goal = "tl" if args_cli.direction == "clockwise" else "br"
        goal_xy = named_point_xy(default_goal, half_extent)
        goal_label = default_goal

    return build_segment(start_xy, goal_xy, start_label=start_label, goal_label=goal_label)


def build_named_segment(start_name: str, goal_name: str, half_extent: float) -> SegmentPath:
    return build_segment(
        named_point_xy(start_name, half_extent),
        named_point_xy(goal_name, half_extent),
        start_label=start_name,
        goal_label=goal_name,
    )


def resolve_route_path(half_extent: float) -> RoutePath:
    if args_cli.route == "square_ccw":
        names = ("lm", "bl", "br", "tr", "tl", "lm")
        segments = tuple(build_named_segment(names[i], names[i + 1], half_extent) for i in range(len(names) - 1))
        return RoutePath(
            label="square_ccw",
            segments=segments,
            length=sum(segment.length for segment in segments),
            start_xy=segments[0].start_xy,
            goal_xy=segments[-1].goal_xy,
            start_yaw=segments[0].start_yaw,
            mode="full_square",
            direction_name="counterclockwise",
        )
    if args_cli.route == "square_cw":
        names = ("lm", "tl", "tr", "br", "bl", "lm")
        segments = tuple(build_named_segment(names[i], names[i + 1], half_extent) for i in range(len(names) - 1))
        return RoutePath(
            label="square_cw",
            segments=segments,
            length=sum(segment.length for segment in segments),
            start_xy=segments[0].start_xy,
            goal_xy=segments[-1].goal_xy,
            start_yaw=segments[0].start_yaw,
            mode="full_square",
            direction_name="clockwise",
        )

    segment = resolve_segment_path(half_extent)
    return RoutePath(
        label=segment.label,
        segments=(segment,),
        length=segment.length,
        start_xy=segment.start_xy,
        goal_xy=segment.goal_xy,
        start_yaw=segment.start_yaw,
        mode="segment",
    )


def ensure_sim_playing(sim):
    if not sim.is_playing():
        sim.play()


def rgb_frame(camera) -> np.ndarray:
    image = camera.data.output["rgb"]
    if image is None or image.numel() == 0:
        raise RuntimeError("Camera sensor has no RGB data yet.")
    rgb = image[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 255.0)
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
        else:
            rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype(np.uint8)
    return rgb


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def apply_command(
    robot,
    wheel_joint_ids,
    arm_joint_ids,
    command_vec: tuple[float, float, float],
    *,
    physics_dt: float,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
):
    command_t = torch.tensor([command_vec], dtype=torch.float32, device=robot.device)
    if omega_tracker is not None:
        command_t = omega_tracker.compensate(
            command_t,
            robot.data.root_ang_vel_b[:, 2],
            dt=physics_dt,
            command_limit=max_omega_command,
        )
    wheel_targets = twist_to_wheel_targets(command_t, robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()
    return tuple(float(v) for v in command_t[0].detach().cpu().tolist())


def step_n(
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    command_vec,
    substeps,
    physics_dt,
    viewport,
    active_view,
    omega_tracker,
    max_omega_command,
):
    applied_command = command_vec
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False, applied_command
        ensure_sim_playing(sim)
        applied_command = apply_command(
            robot,
            wheel_joint_ids,
            arm_joint_ids,
            command_vec,
            physics_dt=physics_dt,
            omega_tracker=omega_tracker,
            max_omega_command=max_omega_command,
        )
        sim.step()
        robot.update(physics_dt)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True, applied_command


def body_twist_to_world_velocity(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_vx = vx * cos_yaw - vy * sin_yaw
    world_vy = vx * sin_yaw + vy * cos_yaw
    return world_vx, world_vy


def write_kinematic_state(
    robot,
    wheel_joint_ids,
    arm_joint_ids,
    pose: tuple[float, float, float],
    command_vec: tuple[float, float, float],
) -> None:
    x, y, yaw = pose
    vx, vy, wz = command_vec
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = float(x)
    root_pose[:, 1] = float(y)
    root_pose[:, 2] = float(robot.data.default_root_state[0, 2].item())

    yaw_t = torch.full((robot.num_instances,), float(yaw), dtype=torch.float32, device=robot.device)
    zeros = torch.zeros_like(yaw_t)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)

    root_velocity = torch.zeros((robot.num_instances, 6), dtype=torch.float32, device=robot.device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)

    command_t = torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=robot.device)
    wheel_targets = twist_to_wheel_targets(command_t, robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def integrate_body_pose(
    pose: tuple[float, float, float],
    command_vec: tuple[float, float, float],
    dt: float,
) -> tuple[float, float, float]:
    x, y, yaw = pose
    vx, vy, wz = command_vec
    yaw_mid = yaw + 0.5 * wz * dt
    x += (vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid)) * dt
    y += (vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid)) * dt
    yaw = wrap_to_pi(yaw + wz * dt)
    return x, y, yaw


def step_kinematic_n(
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    pose,
    command_vec,
    substeps,
    physics_dt,
    viewport,
    active_view,
):
    current_pose = pose
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False, current_pose
        ensure_sim_playing(sim)
        current_pose = integrate_body_pose(current_pose, command_vec, physics_dt)
        write_kinematic_state(robot, wheel_joint_ids, arm_joint_ids, current_pose, command_vec)
        sim.step()
        robot.update(physics_dt)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True, current_pose


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = wrap_to_pi(float(yaw_t[0].item()))
    return x, y, yaw


def distance_to_segment(point_xy: tuple[float, float], start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> float:
    nearest_x, nearest_y, _ = project_point_to_segment(point_xy, start_xy, goal_xy)
    px, py = point_xy
    return math.hypot(px - nearest_x, py - nearest_y)


def project_point_to_segment(
    point_xy: tuple[float, float],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> tuple[float, float, float]:
    px, py = point_xy
    sx, sy = start_xy
    gx, gy = goal_xy
    dx = gx - sx
    dy = gy - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return sx, sy, 0.0
    t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
    t = clamp(t, 0.0, 1.0)
    nearest_x = sx + t * dx
    nearest_y = sy + t * dy
    return nearest_x, nearest_y, t


def path_progress_m(point_xy: tuple[float, float], path: SegmentPath) -> float:
    px, py = point_xy
    sx, sy = path.start_xy
    tx, ty = path.tangent_xy
    along = (px - sx) * tx + (py - sy) * ty
    return clamp(along, 0.0, path.length)


def target_point_in_body(robot, target_xy: tuple[float, float]) -> tuple[float, float]:
    target_w = torch.tensor([[target_xy[0], target_xy[1], 0.0]], dtype=torch.float32, device=robot.device)
    root_w = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
    root_w[:, :2] = robot.data.root_pos_w[:, :2]
    target_b = quat_apply_inverse(robot.data.root_quat_w, target_w - root_w)[0, :2]
    return float(target_b[0].item()), float(target_b[1].item())


def compute_command(robot, path: SegmentPath, pose: tuple[float, float, float]) -> tuple[tuple[float, float, float], float, float]:
    x, y, _yaw = pose
    dist_to_goal = math.hypot(path.goal_xy[0] - x, path.goal_xy[1] - y)
    if dist_to_goal <= args_cli.position_tolerance:
        return (0.0, 0.0, 0.0), 0.0, dist_to_goal

    target_bx, target_by = target_point_in_body(robot, path.goal_xy)
    yaw_error = math.atan2(target_by, target_bx)

    approach_scale = clamp(dist_to_goal / max(args_cli.approach_distance, 1e-6), args_cli.min_speed_scale, 1.0)
    heading_scale = clamp(math.cos(yaw_error), 0.0, 1.0)
    vx = args_cli.target_speed * approach_scale * max(args_cli.min_speed_scale, heading_scale)
    if abs(yaw_error) >= args_cli.turn_in_place_angle:
        vx = 0.0

    wz = clamp(args_cli.drive_heading_gain * yaw_error, -args_cli.max_wz, args_cli.max_wz)
    return (vx, 0.0, wz), yaw_error, dist_to_goal


def compute_full_square_command(robot, scene_cfg: SquareTrackSceneCfg, direction: str):
    sign = direction_sign(direction)
    pos_xy = robot.data.root_pos_w[:, :2]
    quat_w = robot.data.root_quat_w
    nearest_xy, _nearest_tangent, track_error, phase = compute_square_track_frame(pos_xy, scene_cfg.square_half_extent)

    segment, segment_progress = phase_to_segment_and_progress(phase)
    current_tangent_xy = segment_tangent_clockwise(segment) * sign
    direction_step = 1 if sign > 0.0 else -1
    next_segment = torch.remainder(segment + direction_step, 4)
    next_tangent_xy = segment_tangent_clockwise(next_segment) * sign
    segment_length = 2.0 * scene_cfg.square_half_extent
    distance_to_corner = (
        (1.0 - segment_progress) * segment_length
        if sign > 0.0
        else segment_progress * segment_length
    )
    blend_distance = max(args_cli.lookahead_distance, args_cli.corner_slowdown_distance)
    corner_blend = torch.clamp(1.0 - distance_to_corner / max(blend_distance, 1e-4), min=0.0, max=1.0)
    blended_tangent_xy = (
        (1.0 - corner_blend).unsqueeze(-1) * current_tangent_xy
        + corner_blend.unsqueeze(-1) * next_tangent_xy
    )
    blended_norm = torch.linalg.norm(blended_tangent_xy, dim=-1, keepdim=True)
    target_tangent_xy = torch.where(
        blended_norm > 1e-6,
        blended_tangent_xy / blended_norm,
        current_tangent_xy,
    )

    lookahead_phase = phase + sign * (args_cli.lookahead_distance / max(8.0 * scene_cfg.square_half_extent, 1e-6))
    target_point_xy, _ = square_phase_to_point_and_tangent(lookahead_phase, scene_cfg.square_half_extent)

    target_point_w = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
    target_point_w[:, :2] = target_point_xy
    pos_w_3 = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
    pos_w_3[:, :2] = pos_xy
    lookahead_vec_b = quat_apply_inverse(quat_w, target_point_w - pos_w_3)[:, :2]
    nearest_point_w = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
    nearest_point_w[:, :2] = nearest_xy
    nearest_vec_b = quat_apply_inverse(quat_w, nearest_point_w - pos_w_3)[:, :2]

    _, _, yaw = euler_xyz_from_quat(quat_w)
    path_yaw = torch.atan2(target_tangent_xy[:, 1], target_tangent_xy[:, 0])
    yaw_error_t = torch.atan2(torch.sin(path_yaw - yaw), torch.cos(path_yaw - yaw))
    yaw_error = float(yaw_error_t[0].item())
    point_heading_error_t = torch.atan2(lookahead_vec_b[:, 1], torch.clamp(lookahead_vec_b[:, 0], min=0.04))

    corner_speed_scale = torch.clamp(
        distance_to_corner / max(args_cli.corner_slowdown_distance, 1e-4),
        min=0.55,
        max=1.0,
    )
    heading_speed_scale = torch.clamp(
        1.0 - torch.abs(yaw_error_t) / max(args_cli.heading_slowdown_angle, 1e-4),
        min=0.45,
        max=1.0,
    )
    speed = float(args_cli.target_speed * torch.minimum(corner_speed_scale, heading_speed_scale)[0].item())
    vx = clamp(speed * clamp(math.cos(yaw_error), 0.35, 1.0), args_cli.min_forward_speed, args_cli.target_speed)

    desired_wz = (
        args_cli.square_heading_gain * yaw_error_t
        + args_cli.square_cross_track_gain * point_heading_error_t
        + 0.35 * nearest_vec_b[:, 1]
    )
    wz = float(torch.clamp(desired_wz, min=-args_cli.square_max_wz, max=args_cli.square_max_wz)[0].item())
    return (vx, 0.0, wz), yaw_error, float(track_error[0].item()), float(phase[0].item())


def compute_leg_command(robot, leg: SegmentPath, pose: tuple[float, float, float]):
    x, y, current_yaw = pose
    dist_to_goal = math.hypot(leg.goal_xy[0] - x, leg.goal_xy[1] - y)
    if dist_to_goal <= args_cli.position_tolerance:
        return (0.0, 0.0, 0.0), 0.0, dist_to_goal

    nearest_x, nearest_y, _ = project_point_to_segment((x, y), leg.start_xy, leg.goal_xy)
    nearest_w = torch.tensor([[nearest_x, nearest_y, 0.0]], dtype=torch.float32, device=robot.device)
    pos_w = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
    pos_w[:, :2] = robot.data.root_pos_w[:, :2]
    nearest_vec_b = quat_apply_inverse(robot.data.root_quat_w, nearest_w - pos_w)[0, :2]

    target_bx, target_by = target_point_in_body(robot, leg.goal_xy)
    point_heading_error = math.atan2(target_by, max(target_bx, 0.04))
    yaw_error = wrap_to_pi(leg.start_yaw - current_yaw)

    approach_scale = clamp(dist_to_goal / max(args_cli.approach_distance, 1e-6), args_cli.min_speed_scale, 1.0)
    heading_scale = clamp(1.0 - abs(yaw_error) / max(args_cli.heading_slowdown_angle, 1e-4), 0.05, 1.0)
    vx = clamp(args_cli.target_speed * min(approach_scale, heading_scale), args_cli.min_forward_speed, args_cli.target_speed)
    if abs(yaw_error) >= args_cli.turn_in_place_angle:
        vx = 0.0

    wz = clamp(
        args_cli.square_heading_gain * yaw_error
        + args_cli.square_cross_track_gain * point_heading_error
        + 0.35 * float(nearest_vec_b[1].item()),
        -args_cli.square_max_wz,
        args_cli.square_max_wz,
    )
    return (vx, 0.0, wz), yaw_error, dist_to_goal


def warm_camera(sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, active_view, steps, physics_dt, min_std, omega_tracker, max_omega_command):
    last_std = 0.0
    for _ in range(max(1, steps)):
        ok, _ = step_n(
            sim,
            robot,
            camera,
            wheel_joint_ids,
            arm_joint_ids,
            (0.0, 0.0, 0.0),
            1,
            physics_dt,
            viewport,
            active_view,
            omega_tracker,
            max_omega_command,
        )
        if not ok:
            return False, last_std
        try:
            last_std = float(np.asarray(rgb_frame(camera), dtype=np.float32).std())
        except Exception:
            last_std = 0.0
        if last_std >= min_std:
            return True, last_std
    return False, last_std


def run_episode(
    *,
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    viewport,
    active_view,
    scene_cfg: SquareTrackSceneCfg,
    route: RoutePath,
    stop_flag,
    omega_tracker,
    overhead_camera=None,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.0,
    skip_camera_warmup: bool = False,
    start_pose_override: tuple[tuple[float, float], float] | None = None,
):
    if route.mode == "full_square":
        return run_full_square_episode(
            sim=sim,
            robot=robot,
            camera=camera,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            viewport=viewport,
            active_view=active_view,
            scene_cfg=scene_cfg,
            route=route,
            stop_flag=stop_flag,
            omega_tracker=omega_tracker,
            overhead_camera=overhead_camera,
            rng=rng,
            noise_std=noise_std,
            skip_camera_warmup=skip_camera_warmup,
            start_pose_override=start_pose_override,
        )
    reset_robot_pose(robot, position=(route.start_xy[0], route.start_xy[1], 0.04), yaw=route.start_yaw)

    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    max_steps = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))

    if omega_tracker is not None:
        omega_tracker.reset()
    ok, _ = step_n(
        sim,
        robot,
        camera,
        wheel_joint_ids,
        arm_joint_ids,
        (0.0, 0.0, 0.0),
        max(1, args_cli.settle_steps),
        physics_dt,
        viewport,
        active_view,
        omega_tracker,
        args_cli.max_wz,
    )
    if not ok:
        raise RuntimeError("Simulation app closed during reset settle.")

    camera_ready, warm_std = warm_camera(
        sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, active_view,
        args_cli.camera_warmup_steps, physics_dt, args_cli.min_image_std, omega_tracker, args_cli.max_wz,
    )
    if not camera_ready:
        print(f"[WARN] Camera warmup std stayed at {warm_std:.2f}; continuing anyway.", flush=True)

    frames: list[EpisodeFrame] = []
    track_errors: list[float] = []
    body_speeds: list[float] = []
    image_stds: list[float] = []
    action_history: list[np.ndarray] = []
    prev_action = np.zeros(3, dtype=np.float32)
    lap_progress = 0.0
    best_progress_m = 0.0
    last_progress_time = 0.0
    terminal_reason = "timeout"
    success = False
    last_print_at = -1.0
    max_command = np.array([0.45, 0.35, 2.0], dtype=np.float32)
    segment_index = 0

    print(
        f"[INFO] Starting {route.label} run from ({route.start_xy[0]:+.2f}, {route.start_xy[1]:+.2f}) "
        f"to ({route.goal_xy[0]:+.2f}, {route.goal_xy[1]:+.2f}) with {len(route.segments)} segment(s).",
        flush=True,
    )

    for step_index in range(max_steps):
        if stop_flag.requested:
            terminal_reason = "interrupted"
            break
        if not simulation_app.is_running():
            terminal_reason = "app_closed"
            break

        segment = route.segments[segment_index]
        completed_length_m = sum(item.length for item in route.segments[:segment_index])
        pose = get_pose(robot)
        segment_progress_m = path_progress_m((pose[0], pose[1]), segment)
        progress_m = min(route.length, completed_length_m + segment_progress_m)
        progress_ratio = progress_m / max(route.length, 1e-6)
        track_error = distance_to_segment((pose[0], pose[1]), segment.start_xy, segment.goal_xy)
        if track_error > args_cli.off_track_abort_distance:
            terminal_reason = "off_track"
            break

        command_vec, yaw_error, dist_to_goal = compute_command(robot, segment, pose)
        if dist_to_goal <= args_cli.position_tolerance:
            best_progress_m = max(best_progress_m, min(route.length, completed_length_m + segment.length))
            last_progress_time = step_index * control_dt
            if segment_index >= len(route.segments) - 1:
                terminal_reason = "goal_reached"
                success = True
                break
            segment_index += 1
            continue

        if progress_m >= best_progress_m + args_cli.progress_epsilon:
            best_progress_m = progress_m
            last_progress_time = step_index * control_dt
        if (step_index * control_dt) >= args_cli.stuck_timeout and ((step_index * control_dt) - last_progress_time) >= args_cli.stuck_timeout:
            terminal_reason = "stuck"
            break

        body_lin = robot.data.root_lin_vel_b[0, :2].detach().cpu().numpy()
        body_ang = robot.data.root_ang_vel_b[0, 2:3].detach().cpu().numpy()
        body_vel = np.concatenate([body_lin, body_ang]).astype(np.float32)
        body_speeds.append(float(np.linalg.norm(body_lin)))
        track_errors.append(track_error)

        image = rgb_frame(camera)
        image_stds.append(float(np.asarray(image, dtype=np.float32).std()))

        command_np = np.asarray(command_vec, dtype=np.float32)
        action_np = np.clip(command_np / max_command, -1.0, 1.0)
        action_history.append(action_np)

        frames.append(
            EpisodeFrame(
                image_rgb=image,
                timestamp=float(step_index * control_dt),
                state=prev_action.copy(),
                action=action_np.copy(),
                command=command_np.copy(),
                body_velocity=body_vel,
                track_error=float(track_error),
                lap_progress=float(progress_ratio),
            )
        )
        prev_action = action_np

        ok, applied_command = step_n(
            sim,
            robot,
            camera,
            wheel_joint_ids,
            arm_joint_ids,
            command_vec,
            substeps,
            physics_dt,
            viewport,
            active_view,
            omega_tracker,
            args_cli.max_wz,
        )
        if not ok:
            terminal_reason = "app_closed"
            break

        episode_time = step_index * control_dt
        if episode_time - last_print_at >= 1.0:
            last_print_at = episode_time
            print(
                f"[INFO] {route.label:20s} t={episode_time:5.1f}s seg={segment_index + 1}/{len(route.segments)} "
                f"progress={progress_ratio:0.3f} "
                f"dist={dist_to_goal:0.3f} track={track_error:0.3f} yaw_err={yaw_error:+0.2f} "
                f"pose=({pose[0]:+0.2f},{pose[1]:+0.2f},{pose[2]:+0.2f}) "
                f"cmd=[{applied_command[0]:+0.2f},{applied_command[1]:+0.2f},{applied_command[2]:+0.2f}]",
                flush=True,
            )

    step_n(
        sim,
        robot,
        camera,
        wheel_joint_ids,
        arm_joint_ids,
        (0.0, 0.0, 0.0),
        max(1, args_cli.cooldown_steps),
        physics_dt,
        viewport,
        active_view,
        omega_tracker,
        args_cli.max_wz,
    )

    duration_s = len(frames) * control_dt
    mean_track_error = float(np.mean(track_errors)) if track_errors else float("inf")
    p90_track_error = float(np.quantile(track_errors, 0.9)) if track_errors else float("inf")
    max_track_error = float(np.max(track_errors)) if track_errors else float("inf")
    frames_over_010_ratio = float(np.mean(np.asarray(track_errors) > 0.10)) if track_errors else 1.0
    frames_over_015_ratio = float(np.mean(np.asarray(track_errors) > 0.15)) if track_errors else 1.0
    mean_image_std = float(np.mean(image_stds)) if image_stds else 0.0
    min_image_std_val = float(np.min(image_stds)) if image_stds else 0.0
    mean_speed = float(np.mean(body_speeds)) if body_speeds else 0.0
    action_array = np.asarray(action_history, dtype=np.float32) if action_history else np.zeros((0, 3), dtype=np.float32)
    mean_abs = np.mean(np.abs(action_array), axis=0) if len(action_array) > 0 else np.zeros(3, dtype=np.float32)
    mean_vy_vx = float(mean_abs[1] / max(float(mean_abs[0]), 1e-6))
    final_progress = float(best_progress_m / max(route.length, 1e-6)) if frames else 0.0

    return (
        EpisodeResult(
            direction=route.label,
            task_name=route.label,
            task_index=0,
            frames=frames,
            success=success,
            terminal_reason=terminal_reason,
            final_lap_progress=1.0 if success else final_progress,
            mean_track_error=mean_track_error,
            p90_track_error=p90_track_error,
            max_track_error=max_track_error,
            frames_over_010_ratio=frames_over_010_ratio,
            frames_over_015_ratio=frames_over_015_ratio,
            mean_image_std=mean_image_std,
            min_image_std=min_image_std_val,
            mean_abs_action_vx=float(mean_abs[0]),
            mean_abs_action_vy=float(mean_abs[1]),
            mean_abs_action_wz=float(mean_abs[2]),
            mean_action_vy_vx_ratio=mean_vy_vx,
            mean_speed=mean_speed,
            duration_s=duration_s,
        ),
        [],
    )


def run_full_square_episode(
    *,
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    viewport,
    active_view,
    scene_cfg: SquareTrackSceneCfg,
    route: RoutePath,
    stop_flag,
    omega_tracker,
    overhead_camera=None,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.0,
    skip_camera_warmup: bool = False,
    start_pose_override: tuple[tuple[float, float], float] | None = None,
):
    direction = route.direction_name or "counterclockwise"
    if start_pose_override is not None:
        start_xy, start_yaw = start_pose_override
    else:
        start_xy, start_yaw = route.start_xy, route.start_yaw
    reset_robot_pose(robot, position=(start_xy[0], start_xy[1], 0.04), yaw=start_yaw)

    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    max_steps = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))
    pose = (start_xy[0], start_xy[1], start_yaw)

    if omega_tracker is not None:
        omega_tracker.reset()
    write_kinematic_state(robot, wheel_joint_ids, arm_joint_ids, pose, (0.0, 0.0, 0.0))
    ok, pose = step_kinematic_n(
        sim,
        robot,
        camera,
        wheel_joint_ids,
        arm_joint_ids,
        pose,
        (0.0, 0.0, 0.0),
        max(1, args_cli.settle_steps),
        physics_dt,
        viewport,
        active_view,
    )
    if not ok:
        raise RuntimeError("Simulation app closed during reset settle.")

    warm_std = 0.0
    camera_ready = True
    if not skip_camera_warmup:
        camera_ready = False
        for _ in range(max(1, args_cli.camera_warmup_steps)):
            ok, pose = step_kinematic_n(
                sim,
                robot,
                camera,
                wheel_joint_ids,
                arm_joint_ids,
                pose,
                (0.0, 0.0, 0.0),
                1,
                physics_dt,
                viewport,
                active_view,
            )
            if not ok:
                raise RuntimeError("Simulation app closed during camera warmup.")
            try:
                warm_std = float(np.asarray(rgb_frame(camera), dtype=np.float32).std())
            except Exception:
                warm_std = 0.0
            if warm_std >= args_cli.min_image_std:
                camera_ready = True
                break
        if overhead_camera is not None:
            overhead_camera.update(dt=physics_dt)
    if not camera_ready:
        print(f"[WARN] Camera warmup std stayed at {warm_std:.2f}; continuing anyway.", flush=True)

    frames: list[EpisodeFrame] = []
    track_errors: list[float] = []
    body_speeds: list[float] = []
    image_stds: list[float] = []
    action_history: list[np.ndarray] = []
    prev_action = np.zeros(3, dtype=np.float32)
    lap_progress = 0.0
    best_progress_m = 0.0
    last_progress_time = 0.0
    commanded_forward_distance = 0.0
    terminal_reason = "timeout"
    success = False
    last_print_at = -1.0
    max_command = np.array([0.45, 0.35, 2.0], dtype=np.float32)
    pos_xy = robot.data.root_pos_w[:, :2]
    _nearest_xy, _nearest_tangent, track_error_t, phase_t = compute_square_track_frame(pos_xy, scene_cfg.square_half_extent)
    start_phase = float(phase_t[0].item())
    previous_phase = start_phase
    start_yaw_actual = pose[2]
    external_frames: list[np.ndarray] = []

    print(
        f"[INFO] Starting {route.label} run from ({route.start_xy[0]:+.2f}, {route.start_xy[1]:+.2f}) "
        f"with continuous square lookahead tracking.",
        flush=True,
    )

    for step_index in range(max_steps):
        if stop_flag.requested:
            terminal_reason = "interrupted"
            break
        if not simulation_app.is_running():
            terminal_reason = "app_closed"
            break

        pose = get_pose(robot)
        pos_xy = robot.data.root_pos_w[:, :2]
        _nearest_xy, _nearest_tangent, track_error_t, phase_t = compute_square_track_frame(pos_xy, scene_cfg.square_half_extent)
        track_error = float(track_error_t[0].item())
        phase = float(phase_t[0].item())
        lap_progress = min(1.5, lap_progress + signed_phase_delta(phase, previous_phase, direction))
        previous_phase = phase

        segment_t, _segment_progress_t = phase_to_segment_and_progress(phase_t)
        segment_index = int(segment_t[0].item())
        progress_m = lap_progress * route.length
        if track_error > args_cli.off_track_abort_distance:
            terminal_reason = "off_track"
            break

        current_yaw = pose[2]
        if (
            lap_progress >= args_cli.lap_completion_threshold
            and wrap_phase_error(phase, start_phase) <= args_cli.start_phase_tolerance
            and abs(wrap_to_pi(current_yaw - start_yaw_actual)) <= args_cli.start_yaw_tolerance
            and commanded_forward_distance >= route.length * args_cli.min_square_distance_ratio
        ):
            terminal_reason = "goal_reached"
            success = True
            best_progress_m = route.length
            break

        command_vec, yaw_error, _command_track_error, _command_phase = compute_full_square_command(robot, scene_cfg, direction)
        if rng is not None and noise_std > 0.0:
            # Inject noise on the body-frame command so the recorded trajectory wanders
            # slightly off the ideal line — gives the policy off-axis recovery examples.
            noise = rng.normal(0.0, noise_std, size=3).astype(np.float32)
            scaled = noise * np.array([0.45, 0.35, 2.0], dtype=np.float32)
            noisy = (
                float(command_vec[0]) + float(scaled[0]),
                float(command_vec[1]) + float(scaled[1]),
                float(command_vec[2]) + float(scaled[2]),
            )
            command_vec = (
                clamp(noisy[0], args_cli.min_forward_speed, args_cli.target_speed),
                clamp(noisy[1], -0.20, 0.20),
                clamp(noisy[2], -args_cli.square_max_wz, args_cli.square_max_wz),
            )
        dist_to_goal = max(0.0, route.length * max(args_cli.lap_completion_threshold - lap_progress, 0.0))

        if progress_m >= best_progress_m + args_cli.progress_epsilon:
            best_progress_m = progress_m
            last_progress_time = step_index * control_dt
        if (step_index * control_dt) >= args_cli.stuck_timeout and ((step_index * control_dt) - last_progress_time) >= args_cli.stuck_timeout:
            terminal_reason = "stuck"
            break

        body_vel = np.asarray(command_vec, dtype=np.float32)
        body_speeds.append(abs(float(command_vec[0])))
        track_errors.append(track_error)

        image = rgb_frame(camera)
        image_stds.append(float(np.asarray(image, dtype=np.float32).std()))

        command_np = np.asarray(command_vec, dtype=np.float32)
        commanded_forward_distance += max(0.0, float(command_np[0])) * control_dt
        action_np = np.clip(command_np / max_command, -1.0, 1.0)
        action_history.append(action_np)

        frames.append(
            EpisodeFrame(
                image_rgb=image,
                timestamp=float(step_index * control_dt),
                state=prev_action.copy(),
                action=action_np.copy(),
                command=command_np.copy(),
                body_velocity=body_vel,
                track_error=float(track_error),
                lap_progress=float(lap_progress),
            )
        )
        prev_action = action_np

        ok, pose = step_kinematic_n(
            sim,
            robot,
            camera,
            wheel_joint_ids,
            arm_joint_ids,
            pose,
            command_vec,
            substeps,
            physics_dt,
            viewport,
            active_view,
        )
        if not ok:
            terminal_reason = "app_closed"
            break

        if overhead_camera is not None:
            try:
                if args_cli.external_view == "chase":
                    base_pos = robot.data.root_pos_w[0]
                    base_quat = robot.data.root_quat_w[0]
                    eye_offset = torch.tensor(
                        [[-args_cli.chase_back, 0.0, args_cli.chase_up]],
                        dtype=torch.float32, device=robot.device,
                    )
                    target_offset = torch.tensor(
                        [[args_cli.chase_target_forward, 0.0, args_cli.chase_target_up]],
                        dtype=torch.float32, device=robot.device,
                    )
                    eye_world = (quat_apply(base_quat.unsqueeze(0), eye_offset) + base_pos.unsqueeze(0))
                    tgt_world = (quat_apply(base_quat.unsqueeze(0), target_offset) + base_pos.unsqueeze(0))
                    overhead_camera.set_world_poses_from_view(eye_world, tgt_world)
                overhead_camera.update(dt=substeps * physics_dt)
                ext = overhead_camera.data.output["rgb"]
                if ext is not None and ext.numel() > 0:
                    ext_np = ext[0, ..., :3].detach().cpu().numpy()
                    if ext_np.dtype != np.uint8:
                        ext_np = np.clip(ext_np * (255.0 if ext_np.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                    external_frames.append(ext_np)
            except Exception:
                pass

        episode_time = step_index * control_dt
        if episode_time - last_print_at >= 1.0:
            last_print_at = episode_time
            print(
                f"[INFO] {route.label:20s} t={episode_time:5.1f}s seg={(segment_index % 4) + 1}/4 "
                f"lap={lap_progress:0.3f} dist_cmd={commanded_forward_distance:0.2f} "
                f"track={track_error:0.3f} yaw_err={yaw_error:+0.2f} "
                f"pose=({pose[0]:+0.2f},{pose[1]:+0.2f},{pose[2]:+0.2f}) "
                f"cmd=[{command_vec[0]:+0.2f},{command_vec[1]:+0.2f},{command_vec[2]:+0.2f}]",
                flush=True,
            )

    step_kinematic_n(
        sim,
        robot,
        camera,
        wheel_joint_ids,
        arm_joint_ids,
        pose,
        (0.0, 0.0, 0.0),
        max(1, args_cli.cooldown_steps),
        physics_dt,
        viewport,
        active_view,
    )

    duration_s = len(frames) * control_dt
    mean_track_error = float(np.mean(track_errors)) if track_errors else float("inf")
    p90_track_error = float(np.quantile(track_errors, 0.9)) if track_errors else float("inf")
    max_track_error = float(np.max(track_errors)) if track_errors else float("inf")
    frames_over_010_ratio = float(np.mean(np.asarray(track_errors) > 0.10)) if track_errors else 1.0
    frames_over_015_ratio = float(np.mean(np.asarray(track_errors) > 0.15)) if track_errors else 1.0
    mean_image_std = float(np.mean(image_stds)) if image_stds else 0.0
    min_image_std_val = float(np.min(image_stds)) if image_stds else 0.0
    mean_speed = float(np.mean(body_speeds)) if body_speeds else 0.0
    action_array = np.asarray(action_history, dtype=np.float32) if action_history else np.zeros((0, 3), dtype=np.float32)
    mean_abs = np.mean(np.abs(action_array), axis=0) if len(action_array) > 0 else np.zeros(3, dtype=np.float32)
    mean_vy_vx = float(mean_abs[1] / max(float(mean_abs[0]), 1e-6))
    final_progress = 1.0 if success else (float(best_progress_m / max(route.length, 1e-6)) if frames else 0.0)

    result = EpisodeResult(
        direction=direction,
        task_name=direction,
        task_index=0 if direction == "clockwise" else 1,
        frames=frames,
        success=success,
        terminal_reason=terminal_reason,
        final_lap_progress=final_progress,
        mean_track_error=mean_track_error,
        p90_track_error=p90_track_error,
        max_track_error=max_track_error,
        frames_over_010_ratio=frames_over_010_ratio,
        frames_over_015_ratio=frames_over_015_ratio,
        mean_image_std=mean_image_std,
        min_image_std=min_image_std_val,
        mean_abs_action_vx=float(mean_abs[0]),
        mean_abs_action_vy=float(mean_abs[1]),
        mean_abs_action_wz=float(mean_abs[2]),
        mean_action_vy_vx_ratio=mean_vy_vx,
        mean_speed=mean_speed,
        duration_s=duration_s,
    )
    return result, external_frames


def main() -> None:
    scene_cfg = SquareTrackSceneCfg(
        square_half_extent=args_cli.square_half_extent,
        floor_half_extent=args_cli.floor_half_extent,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )
    route = resolve_route_path(scene_cfg.square_half_extent)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    physics_dt = float(args_cli.physics_dt)
    if route.mode == "full_square":
        physics_dt = min(control_dt, max(physics_dt, 1.0 / 30.0))
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    livestream_enabled = bool(getattr(args_cli, "livestream", 0))
    render_interval = substeps if args_cli.headless and not livestream_enabled else 1

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    design_square_loop_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    camera = build_robot_camera_sensor(width=args_cli.image_width, height=args_cli.image_height)
    overhead_camera = None
    if args_cli.record_external:
        overhead_camera = build_overhead_camera_sensor(
            width=args_cli.external_width, height=args_cli.external_height
        )

    sim.reset()
    camera.update(dt=0.0)
    if overhead_camera is not None:
        if args_cli.external_view == "overhead":
            eye = torch.tensor([[0.0, 0.0, float(args_cli.external_z)]], dtype=torch.float32, device=robot.device)
            target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=robot.device)
            overhead_camera.set_world_poses_from_view(eye, target)
        # For chase mode the per-step loop will set the pose; here just initialize.
        overhead_camera.update(dt=0.0)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)
    omega_tracker = None
    if args_cli.enable_omega_feedback:
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(args_cli.square_max_wz, args_cli.max_wz, 2.0),
            ),
        )

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
        tasks=(
            ("clockwise", "counterclockwise")
            if args_cli.mix_directions and route.mode == "full_square"
            else (route.direction_name,)
            if route.direction_name is not None
            else (route.label,)
        ),
        track_layout="square_full_loop" if len(route.segments) > 1 else "square_point_to_point",
        episode_definition="one_autonomous_full_square_lap" if len(route.segments) > 1 else "one_autonomous_point_to_point_run",
    )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print()
    print("=" * 60)
    print("  TurboPi Square Simple Recorder")
    print("=" * 60)
    print(f"  TurboPi USD    : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"  Output session : {writer.session_dir}")
    print(f"  Route          : {route.label}")
    print(f"  Segments       : {len(route.segments)}")
    print(f"  Start          : ({route.start_xy[0]:+.2f}, {route.start_xy[1]:+.2f})")
    print(f"  Goal           : ({route.goal_xy[0]:+.2f}, {route.goal_xy[1]:+.2f})")
    print(f"  Episodes       : {args_cli.num_episodes}")
    print(f"  Control rate   : {args_cli.control_hz:.1f} Hz")
    print(f"  Sim dt         : {physics_dt:.4f} s")
    print(f"  Control steps  : {substeps}")
    print(f"  Target speed   : {args_cli.target_speed:.2f} m/s")
    print(f"  Heading gain   : {args_cli.drive_heading_gain:.2f}")
    print(f"  Wz cap         : {args_cli.max_wz:.2f} rad/s")
    if omega_tracker is not None:
        print(f"  Omega ctrl     : enabled (gain={args_cli.omega_feedback_gain:.2f}, alpha={args_cli.omega_measure_alpha:.2f})")
    else:
        print("  Omega ctrl     : off")
    print(f"  Goal tol       : {args_cli.position_tolerance:.02f} m")
    print(f"  Abort gates    : off-track {args_cli.off_track_abort_distance:.02f} m, stuck {args_cli.stuck_timeout:.1f} s")
    print(f"  Start view     : {active_view}")
    print()

    saved = 0
    attempts = 0
    rng = np.random.default_rng(int(args_cli.seed))
    base_routes = {}
    if args_cli.mix_directions and route.mode == "full_square":
        for dir_name in ("counterclockwise", "clockwise"):
            saved_route = args_cli.route
            args_cli.route = "square_ccw" if dir_name == "counterclockwise" else "square_cw"
            base_routes[dir_name] = resolve_route_path(scene_cfg.square_half_extent)
            args_cli.route = saved_route
    try:
        while saved < args_cli.num_episodes and simulation_app.is_running() and not stop_flag.requested:
            attempts += 1
            ep_route = route
            ep_direction = route.direction_name
            if args_cli.mix_directions and base_routes:
                ep_direction = "counterclockwise" if (saved % 2 == 0) else "clockwise"
                ep_route = base_routes[ep_direction]
            start_override = None
            if args_cli.randomize_start and ep_route.mode == "full_square":
                pose_xy, pose_yaw, _ = pick_random_start_pose(
                    scene_cfg,
                    ep_direction or "counterclockwise",
                    rng,
                    lateral_jitter=args_cli.start_lateral_jitter,
                    yaw_jitter_deg=args_cli.start_yaw_jitter_deg,
                )
                start_override = (pose_xy, pose_yaw)
            result, external_frames = run_episode(
                sim=sim,
                robot=robot,
                camera=camera,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                viewport=viewport,
                active_view=active_view,
                scene_cfg=scene_cfg,
                route=ep_route,
                stop_flag=stop_flag,
                omega_tracker=omega_tracker,
                overhead_camera=overhead_camera,
                rng=rng,
                noise_std=args_cli.action_noise_std,
                skip_camera_warmup=(attempts > 1),
                start_pose_override=start_override,
            )
            if result.success and result.frames:
                episode_dir = writer.save_episode(saved, result)
                if external_frames and overhead_camera is not None:
                    try:
                        import cv2 as _cv2

                        ext_path = episode_dir / "external.mp4"
                        h, w = external_frames[0].shape[:2]
                        ext_writer = _cv2.VideoWriter(
                            str(ext_path),
                            _cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args_cli.control_hz),
                            (w, h),
                        )
                        if ext_writer.isOpened():
                            for ext in external_frames:
                                ext_writer.write(_cv2.cvtColor(ext, _cv2.COLOR_RGB2BGR))
                            ext_writer.release()
                            print(f"[INFO] External MP4   : {ext_path}", flush=True)
                        else:
                            ext_writer.release()
                    except Exception as exc:
                        print(f"[WARN] External MP4 write failed: {exc}", flush=True)
                saved += 1
                print(
                    f"[INFO] Saved episode_{saved - 1:05d} [{ep_route.label}/{ep_direction}] frames={len(result.frames)} "
                    f"progress={result.final_lap_progress:.2f} mean_err={result.mean_track_error:.3f} -> {episode_dir}",
                    flush=True,
                )
            else:
                writer.record_failure()
                print(
                    f"[WARN] Attempt {attempts} failed [{ep_route.label}/{ep_direction}] reason={result.terminal_reason} "
                    f"frames={len(result.frames)} progress={result.final_lap_progress:.2f}",
                    flush=True,
                )
    finally:
        print()
        print(f"[INFO] Session complete : {writer.session_dir}", flush=True)
        print(f"[INFO] Saved episodes   : {saved}", flush=True)


def close_app_and_exit(exit_code: int = 0) -> None:
    """Best-effort app shutdown that refuses to hang forever inside Kit teardown."""
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
