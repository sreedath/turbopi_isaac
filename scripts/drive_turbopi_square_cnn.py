"""Drive the TurboPi in the square-loop arena using a trained CNN checkpoint."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Run a trained TurboPi CNN policy inside the square-loop arena.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained cnn_policy checkpoint.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode when a GUI or livestream is available.",
)
parser.add_argument(
    "--direction",
    type=str,
    choices=("clockwise", "counterclockwise"),
    default="clockwise",
    help="Start pose orientation for the rollout.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Optional wall-clock simulation duration in seconds.")
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0, help="Physics step in seconds.")
parser.add_argument("--control_hz", type=float, default=10.0, help="Control/update frequency in Hz.")
parser.add_argument(
    "--render_interval",
    type=int,
    default=0,
    help="Physics steps per render. 0 uses control-rate rendering for headless runs, every step otherwise.",
)
parser.add_argument(
    "--control_mode",
    type=str,
    choices=("kinematic", "dynamic"),
    default="kinematic",
    help="How to apply CNN commands. `kinematic` matches the simple square recorder and is the default.",
)
parser.add_argument("--camera_warmup_steps", type=int, default=18, help="Zero-command steps used to warm up the robot camera after reset.")
parser.add_argument("--min_image_std", type=float, default=8.0, help="Warn and wait if the camera image stream is too flat or washed out.")
parser.add_argument("--settle_steps", type=int, default=24, help="Zero-command steps after reset before policy control begins.")
parser.add_argument("--square_half_extent", type=float, default=0.45, help="Half-size of the taped square path in meters.")
parser.add_argument("--floor_half_extent", type=float, default=1.40, help="Half-size of the visible arena floor in meters.")
parser.add_argument("--tape_width", type=float, default=0.08, help="Width of the square path marker in meters.")
parser.add_argument("--wall_height", type=float, default=0.55, help="Outer boundary wall height in meters.")
parser.add_argument("--wall_thickness", type=float, default=0.04, help="Outer boundary wall thickness in meters.")
parser.add_argument("--policy_device", default="auto", help="Torch device for the CNN checkpoint.")
parser.add_argument("--smoothing", type=float, default=0.65, help="EMA factor applied to the previous normalized action.")
parser.add_argument("--vx_cap", type=float, default=0.45, help="Forward command cap used to denormalize model output.")
parser.add_argument("--vy_cap", type=float, default=0.35, help="Lateral command cap used to denormalize model output.")
parser.add_argument("--omega_cap", type=float, default=2.0, help="Yaw-rate cap used to denormalize model output.")
parser.add_argument("--min_vx", type=float, default=0.0, help="Minimum absolute forward command floor when nonzero.")
parser.add_argument("--min_vy", type=float, default=0.0, help="Minimum absolute lateral command floor when nonzero.")
parser.add_argument("--min_omega", type=float, default=0.0, help="Minimum absolute yaw-rate command floor when nonzero.")
parser.add_argument("--reset_track_error", type=float, default=0.35, help="Auto-reset if track error exceeds this value when auto reset is enabled.")
parser.add_argument("--enable_auto_reset", dest="enable_auto_reset", action="store_true", default=True, help="Automatically reset and re-prime the policy after invalid/off-track states.")
parser.add_argument("--disable_auto_reset", dest="enable_auto_reset", action="store_false", help="Disable automatic reset on invalid/off-track states.")
parser.add_argument("--enable_omega_feedback", dest="enable_omega_feedback", action="store_true", default=True, help="Enable closed-loop yaw-rate compensation while driving the learned policy.")
parser.add_argument("--disable_omega_feedback", dest="enable_omega_feedback", action="store_false", help="Disable closed-loop yaw-rate compensation while driving the learned policy.")
parser.add_argument("--omega_feedback_gain", type=float, default=2.0, help="Closed-loop yaw-rate feedback gain.")
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured yaw rate in the compensator.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
parser.add_argument("--save_video", type=str, default=None, help="Optional MP4 path. When set, every CNN-input frame is written with a predicted-action overlay.")
parser.add_argument("--video_fps", type=float, default=0.0, help="MP4 fps. 0 means use --control_hz.")
parser.add_argument(
    "--save_external_video",
    type=str,
    default=None,
    help="Optional MP4 path for the static spectator camera (overhead/isometric, controlled by --external_view).",
)
parser.add_argument(
    "--save_chase_video",
    type=str,
    default=None,
    help="Optional MP4 path for a third-person chase camera that follows the robot.",
)
parser.add_argument("--chase_video_width", type=int, default=1920)
parser.add_argument("--chase_video_height", type=int, default=1080)
parser.add_argument("--external_width", type=int, default=1920, help="External MP4 width.")
parser.add_argument("--external_height", type=int, default=1080, help="External MP4 height.")
parser.add_argument("--external_z", type=float, default=2.5, help="Height (m) of the overhead spectator camera (overhead view only).")
parser.add_argument(
    "--external_view",
    type=str,
    choices=("overhead", "isometric", "chase"),
    default="isometric",
    help="`overhead` = top-down. `isometric` = static corner 3/4 view (default). `chase` = third-person follow.",
)
parser.add_argument("--iso_eye_x", type=float, default=1.55, help="Isometric eye X (m).")
parser.add_argument("--iso_eye_y", type=float, default=-1.55, help="Isometric eye Y (m).")
parser.add_argument("--iso_eye_z", type=float, default=1.10, help="Isometric eye Z (m).")
parser.add_argument("--chase_back", type=float, default=0.45)
parser.add_argument("--chase_up", type=float, default=0.30)
parser.add_argument("--chase_target_forward", type=float, default=0.30)
parser.add_argument("--chase_target_up", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering for CNN driving.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from cnn_policy.drive import LoopPolicyRuntime, PolicyRuntimeConfig
from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    resolve_asset_usd,
    reset_robot_pose,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from square_loop import (
    SquareTrackSceneCfg,
    build_overhead_camera_sensor,
    build_robot_camera_sensor,
    design_square_loop_scene,
    observe_track_state,
    start_pose_for_direction,
)


@dataclass(frozen=True)
class ResetResult:
    frame_rgb: np.ndarray
    image_std: float
    pose: tuple[float, float, float]


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing the current cleanup.", flush=True)


def ensure_sim_playing(sim: sim_utils.SimulationContext) -> None:
    if not sim.is_playing():
        sim.play()


def rgb_frame_from_camera(camera) -> np.ndarray:
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


def image_std_rgb(image_rgb: np.ndarray) -> float:
    return float(np.asarray(image_rgb, dtype=np.float32).std())


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def zero_command(device: str | torch.device) -> torch.Tensor:
    return torch.zeros(3, dtype=torch.float32, device=device)


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = wrap_to_pi(float(yaw_t[0].item()))
    return x, y, yaw


def write_kinematic_state(
    robot,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
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
    x += (vx * np.cos(yaw_mid) - vy * np.sin(yaw_mid)) * dt
    y += (vx * np.sin(yaw_mid) + vy * np.cos(yaw_mid)) * dt
    yaw = wrap_to_pi(yaw + wz * dt)
    return x, y, yaw


def apply_dynamic_body_command(
    robot,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    command: torch.Tensor,
    *,
    physics_dt: float,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> torch.Tensor:
    applied_command = command.view(1, 3)
    if omega_tracker is not None:
        applied_command = omega_tracker.compensate(
            applied_command,
            robot.data.root_ang_vel_b[:, 2],
            dt=physics_dt,
            command_limit=max_omega_command,
        )
    wheel_targets = twist_to_wheel_targets(applied_command, robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()
    return applied_command[0]


def step_dynamic_simulation(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    command: torch.Tensor,
    substeps: int,
    physics_dt: float,
    viewport,
    active_view: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> bool:
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False
        ensure_sim_playing(sim)
        apply_dynamic_body_command(
            robot,
            wheel_joint_ids,
            arm_joint_ids,
            command,
            physics_dt=physics_dt,
            omega_tracker=omega_tracker,
            max_omega_command=max_omega_command,
        )
        sim.step()
        robot.update(physics_dt)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True


def step_kinematic_simulation(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    pose: tuple[float, float, float],
    command: torch.Tensor,
    substeps: int,
    physics_dt: float,
    viewport,
    active_view: str,
) -> tuple[bool, tuple[float, float, float]]:
    command_vec = tuple(float(item) for item in command.detach().cpu().tolist())
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


def step_control(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    pose: tuple[float, float, float],
    command: torch.Tensor,
    substeps: int,
    physics_dt: float,
    viewport,
    active_view: str,
    control_mode: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> tuple[bool, tuple[float, float, float]]:
    if control_mode == "kinematic":
        return step_kinematic_simulation(
            sim=sim,
            robot=robot,
            camera=camera,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            pose=pose,
            command=command,
            substeps=substeps,
            physics_dt=physics_dt,
            viewport=viewport,
            active_view=active_view,
        )

    ok = step_dynamic_simulation(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        command=command,
        substeps=substeps,
        physics_dt=physics_dt,
        viewport=viewport,
        active_view=active_view,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
    )
    return ok, get_pose(robot)


def warm_up_camera(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    viewport,
    active_view: str,
    pose: tuple[float, float, float],
    control_mode: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
    warmup_steps: int,
    min_image_std: float,
) -> ResetResult:
    last_frame = None
    last_std = 0.0
    current_pose = pose
    for _ in range(max(1, warmup_steps)):
        ok, current_pose = step_control(
            sim=sim,
            robot=robot,
            camera=camera,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            pose=current_pose,
            command=zero_command(robot.device),
            substeps=1,
            physics_dt=args_cli.physics_dt,
            viewport=viewport,
            active_view=active_view,
            control_mode=control_mode,
            omega_tracker=omega_tracker,
            max_omega_command=max_omega_command,
        )
        if not ok:
            raise RuntimeError("Simulation app closed during camera warmup.")
        last_frame = rgb_frame_from_camera(camera)
        last_std = image_std_rgb(last_frame)
        if last_std >= min_image_std:
            return ResetResult(frame_rgb=last_frame, image_std=last_std, pose=current_pose)
    if last_frame is None:
        raise RuntimeError("Camera warmup produced no frames.")
    return ResetResult(frame_rgb=last_frame, image_std=last_std, pose=current_pose)


def reset_runtime(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    viewport,
    active_view: str,
    scene_cfg: SquareTrackSceneCfg,
    direction: str,
    policy: LoopPolicyRuntime,
    control_mode: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> ResetResult:
    start_position, start_yaw = start_pose_for_direction(scene_cfg, direction)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    pose = (start_position[0], start_position[1], start_yaw)
    policy.reset()
    if omega_tracker is not None:
        omega_tracker.reset()
    ok, pose = step_control(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        pose=pose,
        command=zero_command(robot.device),
        substeps=max(1, args_cli.settle_steps),
        physics_dt=args_cli.physics_dt,
        viewport=viewport,
        active_view=active_view,
        control_mode=control_mode,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
    )
    if not ok:
        raise RuntimeError("Simulation app closed during reset settle.")
    result = warm_up_camera(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        viewport=viewport,
        active_view=active_view,
        pose=pose,
        control_mode=control_mode,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
        warmup_steps=args_cli.camera_warmup_steps,
        min_image_std=args_cli.min_image_std,
    )
    policy.reset(result.frame_rgb)
    return result


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    policy = LoopPolicyRuntime(
        checkpoint_path,
        device=args_cli.policy_device,
        runtime_cfg=PolicyRuntimeConfig(
            smoothing=args_cli.smoothing,
            vx_cap=args_cli.vx_cap,
            vy_cap=args_cli.vy_cap,
            omega_cap=args_cli.omega_cap,
            min_vx=args_cli.min_vx,
            min_vy=args_cli.min_vy,
            min_omega=args_cli.min_omega,
        ),
    )

    control_substeps = steps_per_control(args_cli.physics_dt, args_cli.control_hz)
    if args_cli.render_interval > 0:
        render_interval = args_cli.render_interval
    else:
        livestream_enabled = bool(getattr(args_cli, "livestream", 0))
        render_interval = control_substeps if args_cli.headless and not livestream_enabled else 1

    scene_cfg = SquareTrackSceneCfg(
        square_half_extent=args_cli.square_half_extent,
        floor_half_extent=args_cli.floor_half_extent,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, render_interval=render_interval, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    design_square_loop_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    camera = build_robot_camera_sensor(width=policy.image_width, height=policy.image_height)
    overhead_camera = None
    if args_cli.save_external_video:
        overhead_camera = build_overhead_camera_sensor(
            width=args_cli.external_width, height=args_cli.external_height
        )
    chase_camera = None
    if args_cli.save_chase_video:
        chase_camera = build_overhead_camera_sensor(
            width=args_cli.chase_video_width,
            height=args_cli.chase_video_height,
            prim_path="/World/SpectatorChase",
        )

    sim.reset()
    camera.update(dt=0.0)
    if overhead_camera is not None:
        if args_cli.external_view == "overhead":
            eye = torch.tensor([[0.0, 0.0, float(args_cli.external_z)]], dtype=torch.float32, device=robot.device)
            target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=robot.device)
            overhead_camera.set_world_poses_from_view(eye, target)
        elif args_cli.external_view == "isometric":
            eye = torch.tensor([[args_cli.iso_eye_x, args_cli.iso_eye_y, args_cli.iso_eye_z]],
                               dtype=torch.float32, device=robot.device)
            target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=robot.device)
            overhead_camera.set_world_poses_from_view(eye, target)
        # chase mode is handled per-step in _write_external_frame
        overhead_camera.update(dt=0.0)
    if chase_camera is not None:
        chase_camera.update(dt=0.0)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    omega_tracker = None
    if args_cli.enable_omega_feedback and args_cli.control_mode == "dynamic":
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(args_cli.omega_cap, 2.0),
            ),
        )

    reset_result = reset_runtime(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        viewport=viewport,
        active_view=active_view,
        scene_cfg=scene_cfg,
        direction=args_cli.direction,
        policy=policy,
        control_mode=args_cli.control_mode,
        omega_tracker=omega_tracker,
        max_omega_command=args_cli.omega_cap,
    )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    payload_epoch = policy.payload.get("epoch")
    print()
    print("=" * 60)
    print("  TurboPi CNN Inference Driver")
    print("=" * 60)
    print(f"  Checkpoint       : {checkpoint_path}")
    print(f"  Checkpoint epoch : {payload_epoch}")
    print(f"  Policy device    : {policy.device}")
    print(f"  Model input      : {policy.frame_history} frames @ {policy.image_width}x{policy.image_height}")
    print(f"  Start direction  : {args_cli.direction}")
    print(f"  Start view       : {active_view}")
    print(f"  Control mode     : {args_cli.control_mode}")
    print(f"  Render interval  : {render_interval} physics steps")
    print(
        f"  Caps             : vx={args_cli.vx_cap:.2f} vy={args_cli.vy_cap:.2f} "
        f"omega={args_cli.omega_cap:.2f}"
    )
    print(f"  Smoothing        : {args_cli.smoothing:.2f}")
    print(f"  Auto reset       : {'on' if args_cli.enable_auto_reset else 'off'}")
    print(f"  Warmup image std : {reset_result.image_std:.1f}")
    if omega_tracker is not None:
        print(
            f"  Omega ctrl       : enabled (gain={args_cli.omega_feedback_gain:.2f}, "
            f"alpha={args_cli.omega_measure_alpha:.2f})"
        )
    elif args_cli.control_mode == "kinematic" and args_cli.enable_omega_feedback:
        print("  Omega ctrl       : off (kinematic mode matches training)")
    else:
        print("  Omega ctrl       : off")
    print()

    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    elapsed = 0.0
    pose = reset_result.pose
    last_print_at = -1.0
    last_low_image_warn_at = -1.0
    reset_count = 0

    video_writer = None
    video_fps = args_cli.video_fps if args_cli.video_fps > 0 else args_cli.control_hz
    if args_cli.save_video:
        import cv2 as _cv2
        Path(args_cli.save_video).parent.mkdir(parents=True, exist_ok=True)
        first_frame = reset_result.frame_rgb
        h, w = first_frame.shape[:2]
        video_writer = _cv2.VideoWriter(
            args_cli.save_video,
            _cv2.VideoWriter_fourcc(*"mp4v"),
            float(video_fps),
            (w, h),
        )
        if not video_writer.isOpened():
            video_writer = None
            print(f"[WARN] Could not open MP4 writer at {args_cli.save_video}; inference video disabled.", flush=True)
        else:
            print(f"[INFO] Recording inference MP4 to {args_cli.save_video} ({w}x{h} @ {video_fps:.1f} fps)", flush=True)

    chase_writer = None
    if args_cli.save_chase_video and chase_camera is not None:
        import cv2 as _cv2
        Path(args_cli.save_chase_video).parent.mkdir(parents=True, exist_ok=True)
        chase_writer = _cv2.VideoWriter(
            args_cli.save_chase_video,
            _cv2.VideoWriter_fourcc(*"mp4v"),
            float(video_fps),
            (args_cli.chase_video_width, args_cli.chase_video_height),
        )
        if not chase_writer.isOpened():
            chase_writer = None
            print(f"[WARN] Could not open chase MP4 writer at {args_cli.save_chase_video}; chase video disabled.", flush=True)
        else:
            print(
                f"[INFO] Recording chase MP4 to {args_cli.save_chase_video} "
                f"({args_cli.chase_video_width}x{args_cli.chase_video_height} @ {video_fps:.1f} fps)",
                flush=True,
            )

    external_writer = None
    if args_cli.save_external_video and overhead_camera is not None:
        import cv2 as _cv2
        Path(args_cli.save_external_video).parent.mkdir(parents=True, exist_ok=True)
        external_writer = _cv2.VideoWriter(
            args_cli.save_external_video,
            _cv2.VideoWriter_fourcc(*"mp4v"),
            float(video_fps),
            (args_cli.external_width, args_cli.external_height),
        )
        if not external_writer.isOpened():
            external_writer = None
            print(
                f"[WARN] Could not open external MP4 writer at {args_cli.save_external_video}; external video disabled.",
                flush=True,
            )
        else:
            print(
                f"[INFO] Recording external MP4 to {args_cli.save_external_video} "
                f"({args_cli.external_width}x{args_cli.external_height} @ {video_fps:.1f} fps)",
                flush=True,
            )

    def _write_external_frame():
        if external_writer is None or overhead_camera is None:
            return
        import cv2 as _cv2
        try:
            if args_cli.external_view == "chase":
                base_pos = robot.data.root_pos_w[0]
                base_quat = robot.data.root_quat_w[0]
                eye_off = torch.tensor([[-args_cli.chase_back, 0.0, args_cli.chase_up]],
                                       dtype=torch.float32, device=robot.device)
                tgt_off = torch.tensor([[args_cli.chase_target_forward, 0.0, args_cli.chase_target_up]],
                                       dtype=torch.float32, device=robot.device)
                eye_w = quat_apply(base_quat.unsqueeze(0), eye_off) + base_pos.unsqueeze(0)
                tgt_w = quat_apply(base_quat.unsqueeze(0), tgt_off) + base_pos.unsqueeze(0)
                overhead_camera.set_world_poses_from_view(eye_w, tgt_w)
            overhead_camera.update(dt=control_dt)
            ext = overhead_camera.data.output.get("rgb")
            if ext is None or ext.numel() == 0:
                return
            ext_np = ext[0, ..., :3].detach().cpu().numpy()
            if ext_np.dtype != np.uint8:
                if np.issubdtype(ext_np.dtype, np.floating):
                    if ext_np.max() <= 1.0:
                        ext_np = ext_np * 255.0
                ext_np = np.clip(ext_np, 0, 255).astype(np.uint8)
            external_writer.write(_cv2.cvtColor(ext_np, _cv2.COLOR_RGB2BGR))
        except Exception:
            pass

    def _write_chase_frame():
        if chase_writer is None or chase_camera is None:
            return
        import cv2 as _cv2
        try:
            base_pos = robot.data.root_pos_w[0]
            base_quat = robot.data.root_quat_w[0]
            eye_off = torch.tensor([[-args_cli.chase_back, 0.0, args_cli.chase_up]],
                                   dtype=torch.float32, device=robot.device)
            tgt_off = torch.tensor([[args_cli.chase_target_forward, 0.0, args_cli.chase_target_up]],
                                   dtype=torch.float32, device=robot.device)
            eye_w = quat_apply(base_quat.unsqueeze(0), eye_off) + base_pos.unsqueeze(0)
            tgt_w = quat_apply(base_quat.unsqueeze(0), tgt_off) + base_pos.unsqueeze(0)
            chase_camera.set_world_poses_from_view(eye_w, tgt_w)
            chase_camera.update(dt=control_dt)
            ext = chase_camera.data.output.get("rgb")
            if ext is None or ext.numel() == 0:
                return
            ext_np = ext[0, ..., :3].detach().cpu().numpy()
            if ext_np.dtype != np.uint8:
                if np.issubdtype(ext_np.dtype, np.floating) and ext_np.max() <= 1.0:
                    ext_np = ext_np * 255.0
                ext_np = np.clip(ext_np, 0, 255).astype(np.uint8)
            chase_writer.write(_cv2.cvtColor(ext_np, _cv2.COLOR_RGB2BGR))
        except Exception:
            pass

    def _overlay_and_write(frame_rgb, pred_arr, command_arr, t, reset_count_val):
        if video_writer is None:
            return
        import cv2 as _cv2
        bgr = _cv2.cvtColor(frame_rgb, _cv2.COLOR_RGB2BGR).copy()
        text_lines = [
            f"t={t:5.1f}s fps={video_fps:.0f} resets={reset_count_val}",
            f"pred [vx,vy,wz]=[{pred_arr[0]:+.2f},{pred_arr[1]:+.2f},{pred_arr[2]:+.2f}]",
            f"cmd  [vx,vy,wz]=[{command_arr[0]:+.2f},{command_arr[1]:+.2f},{command_arr[2]:+.2f}]",
        ]
        for i, line in enumerate(text_lines):
            _cv2.putText(bgr, line, (4, 12 + 12 * i), _cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, _cv2.LINE_AA)
        video_writer.write(bgr)

    try:
        while simulation_app.is_running() and not stop_flag.requested:
            ensure_sim_playing(sim)
            tracking = observe_track_state(robot, scene_cfg)
            if tracking.has_nan or tracking.track_error > args_cli.reset_track_error:
                reason = "invalid_state" if tracking.has_nan else f"off_track({tracking.track_error:.3f} m)"
                print(f"[WARN] Policy rollout reset: {reason}", flush=True)
                if not args_cli.enable_auto_reset:
                    break
                reset_result = reset_runtime(
                    sim=sim,
                    robot=robot,
                    camera=camera,
                    wheel_joint_ids=wheel_joint_ids,
                    arm_joint_ids=arm_joint_ids,
                    viewport=viewport,
                    active_view=active_view,
                    scene_cfg=scene_cfg,
                    direction=args_cli.direction,
                    policy=policy,
                    control_mode=args_cli.control_mode,
                    omega_tracker=omega_tracker,
                    max_omega_command=args_cli.omega_cap,
                )
                pose = reset_result.pose
                reset_count += 1
                elapsed = 0.0
                last_print_at = -1.0
                print(f"[INFO] Policy reset complete. Warmup image std={reset_result.image_std:.1f}", flush=True)
                continue

            frame_rgb = rgb_frame_from_camera(camera)
            image_std = image_std_rgb(frame_rgb)
            if image_std < args_cli.min_image_std:
                if elapsed - last_low_image_warn_at >= 1.0:
                    last_low_image_warn_at = elapsed
                    print(
                        f"[WARN] Camera image std is low ({image_std:.1f}); continuing with the current frame.",
                        flush=True,
                    )

            pred, smoothed, command_np = policy.predict(frame_rgb)
            _overlay_and_write(frame_rgb, pred, command_np, elapsed, reset_count)
            _write_external_frame()
            _write_chase_frame()
            command_t = torch.as_tensor(command_np, dtype=torch.float32, device=robot.device)
            ok, pose = step_control(
                sim=sim,
                robot=robot,
                camera=camera,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                pose=pose,
                command=command_t,
                substeps=control_substeps,
                physics_dt=args_cli.physics_dt,
                viewport=viewport,
                active_view=active_view,
                control_mode=args_cli.control_mode,
                omega_tracker=omega_tracker,
                max_omega_command=args_cli.omega_cap,
            )
            if not ok:
                break

            if elapsed - last_print_at >= 1.0:
                last_print_at = elapsed
                print(
                    f"[INFO] t={elapsed:5.1f}s err={tracking.track_error:0.3f} phase={tracking.track_phase:0.3f} "
                    f"img_std={image_std:0.1f} pred=[{pred[0]:0.2f}, {pred[1]:0.2f}, {pred[2]:0.2f}] "
                    f"cmd=[{command_np[0]:0.2f}, {command_np[1]:0.2f}, {command_np[2]:0.2f}]",
                    flush=True,
                )

            elapsed += control_dt
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
    finally:
        try:
            _, pose = step_control(
                sim=sim,
                robot=robot,
                camera=camera,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                pose=pose,
                command=zero_command(robot.device),
                substeps=max(1, args_cli.settle_steps),
                physics_dt=args_cli.physics_dt,
                viewport=viewport,
                active_view=active_view,
                control_mode=args_cli.control_mode,
                omega_tracker=omega_tracker,
                max_omega_command=args_cli.omega_cap,
            )
        except Exception:
            pass
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Inference MP4 saved to {args_cli.save_video}")
        if external_writer is not None:
            external_writer.release()
            print(f"[INFO] External MP4 saved to {args_cli.save_external_video}")
        if chase_writer is not None:
            chase_writer.release()
            print(f"[INFO] Chase MP4 saved to {args_cli.save_chase_video}")
        print()
        print(f"[INFO] CNN drive finished after {elapsed:.1f}s with {reset_count} resets.")


if __name__ == "__main__":
    main()
    simulation_app.close()
