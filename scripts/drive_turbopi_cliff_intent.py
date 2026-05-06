"""Drive TurboPi in the cliff scene using a task-conditioned CNN checkpoint."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Run a task-conditioned TurboPi CNN policy in the cliff scene.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, choices=("go_left", "go_right"), default="go_left")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", type=str, choices=("isometric", "overview", "chase", "robot"), default="isometric")
parser.add_argument("--duration", type=float, default=0.0, help="Wall-clock simulation duration in seconds. 0 runs until closed.")
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--render_interval", type=int, default=0)
parser.add_argument("--control_mode", choices=("dynamic", "kinematic"), default="dynamic")
parser.add_argument("--camera_warmup_steps", type=int, default=18)
parser.add_argument("--settle_steps", type=int, default=24)
parser.add_argument("--min_image_std", type=float, default=8.0)
parser.add_argument("--road_length", type=float, default=2.20)
parser.add_argument("--road_width", type=float, default=0.28)
parser.add_argument("--rectangle_half_width", type=float, default=0.78)
parser.add_argument("--cliff_height", type=float, default=1.35)
parser.add_argument("--policy_device", default="auto")
parser.add_argument("--smoothing", type=float, default=0.65)
parser.add_argument("--vx_cap", type=float, default=0.45)
parser.add_argument("--vy_cap", type=float, default=0.35)
parser.add_argument("--omega_cap", type=float, default=2.0)
parser.add_argument("--min_vx", type=float, default=0.0)
parser.add_argument("--min_vy", type=float, default=0.0)
parser.add_argument("--min_omega", type=float, default=0.0)
parser.add_argument("--fall_margin", type=float, default=0.25, help="Fall detection margin below road height.")
parser.add_argument("--continue_after_fall", action="store_true", help="Keep recording until --duration even after a fall.")
parser.add_argument("--no_rollers", action="store_true")
parser.add_argument("--save_video", type=str, default=None)
parser.add_argument("--video_fps", type=float, default=0.0)
parser.add_argument("--save_external_video", type=str, default=None)
parser.add_argument("--external_width", type=int, default=1920)
parser.add_argument("--external_height", type=int, default=1080)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from cliff_scene import CliffRoadSceneCfg, design_cliff_road_scene, start_pose
from cnn_policy.intent_drive import IntentPolicyRuntime, IntentPolicyRuntimeConfig
from common import (
    PERSPECTIVE_CAMERA_PATH,
    ROBOT_CAMERA_PATH,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot_pose,
    resolve_asset_usd,
    set_robot_camera_mount,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)

CLIFF_CAMERA_POS = (0.080, 0.0, 0.030)
CLIFF_CAMERA_ROT = (0.996195, 0.0, -0.087156, 0.0)


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Stopping rollout.", flush=True)


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_sim_playing(sim: sim_utils.SimulationContext) -> None:
    if not sim.is_playing():
        sim.play()


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def build_robot_camera_sensor(*, width: int, height: int) -> Camera:
    camera_cfg = CameraCfg(
        prim_path=ROBOT_CAMERA_PATH,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=None,
    )
    return Camera(camera_cfg)


def build_external_camera_sensor(*, width: int, height: int, prim_path: str = "/World/CliffInferenceCamera") -> Camera:
    camera_cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )
    return Camera(camera_cfg)


def rgb_frame(camera: Camera) -> np.ndarray:
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


def write_camera_frame(video_writer, camera: Camera) -> None:
    if video_writer is None:
        return
    try:
        frame = rgb_frame(camera)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception:
        return


def set_isometric_camera(sim: sim_utils.SimulationContext, viewport, scene_cfg: CliffRoadSceneCfg) -> str:
    if viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
    z = scene_cfg.cliff_height
    sim.set_camera_view(eye=[1.85, -2.25, z + 1.10], target=[0.0, -0.15, z - 0.10])
    return "isometric"


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    return x, y, wrap_to_pi(float(yaw_t[0].item()))


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


def write_kinematic_state(
    robot,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    pose: tuple[float, float, float],
    command_vec: tuple[float, float, float],
    root_z: float,
) -> None:
    x, y, yaw = pose
    vx, vy, wz = command_vec
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = float(x)
    root_pose[:, 1] = float(y)
    root_pose[:, 2] = float(root_z)

    yaw_t = torch.full((robot.num_instances,), float(yaw), dtype=torch.float32, device=robot.device)
    zeros = torch.zeros_like(yaw_t)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((robot.num_instances, 6), dtype=torch.float32, device=robot.device))

    command_t = torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def apply_dynamic_command(robot, wheel_joint_ids, arm_joint_ids, command_vec: tuple[float, float, float]) -> None:
    command_t = torch.tensor([command_vec], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def step_control(
    *,
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    command_vec: tuple[float, float, float],
    substeps: int,
    physics_dt: float,
    viewport,
    active_view: str,
    control_mode: str,
    pose: tuple[float, float, float],
    root_z: float,
    external_camera=None,
    external_writer=None,
    external_write_every: int = 1,
) -> tuple[bool, tuple[float, float, float]]:
    current_pose = pose
    write_every = max(1, int(external_write_every))
    for substep_index in range(substeps):
        if not simulation_app.is_running():
            return False, current_pose
        ensure_sim_playing(sim)
        if control_mode == "kinematic":
            current_pose = integrate_body_pose(current_pose, command_vec, physics_dt)
            write_kinematic_state(robot, wheel_joint_ids, arm_joint_ids, current_pose, command_vec, root_z)
        else:
            apply_dynamic_command(robot, wheel_joint_ids, arm_joint_ids, command_vec)
        sim.step()
        robot.update(physics_dt)
        if external_camera is not None and substep_index % write_every == 0:
            external_camera.update(dt=physics_dt)
            write_camera_frame(external_writer, external_camera)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
        if control_mode == "dynamic":
            current_pose = get_pose(robot)
    camera.update(dt=substeps * physics_dt)
    return True, current_pose


def main() -> None:
    scene_cfg = CliffRoadSceneCfg(
        road_length=args_cli.road_length,
        road_width=args_cli.road_width,
        rectangle_half_width=args_cli.rectangle_half_width,
        cliff_height=args_cli.cliff_height,
    )
    runtime = IntentPolicyRuntime(
        args_cli.checkpoint,
        task=args_cli.task,
        device=args_cli.policy_device,
        runtime_cfg=IntentPolicyRuntimeConfig(
            smoothing=args_cli.smoothing,
            vx_cap=args_cli.vx_cap,
            vy_cap=args_cli.vy_cap,
            omega_cap=args_cli.omega_cap,
            min_vx=args_cli.min_vx,
            min_vy=args_cli.min_vy,
            min_omega=args_cli.min_omega,
        ),
    )

    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    render_interval = args_cli.render_interval
    if render_interval <= 0:
        render_interval = substeps if args_cli.headless and not bool(getattr(args_cli, "livestream", 0)) else 1

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    design_cliff_road_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(CLIFF_CAMERA_POS, CLIFF_CAMERA_ROT)
    camera = build_robot_camera_sensor(width=runtime.image_width, height=runtime.image_height)
    external_camera = None
    if args_cli.save_external_video:
        external_camera = build_external_camera_sensor(width=args_cli.external_width, height=args_cli.external_height)

    sim.reset()
    start_position, start_yaw = start_pose(scene_cfg)
    root_z = scene_cfg.cliff_height + scene_cfg.start_height
    reset_robot_pose(robot, position=(start_position[0], start_position[1], root_z), yaw=start_yaw)
    if external_camera is not None:
        external_camera.set_world_poses_from_view(
            torch.tensor([[1.85, -2.25, scene_cfg.cliff_height + 1.10]], dtype=torch.float32, device=robot.device),
            torch.tensor([[0.0, -0.15, scene_cfg.cliff_height - 0.10]], dtype=torch.float32, device=robot.device),
        )
        external_camera.update(dt=0.0)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    if args_cli.view == "isometric":
        active_view = set_isometric_camera(sim, viewport, scene_cfg)
    else:
        active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    pose = (float(start_position[0]), float(start_position[1]), float(start_yaw))
    camera.update(dt=0.0)
    for _ in range(max(1, args_cli.settle_steps)):
        ok, pose = step_control(
            sim=sim,
            robot=robot,
            camera=camera,
            external_camera=None,
            external_writer=None,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            command_vec=(0.0, 0.0, 0.0),
            substeps=1,
            physics_dt=physics_dt,
            viewport=viewport,
            active_view=active_view,
            control_mode=args_cli.control_mode,
            pose=pose,
            root_z=root_z,
        )
        if not ok:
            return

    initial_frame = None
    warm_std = 0.0
    for _ in range(max(1, args_cli.camera_warmup_steps)):
        ok, pose = step_control(
            sim=sim,
            robot=robot,
            camera=camera,
            external_camera=None,
            external_writer=None,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            command_vec=(0.0, 0.0, 0.0),
            substeps=1,
            physics_dt=physics_dt,
            viewport=viewport,
            active_view=active_view,
            control_mode=args_cli.control_mode,
            pose=pose,
            root_z=root_z,
        )
        if not ok:
            return
        try:
            initial_frame = rgb_frame(camera)
            warm_std = image_std_rgb(initial_frame)
        except Exception:
            warm_std = 0.0
        if warm_std >= args_cli.min_image_std:
            break
    if initial_frame is None:
        raise RuntimeError("Camera did not produce an RGB frame during warmup.")
    if warm_std < args_cli.min_image_std:
        print(f"[WARN] Camera warmup std stayed at {warm_std:.2f}; continuing anyway.", flush=True)
    runtime.reset(initial_frame, task=args_cli.task)

    video_writer = None
    if args_cli.save_video:
        video_path = Path(args_cli.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(args_cli.video_fps if args_cli.video_fps > 0.0 else args_cli.control_hz)
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (runtime.image_width, runtime.image_height),
        )
        if not video_writer.isOpened():
            video_writer.release()
            video_writer = None
            print(f"[WARN] Could not open video writer for {video_path}", flush=True)

    external_writer = None
    external_write_every = 1
    if args_cli.save_external_video and external_camera is not None:
        external_path = Path(args_cli.save_external_video)
        external_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(args_cli.video_fps if args_cli.video_fps > 0.0 else args_cli.control_hz)
        physics_fps = 1.0 / max(physics_dt, 1e-9)
        external_write_every = max(1, int(round(physics_fps / max(fps, 1e-6))))
        external_writer = cv2.VideoWriter(
            str(external_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (args_cli.external_width, args_cli.external_height),
        )
        if not external_writer.isOpened():
            external_writer.release()
            external_writer = None
            print(f"[WARN] Could not open external video writer for {external_path}", flush=True)
        else:
            print(
                f"[INFO] Recording external MP4 to {external_path} "
                f"({args_cli.external_width}x{args_cli.external_height})",
                flush=True,
            )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print()
    print("=" * 60)
    print("  TurboPi Cliff Intent CNN Driver")
    print("=" * 60)
    print(f"  TurboPi USD  : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"  Checkpoint   : {args_cli.checkpoint}")
    print(f"  Task         : {args_cli.task}")
    print(f"  Control mode : {args_cli.control_mode}")
    print(f"  Control rate : {args_cli.control_hz:.1f} Hz")
    print(f"  Image        : {runtime.image_width}x{runtime.image_height}")
    print(f"  Initial view : {active_view}")
    print()

    elapsed = 0.0
    step_index = 0
    fall_reported = False
    try:
        while simulation_app.is_running() and not stop_flag.requested:
            frame = rgb_frame(camera)
            pred, smoothed, command = runtime.predict(frame)
            command_vec = tuple(float(v) for v in command.tolist())

            if video_writer is not None:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            ok, pose = step_control(
                sim=sim,
                robot=robot,
                camera=camera,
                external_camera=external_camera,
                external_writer=external_writer,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                command_vec=command_vec,
                substeps=substeps,
                physics_dt=physics_dt,
                viewport=viewport,
                active_view=active_view,
                control_mode=args_cli.control_mode,
                pose=pose,
                root_z=root_z,
                external_write_every=external_write_every,
            )
            if not ok:
                break

            z = float(robot.data.root_pos_w[0, 2].item())
            if z < scene_cfg.cliff_height - args_cli.fall_margin:
                if not fall_reported:
                    action = "Continuing recording." if args_cli.continue_after_fall else "Stopping."
                    print(f"[INFO] Robot fell below road height: z={z:.2f}. {action}", flush=True)
                    fall_reported = True
                if not args_cli.continue_after_fall:
                    break

            elapsed += control_dt
            step_index += 1
            if step_index % max(1, int(round(args_cli.control_hz))) == 0:
                print(
                    f"[INFO] t={elapsed:5.1f}s pose=({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f}) "
                    f"pred=[{pred[0]:+.2f},{pred[1]:+.2f},{pred[2]:+.2f}] "
                    f"cmd=[{command_vec[0]:+.2f},{command_vec[1]:+.2f},{command_vec[2]:+.2f}]",
                    flush=True,
                )

            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
    finally:
        if video_writer is not None:
            video_writer.release()
        if external_writer is not None:
            external_writer.release()


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
