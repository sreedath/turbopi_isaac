"""Run an ACT + language + CVAE checkpoint on the mountain cliff scene."""

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

parser = argparse.ArgumentParser(description="Drive TurboPi with a mountain ACT language checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", choices=("go_left", "go_right"), default="go_left")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", choices=("overview", "chase", "robot", "isometric"), default="chase")
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--control_mode", choices=("dynamic", "kinematic"), default="dynamic")
parser.add_argument("--policy_device", default="auto")
parser.add_argument("--vx_cap", type=float, default=0.45)
parser.add_argument("--vy_cap", type=float, default=0.35)
parser.add_argument("--wz_cap", type=float, default=2.0)
parser.add_argument("--smoothing", type=float, default=0.35)
parser.add_argument("--settle_steps", type=int, default=24)
parser.add_argument("--camera_warmup_steps", type=int, default=12)
parser.add_argument("--no_rollers", action="store_true")
parser.add_argument("--save_video", type=str, default=None)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_output_dir", type=str, default=None, help="Optional directory for multi-view inference MP4s.")
parser.add_argument("--video_width", type=int, default=1920)
parser.add_argument("--video_height", type=int, default=1080)
parser.add_argument(
    "--video_views",
    type=str,
    default="robot,chase,isometric",
    help="Comma-separated video views to record. Choices: robot,chase,isometric.",
)
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

from act_policy.runtime import ACTPolicyRuntime, ACTRuntimeConfig
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
from mountain_cliff_scene import MountainCliffSceneCfg, design_mountain_cliff_scene, start_pose

CAMERA_POS = (0.140, 0.0, 0.115)
CAMERA_ROT = (0.987688, 0.0, -0.156434, 0.0)
POLICY_CAMERA_PATH = "/World/TurboPiPolicyRobotCamera"
VIDEO_CAMERA_ROOT = "/World/TurboPiInferenceVideoCamera"
VIDEO_VIEWS = ("robot", "chase", "isometric")


class StopFlag:
    requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[drive] signal {signum}; stopping.", flush=True)


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    return x, y, wrap_to_pi(float(yaw_t[0].item()))


def build_camera(
    width: int,
    height: int,
    *,
    prim_path: str = POLICY_CAMERA_PATH,
    focal_length: float = 18.0,
) -> Camera:
    return Camera(
        CameraCfg(
            prim_path=prim_path,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.03, 100.0),
            ),
        )
    )


def parse_video_views(views_arg: str) -> tuple[str, ...]:
    views = tuple(view.strip() for view in views_arg.split(",") if view.strip())
    unknown = tuple(view for view in views if view not in VIDEO_VIEWS)
    if unknown:
        raise ValueError(f"Unknown video view(s): {', '.join(unknown)}. Valid views: {', '.join(VIDEO_VIEWS)}")
    if not views:
        raise ValueError("At least one video view must be selected.")
    return views


def build_video_cameras(width: int, height: int, views: tuple[str, ...]) -> dict[str, Camera]:
    return {
        view: build_camera(
            width,
            height,
            prim_path=f"{VIDEO_CAMERA_ROOT}_{view}",
            focal_length=20.0,
        )
        for view in views
    }


def update_policy_camera(camera: Camera, robot) -> None:
    base_pos = robot.data.root_pos_w[0]
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = float(yaw_t[0].item())
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    def to_world(offset: tuple[float, float, float]) -> list[float]:
        x, y, z = offset
        return [
            float(base_pos[0].item()) + cos_yaw * x - sin_yaw * y,
            float(base_pos[1].item()) + sin_yaw * x + cos_yaw * y,
            float(base_pos[2].item()) + z,
        ]

    eye = to_world((0.18, 0.0, 0.18))
    target = to_world((1.35, 0.0, 0.04))
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=robot.device),
        torch.tensor([target], dtype=torch.float32, device=robot.device),
    )


def camera_pose_from_robot(robot, eye_offset: tuple[float, float, float], target_offset: tuple[float, float, float]) -> tuple[list[float], list[float]]:
    base_pos = robot.data.root_pos_w[0]
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = float(yaw_t[0].item())
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    def to_world(offset: tuple[float, float, float]) -> list[float]:
        x, y, z = offset
        return [
            float(base_pos[0].item()) + cos_yaw * x - sin_yaw * y,
            float(base_pos[1].item()) + sin_yaw * x + cos_yaw * y,
            float(base_pos[2].item()) + z,
        ]

    return to_world(eye_offset), to_world(target_offset)


def set_camera_pose(camera: Camera, eye: list[float], target: list[float], device: str) -> None:
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=device),
        torch.tensor([target], dtype=torch.float32, device=device),
    )


def isometric_pose(scene_cfg: MountainCliffSceneCfg) -> tuple[list[float], list[float]]:
    return [3.10, -3.30, scene_cfg.road_z + 1.80], [0.35, 1.15, scene_cfg.road_z - 0.10]


def update_video_camera(camera: Camera, robot, scene_cfg: MountainCliffSceneCfg, view: str, dt: float) -> None:
    if view == "robot":
        eye, target = camera_pose_from_robot(robot, (0.18, 0.0, 0.18), (1.35, 0.0, 0.04))
    elif view == "chase":
        eye, target = camera_pose_from_robot(robot, (-1.65, -0.08, 0.72), (0.85, 0.02, 0.08))
    elif view == "isometric":
        eye, target = isometric_pose(scene_cfg)
    else:
        raise ValueError(f"Unknown video view: {view}")
    set_camera_pose(camera, eye, target, robot.device)
    camera.update(dt=dt)


def update_video_cameras(video_cameras: dict[str, Camera] | None, robot, scene_cfg: MountainCliffSceneCfg, dt: float) -> None:
    if not video_cameras:
        return
    for view, video_camera in video_cameras.items():
        update_video_camera(video_camera, robot, scene_cfg, view, dt)


def open_video_writers(task_name: str, views: tuple[str, ...]) -> dict[str, cv2.VideoWriter]:
    if not args_cli.video_output_dir:
        return {}
    video_dir = Path(args_cli.video_output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, cv2.VideoWriter] = {}
    for view in views:
        path = video_dir / f"mountain_act_inference_{task_name}_{view}_{args_cli.video_width}x{args_cli.video_height}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.video_fps),
            (args_cli.video_width, args_cli.video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        writers[view] = writer
        print(f"[drive-video] recording {view} -> {path}", flush=True)
    return writers


def write_video_frames(
    video_cameras: dict[str, Camera] | None,
    video_writers: dict[str, cv2.VideoWriter],
    robot,
    scene_cfg: MountainCliffSceneCfg,
    dt: float,
) -> None:
    if not video_cameras or not video_writers:
        return
    update_video_cameras(video_cameras, robot, scene_cfg, dt)
    for view, writer in video_writers.items():
        writer.write(cv2.cvtColor(rgb_frame(video_cameras[view]), cv2.COLOR_RGB2BGR))


def close_video_writers(video_writers: dict[str, cv2.VideoWriter]) -> None:
    for writer in video_writers.values():
        writer.release()


def rgb_frame(camera: Camera) -> np.ndarray:
    image = camera.data.output["rgb"]
    if image is None or image.numel() == 0:
        raise RuntimeError("Camera has no RGB data yet.")
    rgb = image[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def integrate(pose, command, dt):
    x, y, yaw = pose
    vx, vy, wz = command[:3]
    yaw_mid = yaw + 0.5 * wz * dt
    return (
        x + (vx * np.cos(yaw_mid) - vy * np.sin(yaw_mid)) * dt,
        y + (vx * np.sin(yaw_mid) + vy * np.cos(yaw_mid)) * dt,
        wrap_to_pi(yaw + wz * dt),
    )


def write_kinematic(robot, wheel_joint_ids, arm_joint_ids, pose, command, root_z):
    x, y, yaw = pose
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = x
    root_pose[:, 1] = y
    root_pose[:, 2] = root_z
    yaw_t = torch.full((robot.num_instances,), yaw, dtype=torch.float32, device=robot.device)
    zeros = torch.zeros_like(yaw_t)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((robot.num_instances, 6), dtype=torch.float32, device=robot.device))
    command_t = torch.as_tensor(np.asarray(command[:3], dtype=np.float32)[None, :], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def apply_dynamic(robot, wheel_joint_ids, arm_joint_ids, command):
    command_t = torch.as_tensor(np.asarray(command[:3], dtype=np.float32)[None, :], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def main() -> None:
    runtime = ACTPolicyRuntime(
        args_cli.checkpoint,
        task=args_cli.task,
        device=args_cli.policy_device,
        runtime_cfg=ACTRuntimeConfig(vx_cap=args_cli.vx_cap, vy_cap=args_cli.vy_cap, wz_cap=args_cli.wz_cap, smoothing=args_cli.smoothing),
    )
    scene_cfg = MountainCliffSceneCfg()
    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = max(1, int(round(control_dt / physics_dt)))
    render_interval = substeps if args_cli.headless and not bool(getattr(args_cli, "livestream", 0)) else 1
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device))
    design_mountain_cliff_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(CAMERA_POS, CAMERA_ROT)
    camera = build_camera(runtime.image_width, runtime.image_height)
    video_views = parse_video_views(args_cli.video_views)
    video_cameras = build_video_cameras(args_cli.video_width, args_cli.video_height, video_views) if args_cli.video_output_dir else None
    sim.reset()
    start_position, start_yaw = start_pose(scene_cfg)
    root_z = scene_cfg.road_z + scene_cfg.start_height
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    sim.play()
    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode("overview" if args_cli.view == "isometric" else args_cli.view, sim, robot, viewport)
    if args_cli.view == "isometric" and viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
        sim.set_camera_view(eye=[1.75, -2.50, scene_cfg.road_z + 1.35], target=[0.75, 0.95, scene_cfg.road_z])
        active_view = "isometric"
    elif args_cli.view == "robot" and viewport is not None:
        viewport.set_active_camera(POLICY_CAMERA_PATH)
        active_view = "robot"
    pose = (float(start_position[0]), float(start_position[1]), float(start_yaw))
    for _ in range(max(1, args_cli.settle_steps + args_cli.camera_warmup_steps)):
        update_policy_camera(camera, robot)
        sim.step()
        robot.update(physics_dt)
        camera.update(dt=physics_dt)
        update_video_cameras(video_cameras, robot, scene_cfg, physics_dt)

    writer = None
    if args_cli.save_video:
        path = Path(args_cli.save_video)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps, (runtime.image_width, runtime.image_height))
    video_writers = open_video_writers(args_cli.task, video_views)
    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)
    print(f"[drive] checkpoint={args_cli.checkpoint} task={args_cli.task} params={runtime.model.parameter_count():,}")
    elapsed = 0.0
    try:
        while simulation_app.is_running() and not stop_flag.requested:
            frame = rgb_frame(camera)
            _raw, command = runtime.predict(frame)
            if writer is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            for _ in range(substeps):
                if args_cli.control_mode == "kinematic":
                    pose = integrate(pose, command, physics_dt)
                    write_kinematic(robot, wheel_joint_ids, arm_joint_ids, pose, command, root_z)
                else:
                    apply_dynamic(robot, wheel_joint_ids, arm_joint_ids, command)
                sim.step()
                robot.update(physics_dt)
                if active_view == "chase":
                    update_chase_camera(robot, viewport)
                if args_cli.control_mode == "dynamic":
                    pose = get_pose(robot)
            update_policy_camera(camera, robot)
            camera.update(dt=control_dt)
            write_video_frames(video_cameras, video_writers, robot, scene_cfg, control_dt)
            elapsed += control_dt
            if args_cli.duration > 0 and elapsed >= args_cli.duration:
                break
    finally:
        if writer is not None:
            writer.release()
        close_video_writers(video_writers)


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
