"""Standalone TurboPi viewer for the procedural mountain cliff road scene."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch TurboPi on a realistic procedural mountain cliff road.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("isometric", "overview", "chase", "robot"),
    default="isometric",
    help="Initial viewport mode.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds. 0 runs until closed.")
parser.add_argument("--road_width", type=float, default=0.48, help="Drivable mountain-road width in meters.")
parser.add_argument("--road_z", type=float, default=0.82, help="Road height above the valley floor.")
parser.add_argument("--lower_terrain_z", type=float, default=-0.42, help="Lower valley height.")
parser.add_argument("--save_frame", type=str, default=None, help="Optional PNG/JPG screenshot from the isometric camera.")
parser.add_argument("--save_frame_dir", type=str, default=None, help="Optional directory for all static preview presets.")
parser.add_argument("--save_video", type=str, default=None, help="Optional MP4 preview from the isometric camera.")
parser.add_argument(
    "--camera_preset",
    type=str,
    choices=("isometric", "road", "valley", "chase"),
    default="isometric",
    help="Static camera pose used by saved preview images/videos.",
)
parser.add_argument("--preview_width", type=int, default=1920)
parser.add_argument("--preview_height", type=int, default=1080)
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.save_frame or args_cli.save_frame_dir or args_cli.save_video)

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

from common import (
    PERSPECTIVE_CAMERA_PATH,
    TURBOPI_URDF,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    resolve_asset_usd,
    reset_robot_pose,
    set_robot_camera_mount,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from mountain_cliff_scene import MountainCliffSceneCfg, design_mountain_cliff_scene, start_pose

MOUNTAIN_CAMERA_POS = (0.080, 0.0, 0.030)
MOUNTAIN_CAMERA_ROT = (0.996195, 0.0, -0.087156, 0.0)


def build_preview_camera(*, width: int, height: int) -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/MountainCliffRoad/PreviewCamera",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
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


def set_isometric_camera(sim: sim_utils.SimulationContext, viewport, scene_cfg: MountainCliffSceneCfg) -> str:
    if viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
    sim.set_camera_view(
        eye=[3.10, -3.30, scene_cfg.road_z + 1.80],
        target=[0.35, 1.15, scene_cfg.road_z - 0.10],
    )
    return "isometric"


def preview_camera_pose(scene_cfg: MountainCliffSceneCfg, preset: str | None = None) -> tuple[list[float], list[float]]:
    preset = preset or args_cli.camera_preset
    if preset == "chase":
        start_position, start_yaw = start_pose(scene_cfg)
        cos_yaw = math.cos(start_yaw)
        sin_yaw = math.sin(start_yaw)

        def transform(offset: tuple[float, float, float]) -> list[float]:
            return [
                start_position[0] + cos_yaw * offset[0] - sin_yaw * offset[1],
                start_position[1] + sin_yaw * offset[0] + cos_yaw * offset[1],
                start_position[2] + offset[2],
            ]

        return transform((-1.65, -0.08, 0.72)), transform((0.85, 0.02, 0.08))
    if preset == "road":
        return [1.85, -1.65, scene_cfg.road_z + 0.78], [0.40, 1.55, scene_cfg.road_z + 0.02]
    if preset == "valley":
        return [3.10, -0.55, scene_cfg.road_z + 0.70], [0.10, 1.70, scene_cfg.lower_terrain_z + 0.28]
    return [3.10, -3.30, scene_cfg.road_z + 1.80], [0.35, 1.15, scene_cfg.road_z - 0.10]


def set_preview_camera_pose(
    camera: Camera,
    scene_cfg: MountainCliffSceneCfg,
    device: str,
    *,
    preset: str | None = None,
) -> None:
    eye, target = preview_camera_pose(scene_cfg, preset)
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=device),
        torch.tensor([target], dtype=torch.float32, device=device),
    )
    camera.update(dt=0.0)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = MountainCliffSceneCfg(
        road_width=args_cli.road_width,
        road_z=args_cli.road_z,
        lower_terrain_z=args_cli.lower_terrain_z,
    )
    design_mountain_cliff_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(MOUNTAIN_CAMERA_POS, MOUNTAIN_CAMERA_ROT)
    preview_camera = None
    if args_cli.save_frame or args_cli.save_frame_dir or args_cli.save_video:
        preview_camera = build_preview_camera(width=args_cli.preview_width, height=args_cli.preview_height)

    sim.reset()
    start_position, start_yaw = start_pose(scene_cfg)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    if preview_camera is not None:
        set_preview_camera_pose(preview_camera, scene_cfg, robot.device)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    idle_targets = twist_to_wheel_targets(torch.zeros((robot.num_instances, 3), device=robot.device), robot.device)

    viewport = get_viewport()
    if args_cli.view == "isometric":
        active_view = set_isometric_camera(sim, viewport, scene_cfg)
    else:
        active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    print(f"[INFO] TurboPi USD  : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"[INFO] TurboPi URDF : {TURBOPI_URDF}")
    print("[INFO] Scene        : procedural mountain cliff road")
    print(f"[INFO] Road width   : {scene_cfg.road_width:.2f} m")
    print(f"[INFO] Road height  : {scene_cfg.road_z:.2f} m")
    print(f"[INFO] Initial view : {active_view}")

    writer = None
    if args_cli.save_video:
        video_path = Path(args_cli.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(args_cli.video_fps), (args_cli.preview_width, args_cli.preview_height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")

    sim_dt = float(sim_cfg.dt)
    elapsed = 0.0
    frame_index = 0
    write_every = max(1, int(round((1.0 / sim_dt) / max(args_cli.video_fps, 1e-6))))
    saved_frame = False
    saved_frame_set = False

    try:
        while simulation_app.is_running():
            if not sim.is_playing():
                sim.play()

            robot.set_joint_velocity_target(idle_targets, joint_ids=wheel_joint_ids)
            hold_arm_posture(robot, arm_joint_ids)
            robot.write_data_to_sim()

            sim.step()
            robot.update(sim_dt)

            if active_view == "chase":
                update_chase_camera(robot, viewport)

            if preview_camera is not None:
                preview_camera.update(dt=sim_dt)
                if args_cli.save_frame_dir and not saved_frame_set and frame_index >= 8:
                    frame_dir = Path(args_cli.save_frame_dir)
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    for preset in ("isometric", "road", "valley", "chase"):
                        set_preview_camera_pose(preview_camera, scene_cfg, robot.device, preset=preset)
                        for _ in range(3):
                            sim.step()
                            robot.update(sim_dt)
                            preview_camera.update(dt=sim_dt)
                        frame_path = frame_dir / f"{preset}.png"
                        cv2.imwrite(str(frame_path), cv2.cvtColor(rgb_frame(preview_camera), cv2.COLOR_RGB2BGR))
                        print(f"[INFO] Saved preview frame: {frame_path}")
                    saved_frame_set = True
                if args_cli.save_frame and not saved_frame and frame_index >= 8:
                    frame_path = Path(args_cli.save_frame)
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(frame_path), cv2.cvtColor(rgb_frame(preview_camera), cv2.COLOR_RGB2BGR))
                    print(f"[INFO] Saved preview frame: {frame_path}")
                    saved_frame = True
                if writer is not None and frame_index % write_every == 0:
                    writer.write(cv2.cvtColor(rgb_frame(preview_camera), cv2.COLOR_RGB2BGR))

            frame_index += 1
            elapsed += sim_dt
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
            if args_cli.duration <= 0.0 and args_cli.save_frame and saved_frame and writer is None:
                break
            if args_cli.duration <= 0.0 and args_cli.save_frame_dir and saved_frame_set and writer is None:
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved preview video: {args_cli.save_video}")


if __name__ == "__main__":
    main()
    simulation_app.close()
