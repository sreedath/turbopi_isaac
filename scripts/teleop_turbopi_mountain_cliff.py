"""Keyboard teleoperation for TurboPi in the procedural mountain cliff scene."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive TurboPi with keyboard teleoperation on the mountain cliff road.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="chase",
    help="Initial viewport mode.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Optional simulation duration in seconds. Use 0 to run until the app is closed.",
)
parser.add_argument("--road_width", type=float, default=0.48, help="Drivable mountain-road width in meters.")
parser.add_argument("--road_z", type=float, default=0.82, help="Road height above the valley floor.")
parser.add_argument("--lower_terrain_z", type=float, default=-0.42, help="Lower valley height.")
parser.add_argument("--vx", type=float, default=0.28, help="Forward/backward speed in m/s for a full key press.")
parser.add_argument("--vy", type=float, default=0.18, help="Lateral speed in m/s for a full key press.")
parser.add_argument("--wz", type=float, default=1.00, help="Yaw rate in rad/s for a full key press.")
parser.add_argument(
    "--command_filter",
    type=float,
    default=0.25,
    help="First-order smoothing factor in [0, 1]. Higher values track the keyboard more tightly.",
)
parser.add_argument(
    "--disable_omega_feedback",
    action="store_true",
    help="Send yaw commands directly without closed-loop compensation.",
)
parser.add_argument(
    "--omega_feedback_gain",
    type=float,
    default=0.6,
    help="Closed-loop yaw-rate feedback gain, additive on top of the desired wz.",
)
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured yaw rate.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.headless and not bool(getattr(args_cli, "livestream", 0)):
    raise RuntimeError("Keyboard teleop needs a GUI/livestream app window. Use --livestream 2 instead of --headless.")
if os.environ.get("DISPLAY") is None and not bool(getattr(args_cli, "livestream", 0)):
    print("[INFO] DISPLAY is not set. Keyboard teleop needs --livestream 2 in this remote session.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    activate_view_mode,
    cycle_view_mode,
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

MOUNTAIN_CAMERA_POS = (0.080, 0.0, 0.030)
MOUNTAIN_CAMERA_ROT = (0.996195, 0.0, -0.087156, 0.0)


def reset_to_mountain_start(robot, scene_cfg: MountainCliffSceneCfg) -> None:
    start_position, start_yaw = start_pose(scene_cfg)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
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

    sim.reset()
    reset_to_mountain_start(robot, scene_cfg)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    filtered_command = torch.zeros((robot.num_instances, 3), dtype=torch.float32, device=robot.device)
    omega_tracker = None
    if not args_cli.disable_omega_feedback:
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(args_cli.wz, 2.0),
            ),
        )

    viewport = get_viewport()
    state = {"view_mode": activate_view_mode(args_cli.view, sim, robot, viewport)}

    try:
        teleop = Se2Keyboard(
            Se2KeyboardCfg(
                sim_device=args_cli.device,
                v_x_sensitivity=args_cli.vx,
                v_y_sensitivity=args_cli.vy,
                omega_z_sensitivity=args_cli.wz,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create the keyboard teleop device. Run with a local GUI session or use --livestream 2."
        ) from exc
    teleop._INPUT_KEY_MAPPING.update(
        {
            "W": np.asarray([1.0, 0.0, 0.0]) * args_cli.vx,
            "S": np.asarray([-1.0, 0.0, 0.0]) * args_cli.vx,
            "Q": np.asarray([0.0, 1.0, 0.0]) * args_cli.vy,
            "E": np.asarray([0.0, -1.0, 0.0]) * args_cli.vy,
            "A": np.asarray([0.0, 0.0, 1.0]) * args_cli.wz,
            "D": np.asarray([0.0, 0.0, -1.0]) * args_cli.wz,
        }
    )

    def reset_cb() -> None:
        reset_to_mountain_start(robot, scene_cfg)
        filtered_command.zero_()
        if omega_tracker is not None:
            omega_tracker.reset()
        print("[INFO] Robot reset to mountain road start.")

    def cycle_camera_cb() -> None:
        state["view_mode"] = activate_view_mode(cycle_view_mode(state["view_mode"]), sim, robot, viewport)
        print(f"[INFO] View mode -> {state['view_mode']}")

    def close_cb() -> None:
        simulation_app.close()

    teleop.add_callback("R", reset_cb)
    teleop.add_callback("C", cycle_camera_cb)
    teleop.add_callback("ESCAPE", close_cb)
    teleop.reset()

    print(f"[INFO] TurboPi USD : {resolve_asset_usd(args_cli.asset_usd)}")
    print("[INFO] Scene       : procedural mountain cliff road")
    print(f"[INFO] Road width  : {scene_cfg.road_width:.2f} m")
    print("[INFO] Controls    : W/S drive, A/D yaw, Q/E strafe. Arrow/Numpad keys also work.")
    print("[INFO] Extra keys  : L zeroes command, R reset, C camera, ESC close.")
    print(f"[INFO] Initial view: {state['view_mode']}")
    if omega_tracker is not None:
        print(
            f"[INFO] Omega ctrl  : enabled (gain={args_cli.omega_feedback_gain:.2f}, "
            f"alpha={args_cli.omega_measure_alpha:.2f})"
        )
    print("[INFO] Livestream  : click once inside the Isaac viewport before using the keyboard in a remote session.")
    if viewport is None and args_cli.view != "overview":
        print("[INFO] No interactive viewport is available in this launch mode, so the script fell back to overview.")

    sim_dt = sim.get_physics_dt()
    elapsed = 0.0
    alpha = float(max(0.0, min(1.0, args_cli.command_filter)))

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.play()
            continue

        raw_command = teleop.advance().view(robot.num_instances, 3)
        filtered_command.mul_(1.0 - alpha).add_(alpha * raw_command)
        applied_command = filtered_command
        if omega_tracker is not None:
            applied_command = omega_tracker.compensate(
                filtered_command,
                robot.data.root_ang_vel_b[:, 2],
                dt=float(sim_dt),
                command_limit=args_cli.wz,
            )

        robot.set_joint_velocity_target(
            twist_to_wheel_targets(applied_command, robot.device),
            joint_ids=wheel_joint_ids,
        )
        hold_arm_posture(robot, arm_joint_ids)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)

        if state["view_mode"] == "chase":
            update_chase_camera(robot, viewport)

        elapsed += sim_dt
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
