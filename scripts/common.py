from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from mecanum_builder import add_all_mecanum_rollers

ViewMode = Literal["overview", "chase", "robot"]


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = BUNDLE_ROOT / "assets" / "turbopi"
TURBOPI_USD = ASSET_ROOT / "turbopi.usd"
TURBOPI_URDF = ASSET_ROOT / "turbopi_description" / "urdf" / "turbopi.urdf"

ROBOT_PRIM_PATH = "/World/TurboPi"
ROBOT_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/camera_link/RobotCamera"
CHASE_CAMERA_PATH = "/World/TurboPiChaseCamera"
PERSPECTIVE_CAMERA_PATH = "/OmniverseKit_Persp"

CAMERA_LINK_TO_SENSOR_POS = (0.040, 0.0, 0.0)
CAMERA_LINK_TO_SENSOR_ROT = (0.965926, 0.0, -0.258819, 0.0)

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.130
WHEEL_BASE = 0.119
LW_SUM_HALF = (WHEEL_BASE + TRACK_WIDTH) / 2.0

WHEEL_FORWARD_SIGN = {
    "wheel_lf_joint": -1.0,
    "wheel_lb_joint": +1.0,
    "wheel_rf_joint": -1.0,
    "wheel_rb_joint": +1.0,
}

VIEW_MODES: tuple[ViewMode, ...] = ("overview", "chase", "robot")


@dataclass(frozen=True)
class OmegaTrackerCfg:
    """Closed-loop correction for commanded yaw rate.

    The compensator adds a proportional (optionally + integral) correction on
    top of the desired yaw command. With the corrected roller geometry the
    open-loop plant gain is roughly 1.0, so a small feedback gain is enough
    to soak up residual drift at corners without ringing on straights.
    """

    feedback_gain: float = 0.6
    measurement_alpha: float = 0.2
    integrator_gain: float = 0.0
    integrator_limit: float = 0.5
    command_limit: float = 2.0


class OmegaTracker:
    """Simple filtered-feedback compensator for body yaw-rate tracking."""

    def __init__(self, num_envs: int, device: str | torch.device, cfg: OmegaTrackerCfg | None = None):
        self.cfg = cfg or OmegaTrackerCfg()
        self.filtered_wz = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.integral = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            self.filtered_wz.zero_()
            self.integral.zero_()
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.filtered_wz.device)
        self.filtered_wz[env_ids_t] = 0.0
        self.integral[env_ids_t] = 0.0

    def compensate(
        self,
        desired_command: torch.Tensor | list[float] | tuple[float, float, float],
        measured_wz: torch.Tensor,
        *,
        dt: float,
        command_limit: float | None = None,
    ) -> torch.Tensor:
        command_t = torch.as_tensor(desired_command, dtype=torch.float32, device=self.filtered_wz.device)
        if command_t.ndim == 1:
            command_t = command_t.unsqueeze(0)
        measured_wz_t = torch.as_tensor(measured_wz, dtype=torch.float32, device=self.filtered_wz.device).view(-1)
        alpha = float(max(0.0, min(1.0, self.cfg.measurement_alpha)))
        self.filtered_wz.mul_(1.0 - alpha).add_(alpha * measured_wz_t)
        error = command_t[:, 2] - self.filtered_wz
        if self.cfg.integrator_gain > 0.0 and dt > 0.0:
            self.integral.add_(error * float(dt))
            self.integral.clamp_(-self.cfg.integrator_limit, self.cfg.integrator_limit)
        else:
            self.integral.zero_()

        corrected = command_t.clone()
        # Additive feedforward + feedback. The earlier version replaced the
        # command with just `K * error`, which only looked stable because the
        # pre-fix plant had roughly 2x gain and accidentally re-amplified the
        # signal. With the corrected plant (~1x gain) the replacement form
        # ringed on every corner, so feed the desired wz through as a
        # feedforward and only trim it with the feedback error.
        corrected[:, 2] = (
            command_t[:, 2]
            + self.cfg.feedback_gain * error
            + self.cfg.integrator_gain * self.integral
        )
        max_abs = float(self.cfg.command_limit if command_limit is None else command_limit)
        corrected[:, 2].clamp_(-max_abs, max_abs)
        return corrected


def resolve_asset_usd(asset_usd: str | None = None) -> Path:
    """Return the USD path to load for the standalone TurboPi bundle."""
    usd_path = Path(asset_usd).expanduser().resolve() if asset_usd else TURBOPI_USD
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"TurboPi USD not found: {usd_path}. Expected the bundled asset at {TURBOPI_USD} or an override via"
            " --asset_usd."
        )
    return usd_path


def build_turbopi_cfg(
    asset_usd: str | None = None,
    prim_path: str = ROBOT_PRIM_PATH,
    *,
    add_rollers: bool = True,
) -> ArticulationCfg:
    """Create the articulation config used by the standalone viewer and teleop scripts."""
    actuators = {
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            velocity_limit_sim=35.0,
            effort_limit_sim=0.22,
            stiffness=0.0,
            damping=20.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[23]"],
            velocity_limit_sim=3.0,
            effort_limit_sim=2.0,
            stiffness=10.0,
            damping=1.0,
        ),
    }
    if add_rollers:
        actuators["rollers"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_roller_.*_joint"],
            velocity_limit_sim=100.0,
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        )

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(resolve_asset_usd(asset_usd)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.04)),
        actuators=actuators,
    )


def design_basic_scene() -> None:
    """Spawn a simple floor and light rig around the TurboPi."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)


def _set_camera_common_attrs(camera: UsdGeom.Camera) -> None:
    camera.CreateFocalLengthAttr(13.0)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateHorizontalApertureAttr(10.0)
    camera.CreateVerticalApertureAttr(7.5)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    camera_prim = camera.GetPrim()
    coi_attr = camera_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_attr or not coi_attr.IsValid():
        camera_prim.CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
        ).Set(Gf.Vec3d(0.0, 0.0, -10.0))


def ensure_robot_camera(camera_path: str = ROBOT_CAMERA_PATH) -> str:
    """Create a robot-mounted camera prim under ``camera_link``."""
    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera_prim = camera.GetPrim()
    _set_camera_common_attrs(camera)
    set_robot_camera_mount(CAMERA_LINK_TO_SENSOR_POS, CAMERA_LINK_TO_SENSOR_ROT, camera_path=camera_path)
    return camera_path


def set_robot_camera_mount(
    pos: tuple[float, float, float],
    rot_wxyz: tuple[float, float, float, float],
    *,
    camera_path: str = ROBOT_CAMERA_PATH,
) -> None:
    """Set the robot camera prim transform relative to ``camera_link``."""
    stage = get_current_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera_prim = camera.GetPrim()
        _set_camera_common_attrs(camera)
    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xformable.AddOrientOp().Set(Gf.Quatf(*rot_wxyz))


def ensure_chase_camera(camera_path: str = CHASE_CAMERA_PATH) -> str:
    """Create the external follow camera used by teleop."""
    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, camera_path)
    _set_camera_common_attrs(camera)
    return camera_path


def spawn_turbopi(asset_usd: str | None = None, add_rollers: bool = True) -> Articulation:
    """Spawn the standalone TurboPi articulation and attach its helper cameras."""
    robot = Articulation(build_turbopi_cfg(asset_usd=asset_usd, add_rollers=add_rollers))
    stage = get_current_stage()
    if add_rollers:
        add_all_mecanum_rollers(ROBOT_PRIM_PATH, stage)
    ensure_robot_camera()
    ensure_chase_camera()
    return robot


def set_overview_camera(sim: sim_utils.SimulationContext) -> None:
    """Place the main viewport in a sane overview pose."""
    sim.set_camera_view(eye=[1.9, -1.9, 1.35], target=[0.0, 0.0, 0.08])


def get_viewport():
    """Return the main viewport if the current launch mode exposes one."""
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        return get_viewport_from_window_name("Viewport")
    except Exception:
        return None


def activate_view_mode(
    view_mode: ViewMode, sim: sim_utils.SimulationContext, robot: Articulation, viewport=None
) -> ViewMode:
    """Switch between overview, chase, and robot-mounted camera views."""
    if viewport is None:
        set_overview_camera(sim)
        return "overview"

    if view_mode == "overview":
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
        set_overview_camera(sim)
        return "overview"

    if view_mode == "robot":
        viewport.set_active_camera(ROBOT_CAMERA_PATH)
        return "robot"

    viewport.set_active_camera(CHASE_CAMERA_PATH)
    update_chase_camera(robot, viewport)
    return "chase"


def cycle_view_mode(current_mode: ViewMode) -> ViewMode:
    """Advance to the next camera mode."""
    current_index = VIEW_MODES.index(current_mode)
    return VIEW_MODES[(current_index + 1) % len(VIEW_MODES)]


def update_chase_camera(
    robot: Articulation,
    viewport,
    camera_path: str = CHASE_CAMERA_PATH,
    eye_offset: tuple[float, float, float] = (-1.65, -0.08, 0.72),
    target_offset: tuple[float, float, float] = (0.85, 0.02, 0.08),
    yaw_only: bool = True,
) -> None:
    """Update the chase camera to follow the robot base."""
    if viewport is None:
        return

    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    base_pos = robot.data.root_pos_w[0]
    base_quat = robot.data.root_quat_w[0]

    eye_offset_t = torch.tensor([eye_offset], dtype=torch.float32, device=robot.device)
    target_offset_t = torch.tensor([target_offset], dtype=torch.float32, device=robot.device)

    if yaw_only:
        w, x, y, z = [float(value.item()) for value in base_quat]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        eye_local = eye_offset_t[0]
        target_local = target_offset_t[0]
        eye_world = base_pos + torch.tensor(
            [
                cos_yaw * eye_local[0] - sin_yaw * eye_local[1],
                sin_yaw * eye_local[0] + cos_yaw * eye_local[1],
                eye_local[2],
            ],
            dtype=torch.float32,
            device=robot.device,
        )
        target_world = base_pos + torch.tensor(
            [
                cos_yaw * target_local[0] - sin_yaw * target_local[1],
                sin_yaw * target_local[0] + cos_yaw * target_local[1],
                target_local[2],
            ],
            dtype=torch.float32,
            device=robot.device,
        )
    else:
        eye_world = quat_apply(base_quat.unsqueeze(0), eye_offset_t)[0] + base_pos
        target_world = quat_apply(base_quat.unsqueeze(0), target_offset_t)[0] + base_pos

    camera_state = ViewportCameraState(camera_path, viewport)
    camera_state.set_position_world(
        Gf.Vec3d(float(eye_world[0]), float(eye_world[1]), float(eye_world[2])), True
    )
    camera_state.set_target_world(
        Gf.Vec3d(float(target_world[0]), float(target_world[1]), float(target_world[2])), True
    )


def _resolve_joint_ids(robot: Articulation, joint_names: list[str]) -> list[int]:
    joint_ids: list[int] = []
    for joint_name in joint_names:
        ids, _ = robot.find_joints(joint_name)
        joint_ids.append(int(ids[0]))
    return joint_ids


def get_wheel_joint_ids(robot: Articulation) -> list[int]:
    """Return wheel joint ids in a stable left-front to right-back order."""
    return _resolve_joint_ids(robot, ["wheel_lf_joint", "wheel_lb_joint", "wheel_rf_joint", "wheel_rb_joint"])


def get_arm_joint_ids(robot: Articulation) -> list[int]:
    """Return the arm joints that should be held in their default pose."""
    joint_ids, _ = robot.find_joints("joint[23]")
    return [int(joint_id) for joint_id in joint_ids]


def hold_arm_posture(robot: Articulation, arm_joint_ids: list[int]) -> None:
    """Keep the little camera mast from sagging while the base drives around."""
    if not arm_joint_ids:
        return
    robot.set_joint_position_target(robot.data.default_joint_pos[:, arm_joint_ids], joint_ids=arm_joint_ids)


def twist_to_wheel_targets(command: torch.Tensor | list[float] | tuple[float, float, float], device: str) -> torch.Tensor:
    """Convert body twist commands ``(vx, vy, wz)`` into mecanum wheel velocity targets.

    Note on the vy sign: empirically the Isaac Sim mecanum plant produces
    motion in the ``-vy`` direction for a ``+vy`` command when the rollers
    are in the standard LF=+45/RF=-45/LB=-45/RB=+45 X-pattern. Flipping the
    roller angles removes this inversion but breaks the IK's match to the
    yaw kinematic constraint, so the robot stalls mid-turn. The cheapest
    fix that keeps both yaw and strafe correct is to negate vy right here
    at the IK: +vy command still means "strafe body-left" from the caller's
    perspective, but the wheel targets are computed with the opposite sign.
    """
    command_t = torch.as_tensor(command, dtype=torch.float32, device=device)
    if command_t.ndim == 1:
        command_t = command_t.unsqueeze(0)

    vx = command_t[:, 0]
    vy = -command_t[:, 1]
    wz = command_t[:, 2]

    omega_lf = (vx - vy - LW_SUM_HALF * wz) / WHEEL_RADIUS
    omega_lb = (vx + vy - LW_SUM_HALF * wz) / WHEEL_RADIUS
    omega_rf = (vx + vy + LW_SUM_HALF * wz) / WHEEL_RADIUS
    omega_rb = (vx - vy + LW_SUM_HALF * wz) / WHEEL_RADIUS

    wheel_targets = torch.stack((omega_lf, omega_lb, omega_rf, omega_rb), dim=-1)
    wheel_signs = torch.tensor(
        [
            WHEEL_FORWARD_SIGN["wheel_lf_joint"],
            WHEEL_FORWARD_SIGN["wheel_lb_joint"],
            WHEEL_FORWARD_SIGN["wheel_rf_joint"],
            WHEEL_FORWARD_SIGN["wheel_rb_joint"],
        ],
        dtype=torch.float32,
        device=device,
    )
    return wheel_targets * wheel_signs


def reset_robot(robot: Articulation, position: tuple[float, float, float] = (0.0, 0.0, 0.04)) -> None:
    """Reset the robot root and joints to a clean startup pose."""
    reset_robot_pose(robot, position=position)


def reset_robot_pose(
    robot: Articulation,
    position: tuple[float, float, float] = (0.0, 0.0, 0.04),
    *,
    yaw: float | None = None,
    quat_wxyz: tuple[float, float, float, float] | None = None,
) -> None:
    """Reset the robot root and joints to a clean startup pose with an optional yaw."""
    if yaw is not None and quat_wxyz is not None:
        raise ValueError("Specify either yaw or quat_wxyz, not both.")

    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] = torch.tensor(position, dtype=torch.float32, device=robot.device)
    if quat_wxyz is not None:
        root_quat = torch.tensor(quat_wxyz, dtype=torch.float32, device=robot.device).view(1, 4)
        default_root_state[:, 3:7] = root_quat.repeat(robot.num_instances, 1)
    elif yaw is not None:
        yaw_t = torch.full((robot.num_instances,), float(yaw), dtype=torch.float32, device=robot.device)
        zeros = torch.zeros_like(yaw_t)
        default_root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)
    default_root_state[:, 7:] = 0.0

    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()
