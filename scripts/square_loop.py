from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from common import ROBOT_CAMERA_PATH

DirectionName = Literal["clockwise", "counterclockwise"]

DIRECTION_TO_SIGN: dict[DirectionName, float] = {
    "clockwise": 1.0,
    "counterclockwise": -1.0,
}

TASK_INDEX: dict[DirectionName, int] = {
    "clockwise": 0,
    "counterclockwise": 1,
}

TASKS: tuple[DirectionName, ...] = ("clockwise", "counterclockwise")


@dataclass(frozen=True)
class SquareTrackSceneCfg:
    """Geometry and visuals for the square-loop collection arena."""

    square_half_extent: float = 0.45
    floor_half_extent: float = 1.40
    tape_width: float = 0.08
    wall_height: float = 0.55
    wall_thickness: float = 0.04
    floor_z: float = 0.001
    tape_z: float = 0.003
    floor_color: tuple[float, float, float] = (0.14, 0.14, 0.16)
    tape_color: tuple[float, float, float] = (0.95, 0.95, 0.94)
    wall_color: tuple[float, float, float] = (0.10, 0.10, 0.12)
    start_height: float = 0.04


@dataclass(frozen=True)
class ControlLimits:
    """Maximum body-twist command used for normalization and clamping."""

    max_vx: float = 0.45
    max_vy: float = 0.35
    max_wz: float = 2.00

    def as_tensor(self, device: str | torch.device) -> torch.Tensor:
        return torch.tensor([self.max_vx, self.max_vy, self.max_wz], dtype=torch.float32, device=device)


@dataclass(frozen=True)
class TeacherControllerCfg:
    """Parameters for the deterministic square-loop teacher.

    Heading gains were halved after the mecanum roller + wheel damping fix
    made the plant's yaw gain drop from ~2x to ~1x. With the old gains and
    a clean plant the outer heading loop's poles sat at |z|~0.735 with a
    large imaginary part, so the teacher's own wz commands rang every 2-3
    control ticks on the straights. Halving the gains puts the poles on
    the real axis and kills the ring without changing corner behavior.
    """

    target_speed: float = 0.18
    lookahead_distance: float = 0.05
    boundary_gain: float = 0.8
    heading_gain: float = 2.7
    cross_track_heading_gain: float = 1.2
    lateral_gain: float = 0.45
    corner_slowdown: float = 0.75
    min_corner_scale: float = 0.40
    corner_blend_distance: float = 0.10
    corner_slowdown_distance: float = 0.14
    max_lateral_speed: float = 0.045
    track_error_slowdown: float = 0.40
    min_tracking_scale: float = 0.55
    heading_slowdown_angle: float = 0.75
    strafe_suppression_angle: float = 0.35
    min_forward_speed: float = 0.05
    command_filter_alpha_xy: float = 0.24
    command_filter_alpha_wz: float = 0.30


@dataclass(frozen=True)
class TrackingObservation:
    """Compact robot/track state used by the recorder and controller."""

    position_w: torch.Tensor
    track_error: float
    track_phase: float
    body_velocity: torch.Tensor
    body_ang_velocity: torch.Tensor
    height: float
    has_nan: bool


def direction_sign(direction: DirectionName) -> float:
    return DIRECTION_TO_SIGN[direction]


def start_pose_for_direction(
    scene_cfg: SquareTrackSceneCfg, direction: DirectionName
) -> tuple[tuple[float, float, float], float]:
    """Return a consistent spawn pose on the left edge of the square."""
    yaw = 0.5 * math.pi if direction == "clockwise" else -0.5 * math.pi
    return (-scene_cfg.square_half_extent, 0.0, scene_cfg.start_height), yaw


def square_corners_xy(scene_cfg: SquareTrackSceneCfg) -> tuple[tuple[float, float], ...]:
    """Return the taped square centerline corners in clockwise order."""
    h = scene_cfg.square_half_extent
    return ((-h, -h), (-h, h), (h, h), (h, -h))


def outer_wall_half_extent(scene_cfg: SquareTrackSceneCfg) -> float:
    """Return the x/y coordinate of the arena wall center planes."""
    return scene_cfg.floor_half_extent + 0.5 * scene_cfg.wall_thickness


def design_square_loop_scene(scene_cfg: SquareTrackSceneCfg) -> None:
    """Spawn a dark floor, a visible taped square loop, and physical boundary walls."""
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

    light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.95, 0.95, 0.95))
    light_cfg.func("/World/Light", light_cfg)
    distant_light_cfg = sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0), angle=0.5)
    distant_light_cfg.func("/World/SunLight", distant_light_cfg)

    floor_size = 2.0 * scene_cfg.floor_half_extent
    tape_span = 2.0 * scene_cfg.square_half_extent
    wall_half = scene_cfg.floor_half_extent + 0.5 * scene_cfg.wall_thickness
    wall_z = 0.5 * scene_cfg.wall_height

    floor_cfg = sim_utils.CuboidCfg(
        size=(floor_size, floor_size, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.floor_color, roughness=0.95),
    )
    floor_cfg.func("/World/SquareLoop/Floor", floor_cfg, translation=(0.0, 0.0, scene_cfg.floor_z))

    tape_cfg = sim_utils.CuboidCfg(
        size=(tape_span + scene_cfg.tape_width, scene_cfg.tape_width, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.tape_color, roughness=0.65),
    )
    tape_cfg.func(
        "/World/SquareLoop/TapeTop",
        tape_cfg,
        translation=(0.0, scene_cfg.square_half_extent, scene_cfg.tape_z),
    )
    tape_cfg.func(
        "/World/SquareLoop/TapeBottom",
        tape_cfg,
        translation=(0.0, -scene_cfg.square_half_extent, scene_cfg.tape_z),
    )

    tape_side_cfg = sim_utils.CuboidCfg(
        size=(scene_cfg.tape_width, tape_span + scene_cfg.tape_width, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.tape_color, roughness=0.65),
    )
    tape_side_cfg.func(
        "/World/SquareLoop/TapeLeft",
        tape_side_cfg,
        translation=(-scene_cfg.square_half_extent, 0.0, scene_cfg.tape_z),
    )
    tape_side_cfg.func(
        "/World/SquareLoop/TapeRight",
        tape_side_cfg,
        translation=(scene_cfg.square_half_extent, 0.0, scene_cfg.tape_z),
    )

    wall_x_cfg = sim_utils.CuboidCfg(
        size=(floor_size + scene_cfg.wall_thickness, scene_cfg.wall_thickness, scene_cfg.wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.wall_color, roughness=0.90),
    )
    wall_x_cfg.func("/World/SquareLoop/WallTop", wall_x_cfg, translation=(0.0, wall_half, wall_z))
    wall_x_cfg.func("/World/SquareLoop/WallBottom", wall_x_cfg, translation=(0.0, -wall_half, wall_z))

    wall_y_cfg = sim_utils.CuboidCfg(
        size=(scene_cfg.wall_thickness, floor_size + scene_cfg.wall_thickness, scene_cfg.wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.wall_color, roughness=0.90),
    )
    wall_y_cfg.func("/World/SquareLoop/WallLeft", wall_y_cfg, translation=(-wall_half, 0.0, wall_z))
    wall_y_cfg.func("/World/SquareLoop/WallRight", wall_y_cfg, translation=(wall_half, 0.0, wall_z))


def build_robot_camera_sensor(*, width: int, height: int) -> Camera:
    """Attach a camera sensor to the existing robot camera prim."""
    camera_cfg = CameraCfg(
        prim_path=ROBOT_CAMERA_PATH,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=None,
    )
    return Camera(camera_cfg)


def build_overhead_camera_sensor(
    *,
    width: int,
    height: int,
    prim_path: str = "/World/SpectatorOverhead",
) -> Camera:
    """Spawn a fixed top-down spectator camera prim. Caller must position it
    via :meth:`Camera.set_world_poses_from_view` after :meth:`Camera.initialize`.
    """
    camera_cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
    )
    return Camera(camera_cfg)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def square_phase_to_point_and_tangent(phase: torch.Tensor, half_extent: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return point/tangent on the clockwise square boundary for phases in [0, 1)."""
    phase = torch.remainder(phase, 1.0)
    perimeter = 8.0 * half_extent
    segment_length = 2.0 * half_extent
    distance = phase * perimeter
    segment = torch.floor(distance / segment_length).to(torch.long)
    offset = torch.remainder(distance, segment_length)

    point = torch.zeros((len(phase), 2), dtype=phase.dtype, device=phase.device)
    tangent = torch.zeros_like(point)
    h = half_extent

    mask = segment == 0
    point[mask, 0] = -h
    point[mask, 1] = -h + offset[mask]
    tangent[mask] = phase.new_tensor((0.0, 1.0))

    mask = segment == 1
    point[mask, 0] = -h + offset[mask]
    point[mask, 1] = h
    tangent[mask] = phase.new_tensor((1.0, 0.0))

    mask = segment == 2
    point[mask, 0] = h
    point[mask, 1] = h - offset[mask]
    tangent[mask] = phase.new_tensor((0.0, -1.0))

    mask = segment == 3
    point[mask, 0] = h - offset[mask]
    point[mask, 1] = -h
    tangent[mask] = phase.new_tensor((-1.0, 0.0))

    return point, tangent


def compute_square_track_frame(
    pos_xy: torch.Tensor, half_extent: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points onto the nearest point on the taped square boundary."""
    x = pos_xy[:, 0]
    y = pos_xy[:, 1]
    h = half_extent

    clamp_x = torch.clamp(x, -h, h)
    clamp_y = torch.clamp(y, -h, h)

    left = torch.stack((-torch.ones_like(x) * h, clamp_y), dim=-1)
    top = torch.stack((clamp_x, torch.ones_like(y) * h), dim=-1)
    right = torch.stack((torch.ones_like(x) * h, clamp_y), dim=-1)
    bottom = torch.stack((clamp_x, -torch.ones_like(y) * h), dim=-1)

    candidates = torch.stack((left, top, right, bottom), dim=1)
    diff = candidates - pos_xy.unsqueeze(1)
    dist_sq = torch.sum(diff * diff, dim=-1)
    segment = torch.argmin(dist_sq, dim=1)

    tangents = pos_xy.new_tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ]
    )

    phase_candidates = torch.stack(
        (
            0.25 * (clamp_y + h) / (2.0 * h),
            0.25 + 0.25 * (clamp_x + h) / (2.0 * h),
            0.50 + 0.25 * (h - clamp_y) / (2.0 * h),
            0.75 + 0.25 * (h - clamp_x) / (2.0 * h),
        ),
        dim=-1,
    )

    gather_idx = segment.view(-1, 1, 1).expand(-1, 1, 2)
    nearest = torch.gather(candidates, 1, gather_idx).squeeze(1)
    tangent = tangents[segment]
    distance = torch.sqrt(torch.gather(dist_sq, 1, segment.view(-1, 1)).squeeze(1))
    phase = torch.gather(phase_candidates, 1, segment.view(-1, 1)).squeeze(1)
    return nearest, tangent, distance, phase


def phase_to_segment_and_progress(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized square phase into a segment id and progress along that segment."""
    phase_wrapped = torch.remainder(phase, 1.0)
    segment_float = phase_wrapped * 4.0
    segment = torch.floor(segment_float).to(torch.long)
    segment = torch.clamp(segment, min=0, max=3)
    segment_progress = torch.remainder(segment_float, 1.0)
    return segment, segment_progress


def segment_tangent_clockwise(segment: torch.Tensor) -> torch.Tensor:
    """Return clockwise tangent vectors for square segment indices."""
    tangents = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
        device=segment.device,
    )
    return tangents[segment]


def segment_inward_normal(segment: torch.Tensor) -> torch.Tensor:
    """Return inward-facing normals for square segment indices."""
    normals = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
        device=segment.device,
    )
    return normals[segment]


def segment_signed_lateral_offset(pos_xy: torch.Tensor, segment: torch.Tensor, half_extent: float) -> torch.Tensor:
    """Return signed offset from the current square edge along its inward normal."""
    x = pos_xy[:, 0]
    y = pos_xy[:, 1]
    h = half_extent
    offsets = torch.stack((x + h, h - y, h - x, y + h), dim=-1)
    return torch.gather(offsets, 1, segment.view(-1, 1)).squeeze(1)


def observe_track_state(robot, scene_cfg: SquareTrackSceneCfg) -> TrackingObservation:
    """Read the current robot state in the square-loop frame."""
    root_pos_w = robot.data.root_pos_w[0].detach().clone()
    pos_xy = root_pos_w[:2].unsqueeze(0)
    _, _, track_error, track_phase = compute_square_track_frame(pos_xy, scene_cfg.square_half_extent)
    body_velocity = robot.data.root_lin_vel_b[0, :2].detach().clone()
    body_ang_velocity = robot.data.root_ang_vel_b[0, 2:3].detach().clone()
    height = float(root_pos_w[2].item())
    has_nan = bool(torch.isnan(robot.data.root_pos_w).any().item())
    return TrackingObservation(
        position_w=root_pos_w,
        track_error=float(track_error[0].item()),
        track_phase=float(track_phase[0].item()),
        body_velocity=body_velocity,
        body_ang_velocity=body_ang_velocity,
        height=height,
        has_nan=has_nan,
    )


class SquareTrackTeacher:
    """Deterministic teacher that follows the taped square boundary."""

    def __init__(
        self,
        *,
        scene_cfg: SquareTrackSceneCfg,
        limits: ControlLimits,
        controller_cfg: TeacherControllerCfg,
        device: str | torch.device,
    ):
        self.scene_cfg = scene_cfg
        self.limits = limits
        self.cfg = controller_cfg
        self.device = device
        self.max_command = limits.as_tensor(device)
        self.perimeter = 8.0 * scene_cfg.square_half_extent
        self.previous_action = torch.zeros(3, dtype=torch.float32, device=device)
        self.direction: DirectionName = "clockwise"

    def reset(self, direction: DirectionName) -> None:
        self.direction = direction
        self.previous_action.zero_()

    def compute_action(self, robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return previous state action, current normalized action, and body-twist command."""
        sign = direction_sign(self.direction)
        pos_w = robot.data.root_pos_w[:, :2]
        quat_w = robot.data.root_quat_w
        nearest_xy, _, track_error, phase = compute_square_track_frame(pos_w, self.scene_cfg.square_half_extent)
        segment, segment_progress = phase_to_segment_and_progress(phase)
        current_tangent_xy = segment_tangent_clockwise(segment) * sign
        direction_step = 1 if sign > 0.0 else -1
        next_segment = torch.remainder(segment + direction_step, 4)
        next_tangent_xy = segment_tangent_clockwise(next_segment) * sign
        segment_length = 2.0 * self.scene_cfg.square_half_extent
        distance_to_corner = (
            (1.0 - segment_progress) * segment_length if sign > 0.0 else segment_progress * segment_length
        )

        blend_distance = max(self.cfg.lookahead_distance, self.cfg.corner_blend_distance)
        corner_blend = torch.clamp(1.0 - distance_to_corner / max(blend_distance, 1e-4), min=0.0, max=1.0)
        blended_tangent_xy = (1.0 - corner_blend).unsqueeze(-1) * current_tangent_xy + corner_blend.unsqueeze(-1) * next_tangent_xy
        blended_norm = torch.linalg.norm(blended_tangent_xy, dim=-1, keepdim=True)
        target_tangent_xy = torch.where(
            blended_norm > 1e-6,
            blended_tangent_xy / blended_norm,
            current_tangent_xy,
        )
        lookahead_phase = phase + sign * (self.cfg.lookahead_distance / self.perimeter)
        target_point_xy, _ = square_phase_to_point_and_tangent(lookahead_phase, self.scene_cfg.square_half_extent)

        speed = torch.full((1,), self.cfg.target_speed, dtype=torch.float32, device=self.device)
        speed = torch.clamp(speed, min=0.05, max=self.limits.max_vx)
        _, _, yaw = euler_xyz_from_quat(quat_w)
        path_yaw = torch.atan2(target_tangent_xy[:, 1], target_tangent_xy[:, 0])
        yaw_error = wrap_to_pi(path_yaw - yaw)
        turn_error = torch.abs(yaw_error)
        corner_speed_scale = torch.clamp(
            distance_to_corner / max(self.cfg.corner_slowdown_distance, 1e-4),
            min=self.cfg.min_corner_scale,
            max=1.0,
        )
        track_speed_scale = torch.clamp(
            1.0 - self.cfg.track_error_slowdown * torch.clamp(track_error, max=0.15) / 0.15,
            min=self.cfg.min_tracking_scale,
            max=1.0,
        )
        heading_speed_scale = torch.clamp(
            1.0 - turn_error / max(self.cfg.heading_slowdown_angle, 1e-4),
            min=0.05,
            max=1.0,
        )
        speed = speed * torch.minimum(corner_speed_scale, torch.minimum(track_speed_scale, heading_speed_scale))
        speed = torch.clamp(speed, min=self.cfg.min_forward_speed, max=self.limits.max_vx)

        target_point_w = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        target_point_w[:, :2] = target_point_xy
        pos_w_3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        pos_w_3[:, :2] = pos_w
        lookahead_vec_b = quat_apply_inverse(quat_w, target_point_w - pos_w_3)[:, :2]
        nearest_point_w = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        nearest_point_w[:, :2] = nearest_xy
        nearest_vec_b = quat_apply_inverse(quat_w, nearest_point_w - pos_w_3)[:, :2]

        point_heading_error = torch.atan2(lookahead_vec_b[:, 1], torch.clamp(lookahead_vec_b[:, 0], min=0.04))
        strafe_scale = torch.clamp(
            1.0 - turn_error / max(self.cfg.strafe_suppression_angle, 1e-4),
            min=0.0,
            max=1.0,
        )
        lateral_command = torch.clamp(
            self.cfg.boundary_gain * self.cfg.lateral_gain * nearest_vec_b[:, 1] * strafe_scale,
            min=-self.cfg.max_lateral_speed,
            max=self.cfg.max_lateral_speed,
        )
        forward_command = torch.clamp(
            speed * torch.clamp(torch.cos(yaw_error), min=0.15, max=1.0),
            min=self.cfg.min_forward_speed,
            max=self.limits.max_vx,
        )
        desired_wz = self.cfg.heading_gain * yaw_error + self.cfg.cross_track_heading_gain * point_heading_error

        command = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        command[:, 0] = forward_command
        command[:, 1] = lateral_command
        command[:, 2] = desired_wz
        command = torch.clamp(command, min=-self.max_command, max=self.max_command)

        raw_action = torch.clamp(command / self.max_command, -1.0, 1.0)
        action = raw_action.clone()
        keep_xy = 1.0 - self.cfg.command_filter_alpha_xy
        keep_wz = 1.0 - self.cfg.command_filter_alpha_wz
        action[:, :2] = keep_xy * self.previous_action[:2].unsqueeze(0) + self.cfg.command_filter_alpha_xy * raw_action[:, :2]
        action[:, 2] = keep_wz * self.previous_action[2] + self.cfg.command_filter_alpha_wz * raw_action[:, 2]
        action = torch.clamp(action, -1.0, 1.0)

        filtered_command = action * self.max_command
        state = self.previous_action.clone()
        self.previous_action.copy_(action[0])
        info = {
            "track_error": float(track_error[0].item()),
            "track_phase": float(phase[0].item()),
            "segment_index": int(segment[0].item()),
            "distance_to_corner": float(distance_to_corner[0].item()),
            "point_heading_error": float(point_heading_error[0].item()),
            "target_yaw_error": float(yaw_error[0].item()),
        }
        return state, action[0].clone(), filtered_command[0].clone(), info
