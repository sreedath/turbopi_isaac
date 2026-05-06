"""Procedural mountain cliff road scene for TurboPi visual experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import isaaclab.sim as sim_utils


SKY_TEXTURE_PATH = Path(__file__).resolve().parents[1] / "assets" / "generated" / "mountain_sky_latlong.png"


@dataclass(frozen=True)
class MountainCliffSceneCfg:
    """Configuration for a narrow mountain shelf road with a cliff drop."""

    road_width: float = 0.48
    road_thickness: float = 0.055
    road_z: float = 0.82
    start_height: float = 0.055
    lower_terrain_z: float = -0.42
    shoulder_width: float = 0.10
    rail_height: float = 0.12
    scene_extent: float = 5.0
    start_offset: float = 0.32


ROAD_CENTERLINE: tuple[tuple[float, float], ...] = (
    (-1.70, -1.18),
    (-1.14, -0.92),
    (-0.65, -0.54),
    (-0.28, -0.05),
    (0.28, 0.30),
    (0.94, 0.66),
    (1.46, 1.15),
    (1.78, 1.72),
    (1.64, 2.34),
    (1.08, 2.88),
    (0.36, 3.24),
    (-0.34, 3.70),
)

RIGHT_BRANCH_CENTERLINE: tuple[tuple[float, float], ...] = (
    (1.42, 1.10),
    (2.10, 1.20),
    (2.72, 0.92),
    (3.24, 0.46),
    (3.66, -0.16),
    (3.86, -0.88),
    (3.74, -1.58),
)


def _preview(color: tuple[float, float, float], roughness: float = 0.85) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _cuboid(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
    yaw: float = 0.0,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        physics_material=(
            sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.15,
                dynamic_friction=0.95,
                restitution=0.0,
            )
            if collision
            else None
        ),
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation, orientation=_yaw_to_quat(yaw))


def _cylinder(
    prim_path: str,
    *,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _cone(
    prim_path: str,
    *,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.ConeCfg(
        radius=radius,
        height=height,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _sphere(
    prim_path: str,
    *,
    radius: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.SphereCfg(
        radius=radius,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _mesh_grid(
    prim_path: str,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    nx: int,
    ny: int,
    height_fn: Callable[[float, float], float],
    color: tuple[float, float, float],
    roughness: float = 0.92,
    collision: bool = False,
) -> None:
    """Create a static terrain mesh from a height function."""
    import isaacsim.core.utils.prims as prim_utils

    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    points: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            points.append((float(x), float(y), float(height_fn(float(x), float(y)))))

    faces: list[int] = []
    counts: list[int] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend((a, c, b, b, c, d))
            counts.extend((3, 3))

    prim_utils.create_prim(prim_path, "Xform")
    mesh_prim = prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        attributes={
            "points": points,
            "faceVertexIndices": np.asarray(faces, dtype=np.int32),
            "faceVertexCounts": np.asarray(counts, dtype=np.int32),
            "subdivisionScheme": "bilinear",
        },
    )
    material_path = f"{prim_path}/visualMaterial"
    material = _preview(color, roughness)
    material.func(material_path, material)
    sim_utils.bind_visual_material(mesh_prim.GetPrimPath(), material_path)
    if collision:
        mesh_path = mesh_prim.GetPrimPath()
        sim_utils.define_collision_properties(mesh_path, sim_utils.CollisionPropertiesCfg(collision_enabled=True))
        sim_utils.define_mesh_collision_properties(mesh_path, sim_utils.TriangleMeshPropertiesCfg())


def _segment_geometry(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    yaw = math.atan2(-dx, dy)
    return 0.5 * (sx + ex), 0.5 * (sy + ey), dx / length, dy / length, yaw


def _offset_point(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    t: float,
    side_sign: float,
    offset: float,
) -> tuple[float, float]:
    sx, sy = start
    ex, ey = end
    _, _, ux, uy, _ = _segment_geometry(start, end)
    x = sx + t * (ex - sx)
    y = sy + t * (ey - sy)
    return x - uy * side_sign * offset, y + ux * side_sign * offset


def _spawn_boulder_cluster(
    prim_prefix: str,
    *,
    x: float,
    y: float,
    z: float,
    scale: float,
    color: tuple[float, float, float],
    collision: bool = True,
) -> None:
    offsets = (
        (0.00, 0.00, 0.00, 1.00),
        (0.08, -0.04, 0.02, 0.72),
        (-0.06, 0.05, 0.01, 0.58),
        (0.02, 0.09, 0.04, 0.46),
    )
    for idx, (ox, oy, oz, radius_scale) in enumerate(offsets):
        shade = tuple(max(0.02, min(1.0, c + 0.035 * ((idx % 2) - 0.5))) for c in color)
        _sphere(
            f"{prim_prefix}Lump{idx:02d}",
            radius=scale * radius_scale,
            translation=(x + scale * ox, y + scale * oy, z + scale * (0.74 * radius_scale + oz)),
            color=shade,
            collision=collision,
            roughness=0.98,
        )
    _cuboid(
        f"{prim_prefix}Slab",
        size=(scale * 1.15, scale * 0.42, scale * 0.28),
        translation=(x - scale * 0.02, y + scale * 0.04, z + scale * 0.20),
        color=tuple(max(0.02, c - 0.045) for c in color),
        collision=collision,
        roughness=0.98,
        yaw=0.62,
    )


def _spawn_road(scene_cfg: MountainCliffSceneCfg) -> None:
    road_color = (0.18, 0.13, 0.085)
    dust_color = (0.32, 0.24, 0.16)
    edge_color = (0.39, 0.34, 0.27)
    rut_color = (0.12, 0.085, 0.055)
    gravel_color = (0.40, 0.36, 0.29)
    deck_z = scene_cfg.road_z - 0.5 * scene_cfg.road_thickness
    surface_z = scene_cfg.road_z + 0.008
    mark_z = scene_cfg.road_z + 0.015

    for idx, (start, end) in enumerate(zip(ROAD_CENTERLINE[:-1], ROAD_CENTERLINE[1:], strict=False)):
        cx, cy, ux, uy, yaw = _segment_geometry(start, end)
        length = math.dist(start, end)
        total_width = scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width
        _cuboid(
            f"/World/MountainCliffRoad/RoadSegment{idx:02d}",
            size=(total_width, length + 0.06, scene_cfg.road_thickness),
            translation=(cx, cy, deck_z),
            color=dust_color,
            collision=True,
            roughness=0.96,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/TravelSurface{idx:02d}",
            size=(scene_cfg.road_width, length + 0.05, 0.006),
            translation=(cx, cy, surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
            yaw=yaw,
        )
        for side, side_sign in (("Left", -1.0), ("Right", 1.0)):
            ox = -uy * side_sign * (0.5 * scene_cfg.road_width - 0.025)
            oy = ux * side_sign * (0.5 * scene_cfg.road_width - 0.025)
            _cuboid(
                f"/World/MountainCliffRoad/{side}GravelEdge{idx:02d}",
                size=(0.030, length - 0.030, 0.007),
                translation=(cx + ox, cy + oy, mark_z),
                color=edge_color,
                collision=False,
                roughness=0.95,
                yaw=yaw,
            )
        for rut_idx, side_sign in enumerate((-1.0, 1.0)):
            for patch_idx, t in enumerate((0.24, 0.54, 0.80)):
                patch_len = max(0.10, min(0.28, length * 0.24))
                lateral = 0.096 + 0.014 * ((idx + patch_idx) % 2)
                ox = -uy * side_sign * lateral
                oy = ux * side_sign * lateral
                px = start[0] + t * (end[0] - start[0]) + ox
                py = start[1] + t * (end[1] - start[1]) + oy
                _cuboid(
                    f"/World/MountainCliffRoad/Rut{idx:02d}_{rut_idx:02d}_{patch_idx:02d}",
                    size=(0.024, patch_len, 0.005),
                    translation=(px, py, mark_z + 0.002),
                    color=rut_color,
                    collision=False,
                    roughness=1.0,
                    yaw=yaw + 0.025 * ((idx + patch_idx) % 3 - 1),
                )
        for dash_idx in range(max(1, int(length / 0.22))):
            t = (dash_idx + 0.35) / max(1, int(length / 0.22))
            sx, sy = _offset_point(start, end, t=t, side_sign=(-1.0 if dash_idx % 2 else 1.0), offset=0.18)
            _cuboid(
                f"/World/MountainCliffRoad/ShoulderGravel{idx:02d}_{dash_idx:02d}",
                size=(0.032 + 0.008 * (dash_idx % 2), 0.052 + 0.014 * (dash_idx % 3), 0.006),
                translation=(sx, sy, mark_z + 0.001),
                color=gravel_color,
                collision=False,
                roughness=0.98,
                yaw=yaw + 0.35 * ((dash_idx % 3) - 1),
            )

    for idx, point in enumerate(ROAD_CENTERLINE[1:-1], start=1):
        _cylinder(
            f"/World/MountainCliffRoad/CurvePatch{idx:02d}",
            radius=0.5 * (scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width),
            height=scene_cfg.road_thickness,
            translation=(point[0], point[1], deck_z),
            color=dust_color,
            collision=True,
            roughness=0.96,
        )
        _cylinder(
            f"/World/MountainCliffRoad/CurveSurface{idx:02d}",
            radius=0.5 * scene_cfg.road_width,
            height=0.006,
            translation=(point[0], point[1], surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
        )


def _spawn_right_branch(scene_cfg: MountainCliffSceneCfg) -> None:
    """Spawn a short visible right-hand road option ahead of the start view."""
    road_color = (0.18, 0.13, 0.085)
    dust_color = (0.31, 0.23, 0.15)
    edge_color = (0.38, 0.33, 0.26)
    rut_color = (0.11, 0.078, 0.050)
    deck_z = scene_cfg.road_z - 0.5 * scene_cfg.road_thickness
    surface_z = scene_cfg.road_z + 0.008
    mark_z = scene_cfg.road_z + 0.015

    for idx, (start, end) in enumerate(zip(RIGHT_BRANCH_CENTERLINE[:-1], RIGHT_BRANCH_CENTERLINE[1:], strict=False)):
        cx, cy, ux, uy, yaw = _segment_geometry(start, end)
        length = math.dist(start, end)
        total_width = scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width
        _cuboid(
            f"/World/MountainCliffRoad/RightBranchRockShelf{idx:02d}",
            size=(total_width + 0.18, length + 0.10, 0.18),
            translation=(cx, cy, scene_cfg.road_z - 0.15),
            color=(0.28, 0.24, 0.19),
            collision=True,
            roughness=0.99,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/RightBranchRoadSegment{idx:02d}",
            size=(total_width, length + 0.05, scene_cfg.road_thickness),
            translation=(cx, cy, deck_z),
            color=dust_color,
            collision=True,
            roughness=0.97,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/RightBranchSurface{idx:02d}",
            size=(scene_cfg.road_width, length + 0.04, 0.006),
            translation=(cx, cy, surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
            yaw=yaw,
        )
        for side, side_sign in (("Left", -1.0), ("Right", 1.0)):
            ox = -uy * side_sign * (0.5 * scene_cfg.road_width - 0.025)
            oy = ux * side_sign * (0.5 * scene_cfg.road_width - 0.025)
            _cuboid(
                f"/World/MountainCliffRoad/RightBranch{side}Edge{idx:02d}",
                size=(0.026, max(0.10, length - 0.04), 0.007),
                translation=(cx + ox, cy + oy, mark_z),
                color=edge_color,
                collision=False,
                roughness=0.96,
                yaw=yaw,
            )
        for rut_idx, side_sign in enumerate((-1.0, 1.0)):
            for patch_idx, t in enumerate((0.32, 0.68)):
                ox = -uy * side_sign * 0.10
                oy = ux * side_sign * 0.10
                px = start[0] + t * (end[0] - start[0]) + ox
                py = start[1] + t * (end[1] - start[1]) + oy
                _cuboid(
                    f"/World/MountainCliffRoad/RightBranchRut{idx:02d}_{rut_idx:02d}_{patch_idx:02d}",
                    size=(0.022, max(0.10, min(0.24, length * 0.26)), 0.005),
                    translation=(px, py, mark_z + 0.002),
                    color=rut_color,
                    collision=False,
                    roughness=1.0,
                    yaw=yaw,
                )

    for idx, point in enumerate(RIGHT_BRANCH_CENTERLINE[:-1]):
        _cylinder(
            f"/World/MountainCliffRoad/RightBranchCurvePatch{idx:02d}",
            radius=0.5 * (scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width),
            height=scene_cfg.road_thickness,
            translation=(point[0], point[1], deck_z),
            color=dust_color,
            collision=True,
            roughness=0.97,
        )
        _cylinder(
            f"/World/MountainCliffRoad/RightBranchCurveSurface{idx:02d}",
            radius=0.5 * scene_cfg.road_width,
            height=0.006,
            translation=(point[0], point[1], surface_z),
            color=road_color,
            collision=False,
            roughness=0.98,
        )


def _spawn_guard_rails(scene_cfg: MountainCliffSceneCfg) -> None:
    post_color = (0.36, 0.28, 0.20)
    rail_color = (0.47, 0.45, 0.40)
    for idx, (start, end) in enumerate(zip(ROAD_CENTERLINE[:-1], ROAD_CENTERLINE[1:], strict=False)):
        cx, cy, ux, uy, yaw = _segment_geometry(start, end)
        length = math.dist(start, end)
        # Put rails on the valley side of the shelf road, with gaps at turns.
        side_sign = 1.0
        offset = 0.5 * scene_cfg.road_width + scene_cfg.shoulder_width + 0.035
        ox = -uy * side_sign * offset
        oy = ux * side_sign * offset
        _cuboid(
            f"/World/MountainCliffRoad/GuardRail{idx:02d}",
            size=(0.030, max(0.10, length - 0.16), 0.024),
            translation=(cx + ox, cy + oy, scene_cfg.road_z + scene_cfg.rail_height),
            color=rail_color,
            collision=True,
            roughness=0.75,
            yaw=yaw,
        )
        _cuboid(
            f"/World/MountainCliffRoad/LowerGuardRail{idx:02d}",
            size=(0.024, max(0.08, length - 0.22), 0.018),
            translation=(cx + ox, cy + oy, scene_cfg.road_z + 0.060),
            color=(0.32, 0.31, 0.28),
            collision=True,
            roughness=0.85,
            yaw=yaw,
        )
        post_count = max(2, int(length / 0.24))
        for post_idx in range(post_count):
            t = (post_idx + 0.5) / post_count
            px = start[0] + t * (end[0] - start[0]) + ox
            py = start[1] + t * (end[1] - start[1]) + oy
            _cuboid(
                f"/World/MountainCliffRoad/GuardPost{idx:02d}_{post_idx:02d}",
                size=(0.030, 0.030, scene_cfg.rail_height),
                translation=(px, py, scene_cfg.road_z + 0.5 * scene_cfg.rail_height),
                color=tuple(max(0.02, c + 0.025 * ((post_idx % 3) - 1)) for c in post_color),
                collision=True,
                roughness=0.85,
                yaw=yaw,
            )
            if post_idx % 4 == 0:
                _cuboid(
                    f"/World/MountainCliffRoad/GuardReflector{idx:02d}_{post_idx:02d}",
                    size=(0.034, 0.006, 0.018),
                    translation=(px, py, scene_cfg.road_z + scene_cfg.rail_height + 0.020),
                    color=(0.80, 0.58, 0.12),
                    collision=False,
                    roughness=0.45,
                    yaw=yaw,
                )


def _spawn_terrain(scene_cfg: MountainCliffSceneCfg) -> None:
    def valley_height(x: float, y: float) -> float:
        ripple = 0.055 * math.sin(2.7 * x + 0.4) + 0.045 * math.cos(2.2 * y - 0.6)
        slope = -0.055 * x + 0.025 * y
        return scene_cfg.lower_terrain_z + slope + ripple

    def far_mountain_height(x: float, y: float) -> float:
        distance = max(0.0, y - 3.0)
        ridge = 0.18 + 0.10 * distance
        ridge += 0.28 * math.sin(0.75 * y + 0.55 * x)
        ridge += 0.16 * math.cos(1.35 * x - 0.25 * y)
        return scene_cfg.lower_terrain_z + ridge + 0.08 * abs(x)

    _mesh_grid(
        "/World/MountainCliffRoad/ValleyGround",
        x_range=(-4.2, 4.2),
        y_range=(-3.2, 6.4),
        nx=48,
        ny=64,
        height_fn=valley_height,
        color=(0.22, 0.20, 0.16),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/FarMountainRidges",
        x_range=(-5.8, 5.8),
        y_range=(3.2, 7.2),
        nx=48,
        ny=28,
        height_fn=far_mountain_height,
        color=(0.24, 0.22, 0.19),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/LeftRockSlope",
        x_range=(-3.00, -1.58),
        y_range=(-2.8, 5.4),
        nx=12,
        ny=54,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z + 0.30 + 0.78 * (-1.58 - x) + 0.06 * math.sin(4.0 * y),
        color=(0.27, 0.23, 0.19),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/RightDropSlope",
        x_range=(1.35, 2.90),
        y_range=(-2.8, 5.8),
        nx=12,
        ny=54,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z + 0.18 + 0.22 * (x - 1.35) + 0.05 * math.cos(3.5 * y),
        color=(0.20, 0.19, 0.17),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/ForwardRollingHills",
        x_range=(-2.6, 3.4),
        y_range=(1.6, 5.6),
        nx=36,
        ny=36,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z
        + 0.12
        + 0.12 * (y - 1.6)
        + 0.09 * math.sin(2.1 * x + 0.7 * y)
        + 0.05 * math.cos(3.0 * y),
        color=(0.23, 0.22, 0.18),
        collision=True,
    )
    _mesh_grid(
        "/World/MountainCliffRoad/ForegroundTalusField",
        x_range=(-2.80, 2.65),
        y_range=(-2.95, -1.55),
        nx=34,
        ny=12,
        height_fn=lambda x, y: scene_cfg.lower_terrain_z + 0.10 + 0.04 * math.sin(3.4 * x) + 0.05 * math.cos(2.6 * y),
        color=(0.19, 0.18, 0.15),
        collision=True,
    )
    _cuboid(
        "/World/MountainCliffRoad/LowerSolidGround",
        size=(9.0, 10.5, 0.060),
        translation=(0.0, 1.6, scene_cfg.lower_terrain_z - 0.10),
        color=(0.17, 0.16, 0.13),
        collision=True,
        roughness=1.0,
    )

    _cuboid(
        "/World/MountainCliffRoad/River",
        size=(0.20, 8.8, 0.006),
        translation=(2.10, 1.55, scene_cfg.lower_terrain_z - 0.025),
        color=(0.03, 0.22, 0.34),
        collision=False,
        roughness=0.25,
        yaw=-0.28,
    )
    for idx, x_offset in enumerate((-0.05, 0.04, 0.095)):
        _cuboid(
            f"/World/MountainCliffRoad/RiverHighlight{idx:02d}",
            size=(0.024, 7.50 - 0.40 * idx, 0.004),
            translation=(2.10 + x_offset, 1.45 + 0.10 * idx, scene_cfg.lower_terrain_z - 0.019),
            color=(0.12, 0.42, 0.58),
            collision=False,
            roughness=0.22,
            yaw=-0.28,
        )
    for idx, (x, y, sx, sy) in enumerate(
        (
            (1.66, -1.20, 0.26, 0.70),
            (2.22, 0.42, 0.24, 0.86),
            (1.72, 1.20, 0.22, 0.52),
            (2.30, 2.00, 0.20, 0.66),
            (1.88, 3.20, 0.24, 0.82),
            (2.34, 4.40, 0.22, 0.70),
        )
    ):
        _cuboid(
            f"/World/MountainCliffRoad/GravelBar{idx:02d}",
            size=(sx, sy, 0.006),
            translation=(x, y, scene_cfg.lower_terrain_z - 0.014),
            color=(0.42, 0.36, 0.25),
            collision=False,
            roughness=0.95,
            yaw=-0.24,
        )
    _cuboid(
        "/World/MountainCliffRoad/StartShelf",
        size=(1.10, 0.82, 0.10),
        translation=(-1.62, -1.22, scene_cfg.road_z - 0.085),
        color=(0.29, 0.24, 0.19),
        collision=True,
        roughness=0.98,
        yaw=0.42,
    )


def _spawn_rocks_and_plants(scene_cfg: MountainCliffSceneCfg) -> None:
    rock_specs = (
        (-1.52, -1.36, 0.12, (0.30, 0.26, 0.22)),
        (-1.46, -0.58, 0.16, (0.25, 0.23, 0.21)),
        (-1.07, 0.07, 0.11, (0.36, 0.30, 0.24)),
        (-0.72, 0.62, 0.15, (0.29, 0.25, 0.22)),
        (1.70, 0.94, 0.14, (0.32, 0.28, 0.24)),
        (1.80, -0.35, 0.10, (0.27, 0.25, 0.22)),
        (2.10, -1.20, 0.18, (0.33, 0.28, 0.23)),
        (1.34, 1.78, 0.15, (0.31, 0.27, 0.23)),
        (0.88, 2.36, 0.18, (0.28, 0.25, 0.22)),
        (0.14, 3.04, 0.13, (0.35, 0.30, 0.24)),
        (-0.56, 3.72, 0.20, (0.30, 0.27, 0.23)),
        (2.42, 2.78, 0.22, (0.27, 0.25, 0.22)),
    )
    for idx, (x, y, radius, color) in enumerate(rock_specs):
        _spawn_boulder_cluster(
            f"/World/MountainCliffRoad/Boulder{idx:02d}",
            x=x,
            y=y,
            z=scene_cfg.lower_terrain_z,
            scale=radius,
            color=color,
        )

    talus_specs = [
        (-1.34, -1.00, 0.052),
        (-1.22, -0.78, 0.044),
        (-1.02, -0.46, 0.060),
        (-0.78, -0.18, 0.040),
        (-0.44, 0.12, 0.050),
        (0.02, 0.42, 0.042),
        (0.50, 0.72, 0.056),
        (1.26, -1.12, 0.062),
        (1.44, -0.72, 0.046),
        (1.62, -0.18, 0.052),
        (1.78, 0.38, 0.042),
        (1.86, 0.94, 0.058),
    ]
    for idx, (x, y, radius) in enumerate(talus_specs):
        _sphere(
            f"/World/MountainCliffRoad/TalusStone{idx:02d}",
            radius=radius,
            translation=(x, y, scene_cfg.lower_terrain_z + 0.06 + 0.01 * (idx % 3)),
            color=(0.24 + 0.02 * (idx % 2), 0.22, 0.19),
            collision=False,
            roughness=0.98,
        )

    left_slope_specs = (
        (-1.88, -1.06, 0.11, (0.31, 0.27, 0.23)),
        (-2.12, -0.62, 0.14, (0.27, 0.25, 0.22)),
        (-1.96, -0.18, 0.10, (0.34, 0.29, 0.24)),
        (-2.28, 0.36, 0.12, (0.29, 0.26, 0.23)),
        (-2.38, 1.08, 0.16, (0.30, 0.27, 0.23)),
        (-1.92, 1.32, 0.13, (0.35, 0.30, 0.25)),
    )
    for idx, (x, y, radius, color) in enumerate(left_slope_specs):
        z = scene_cfg.lower_terrain_z + 0.30 + 0.78 * (-1.58 - x) + 0.06 * math.sin(4.0 * y)
        _spawn_boulder_cluster(
            f"/World/MountainCliffRoad/LeftChaseBoulder{idx:02d}",
            x=x,
            y=y,
            z=z,
            scale=radius,
            color=color,
            collision=True,
        )

    strata_specs = (
        (-2.02, -0.88, 0.42, 0.12, 0.035, 0.35),
        (-2.22, -0.34, 0.36, 0.10, 0.030, -0.18),
        (-2.34, 0.62, 0.46, 0.13, 0.035, 0.22),
        (-1.84, 0.18, 0.30, 0.09, 0.028, 0.52),
        (-2.08, 1.02, 0.38, 0.11, 0.032, -0.35),
    )
    for idx, (x, y, sx, sy, sz, yaw) in enumerate(strata_specs):
        z = scene_cfg.lower_terrain_z + 0.30 + 0.78 * (-1.58 - x) + 0.06 * math.sin(4.0 * y)
        _cuboid(
            f"/World/MountainCliffRoad/LeftChaseStrataSlab{idx:02d}",
            size=(sx, sy, sz),
            translation=(x, y, z + 0.5 * sz),
            color=(0.26, 0.23, 0.19),
            collision=True,
            roughness=0.99,
            yaw=yaw,
        )


def _spawn_sky_and_lights(scene_cfg: MountainCliffSceneCfg) -> None:
    sky_kwargs: dict[str, object] = {}
    if SKY_TEXTURE_PATH.exists():
        sky_kwargs = {
            "texture_file": str(SKY_TEXTURE_PATH),
            "texture_format": "latlong",
            "visible_in_primary_ray": True,
        }
    sky_cfg = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0), **sky_kwargs)
    sky_cfg.func("/World/MountainCliffRoad/SkyLight", sky_cfg)
    sun_cfg = sim_utils.DistantLightCfg(intensity=1700.0, color=(1.0, 0.86, 0.66), angle=0.24)
    sun_cfg.func("/World/MountainCliffRoad/SunLight", sun_cfg)

    # No sky wall here: chase-view teleop should see open background, not a flat panel.



def design_mountain_cliff_scene(scene_cfg: MountainCliffSceneCfg) -> None:
    """Spawn a realistic-looking mountain shelf road with a canyon drop."""
    _spawn_sky_and_lights(scene_cfg)
    _spawn_terrain(scene_cfg)
    _spawn_road(scene_cfg)
    _spawn_right_branch(scene_cfg)
    _spawn_guard_rails(scene_cfg)
    _spawn_rocks_and_plants(scene_cfg)


def start_pose(scene_cfg: MountainCliffSceneCfg | None = None) -> tuple[tuple[float, float, float], float]:
    """Start at the lower road entrance, facing along the first mountain-road segment."""
    cfg = scene_cfg or MountainCliffSceneCfg()
    start = ROAD_CENTERLINE[0]
    next_point = ROAD_CENTERLINE[1]
    yaw = math.atan2(next_point[1] - start[1], next_point[0] - start[0])
    dx = next_point[0] - start[0]
    dy = next_point[1] - start[1]
    segment_length = max(1e-6, math.hypot(dx, dy))
    t = min(0.55, cfg.start_offset / segment_length)
    x = start[0] + t * dx
    y = start[1] + t * dy
    return (x, y, cfg.road_z + cfg.start_height), yaw
