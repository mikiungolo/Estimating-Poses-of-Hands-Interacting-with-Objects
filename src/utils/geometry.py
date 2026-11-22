"""
Geometry utilities: projection, quaternion, MANO, transformations.
Use pytorch3d.transforms where possible.
"""
from __future__ import annotations

import torch


def project_3d_to_2d(
    points_3d: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points (camera coords) to 2D pixels."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2].clamp(min=1e-6)

    u = fx * x / z + cx
    v = fy * y / z + cy

    return torch.stack([u, v], dim=-1)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (x, y, z, w) -> Rotation matrix (3, 3). Use pytorch3d.transforms.quaternion_to_matrix."""
    raise NotImplementedError("Use pytorch3d.transforms.quaternion_to_matrix")


def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (3, 3) -> Quaternion (x, y, z, w). Use pytorch3d.transforms.matrix_to_quaternion."""
    raise NotImplementedError("Use pytorch3d.transforms.matrix_to_quaternion")


def transform_points(points: torch.Tensor, rotmat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Apply rotation + translation to 3D points."""
    return points @ rotmat.transpose(-2, -1) + trans


def mano_forward(mano_params: torch.Tensor, mano_layer) -> dict:
    """Forward MANO layer (smplx) to get vertices and joints."""
    raise NotImplementedError("Integrate smplx MANO layer")


def estimate_contact_points(hand_verts: torch.Tensor, obj_verts: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
    """Estimate contact map: per-vertex hand-object distance < threshold."""
    raise NotImplementedError("Implement contact estimation")
