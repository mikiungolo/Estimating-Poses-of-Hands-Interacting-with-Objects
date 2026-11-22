"""
Evaluation Metrics for HOI.

Project spec (Table 1 & 2):
- Joint metrics: MPJPE, RRJ, PAJ (PA-MPJPE), AUCJ (0-50mm)
- Vertex metrics: MPVPE, RRV, PAV, AUCV (0-50mm)
- Object metrics: ADD, PD (Projection Distance)

Best practice: Use established libraries where possible:
- pytorch3d: chamfer_distance, mesh ops, point cloud metrics
- smplx: MANO model for vertex/joint generation
- scipy: Procrustes analysis for PA-MPJPE
- Custom: only where no library exists
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import integrate


def compute_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """Mean Per-Joint Position Error (mm)."""
    if pred_joints.ndim == 2:
        pred_joints = pred_joints.unsqueeze(0)
        gt_joints = gt_joints.unsqueeze(0)

    diff = pred_joints - gt_joints
    dist = torch.norm(diff, dim=-1)
    return dist.mean().item()


def compute_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """Procrustes-Aligned MPJPE (PA-MPJPE). Use scipy.spatial.procrustes or custom."""
    raise NotImplementedError("Use scipy.spatial.procrustes or implement")


def compute_mpvpe(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> float:
    """Mean Per-Vertex Position Error (mm) for 778 MANO vertices."""
    if pred_verts.ndim == 2:
        pred_verts = pred_verts.unsqueeze(0)
        gt_verts = gt_verts.unsqueeze(0)

    diff = pred_verts - gt_verts
    dist = torch.norm(diff, dim=-1)
    return dist.mean().item()


def compute_pa_mpvpe(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> float:
    """PA-MPVPE: Procrustes-aligned MPVPE."""
    raise NotImplementedError("Implement using Procrustes")


def compute_add(pred_rot: torch.Tensor, pred_trans: torch.Tensor,
                gt_rot: torch.Tensor, gt_trans: torch.Tensor,
                model_points: torch.Tensor) -> float:
    """ADD (Average Distance of Model Points) for object pose. Use pytorch3d if available."""
    raise NotImplementedError("Use pytorch3d or implement")


def compute_auc(errors: np.ndarray, max_threshold: float = 50.0) -> float:
    """Area Under Curve (0-max_threshold mm)."""
    thresholds = np.linspace(0, max_threshold, 100)
    accuracies = [(errors <= t).mean() for t in thresholds]
    return integrate.trapezoid(accuracies, thresholds) / max_threshold


def compute_rrv(pred_verts: torch.Tensor, gt_verts: torch.Tensor) -> float:
    """Relative Root Vertex Error (RRV) - aligns root (wrist)."""
    raise NotImplementedError("Implement")


def compute_rrj(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> float:
    """Relative Root Joint Error (RRJ) - aligns root (wrist joint 0)."""
    raise NotImplementedError("Implement")


def compute_all_metrics(
    pred: dict,
    target: dict,
    mano_layer=None,
    obj_mesh=None,
) -> dict:
    """Compute all metrics for Table 1 & 2."""
    metrics = {}

    if "hand_joints_3d" in pred and "joints_3d" in target:
        metrics["mpjpe"] = compute_mpjpe(pred["hand_joints_3d"], target["joints_3d"])
        metrics["pa_mpjpe"] = compute_pa_mpjpe(pred["hand_joints_3d"], target["joints_3d"])
        metrics["rrj"] = compute_rrj(pred["hand_joints_3d"], target["joints_3d"])
        errors = torch.norm(pred["hand_joints_3d"] - target["joints_3d"], dim=-1).cpu().numpy().flatten()
        metrics["aucj"] = compute_auc(errors)

    if mano_layer is not None and "hand_mano" in pred and "mano_params" in target:
        raise NotImplementedError("Implement vertex metrics with MANO layer (smplx)")

    if "obj_pose" in pred and "obj_trans" in target and "obj_quat" in target:
        if obj_mesh is not None:
            metrics["add"] = compute_add(...)

    return metrics
