"""
Qualitative visualization: overlay predictions on RGB.

Spec: "overlay predicted 3D pose directly onto original RGB 
using camera intrinsic calibration parameters... predicted MANO 3D 
hand mesh (778 vertices) and transformed YCB 3D object mesh 
oriented via predicted quaternion and translation should be rendered 
with dynamic transparency (e.g., alpha = 0.65) and distinct color overlays"
"""
from __future__ import annotations

import numpy as np


def overlay_predictions(
    rgb: np.ndarray,
    pred: dict,
    target: dict = None,
    intrinsics: np.ndarray = None,
    mano_layer=None,
    obj_meshes: dict = None,
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Overlay predictions (and optionally GT) on RGB image.
    
    Args:
        rgb: (H, W, 3) uint8 or float32 [0,1]
        pred: dict with hand_joints_3d, hand_mano, obj_pose, etc.
        target: optional ground truth for comparison
        intrinsics: (3, 3) camera matrix
        mano_layer: MANO layer to generate hand mesh
        obj_meshes: dict {class_id: trimesh.Trimesh} YCB object meshes
        alpha: mesh transparency (0.65 default)
    Returns:
        vis: (H, W, 3) uint8 image with overlay
    """
    raise NotImplementedError("Implement overlay_predictions")


def render_hand_object(
    rgb: np.ndarray,
    hand_verts: np.ndarray,
    hand_faces: np.ndarray,
    obj_verts: np.ndarray,
    obj_faces: np.ndarray,
    intrinsics: np.ndarray,
    hand_color: tuple = (0.0, 1.0, 0.0),
    obj_color: tuple = (1.0, 0.0, 0.0),
    alpha: float = 0.65,
) -> np.ndarray:
    """Off-screen rendering with pyrender for mesh overlay."""
    raise NotImplementedError("Implement render_hand_object")


def draw_keypoints_2d(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    connections: list = None,
    color: tuple = (0, 255, 0),
    radius: int = 3,
) -> np.ndarray:
    """Draw 2D keypoints on image."""
    raise NotImplementedError("Implement draw_keypoints_2d")


def create_comparison_figure(
    rgb: np.ndarray,
    pred: dict,
    target: dict,
    intrinsics: np.ndarray,
    mano_layer=None,
    obj_meshes: dict = None,
) -> np.ndarray:
    """Create comparison figure: GT left, Pred right (like Figure 4)."""
    raise NotImplementedError("Implement create_comparison_figure")
