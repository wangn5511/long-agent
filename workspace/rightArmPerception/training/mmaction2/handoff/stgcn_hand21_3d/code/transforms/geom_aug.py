# Copyright (c) OpenMMLab. All rights reserved.
"""Geometric and detector-noise augmentations for the RGB-gated fusion pipeline.

Three transforms used by ``stgcn_hand21-rgb-bs64-gated-aug.py`` to address two
real-world failure modes observed at deployment:

* ``JointRandomRotate`` -- synchronous rotation of ``keypoint`` xy,
  ``global_traj_source`` xy and ``img_shape``.  Stores the rotation
  parameters in ``results['_aug_rotation_angle']`` and
  ``results['_aug_rotation_discrete_k']`` so that ``DecodeVideoROI`` can
  rotate video frames by the same transform.  Supports lossless discrete
  rotations (k*90 degrees) AND continuous small angles.  The discrete
  branch exactly matches a physically-rotated camera at deployment.

* ``JointKeypointJitter`` -- Gaussian noise on ``keypoint`` /
  ``global_traj_source`` xy, simulating degraded detector accuracy at
  far hand-camera distance.

* ``RGBQualityJitter`` -- random downsample-then-upsample on
  ``results['imgs']``, simulating the low-detail RGB ROI we get when the
  source hand crop has few pixels.

Discrete rotation conventions match ``np.rot90`` (CCW, ``k=1`` = 90 deg)
so that ``DecodeVideoROI`` can use ``np.rot90`` losslessly on each frame.
Both branches use visual CCW (matching ``np.rot90``).  ``DecodeVideoROI``
should use ``np.rot90`` for the discrete case and
``cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)`` (POSITIVE angle) for
the continuous case.
"""

from typing import Callable, Dict, Sequence, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_discrete_map(
    k: int, h: int, w: int
) -> Tuple[Callable[[np.ndarray], np.ndarray], Tuple[int, int]]:
    """Return ``(map_xy, (new_h, new_w))`` for a ``k * 90`` deg rotation
    matching ``np.rot90(image, k=k)`` exactly."""
    if k == 1:  # 90 deg CCW
        new_h, new_w = w, h

        def fn(xy: np.ndarray) -> np.ndarray:
            out = np.empty_like(xy)
            out[..., 0] = xy[..., 1]
            out[..., 1] = (w - 1) - xy[..., 0]
            return out

    elif k == 2:  # 180 deg
        new_h, new_w = h, w

        def fn(xy: np.ndarray) -> np.ndarray:
            out = np.empty_like(xy)
            out[..., 0] = (w - 1) - xy[..., 0]
            out[..., 1] = (h - 1) - xy[..., 1]
            return out

    elif k == 3:  # 270 deg CCW (== 90 deg CW)
        new_h, new_w = w, h

        def fn(xy: np.ndarray) -> np.ndarray:
            out = np.empty_like(xy)
            out[..., 0] = (h - 1) - xy[..., 1]
            out[..., 1] = xy[..., 0]
            return out

    else:
        raise ValueError(f"k must be in (1, 2, 3); got {k}")
    return fn, (new_h, new_w)


def _make_continuous_map(
    angle_deg: float, h: int, w: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a closure that rotates xy by ``angle_deg`` visual-CCW around
    the image centre (matches np.rot90 at multiples of 90 deg and
    cv2.getRotationMatrix2D with positive angle for arbitrary angles)."""
    cx, cy = w / 2.0, h / 2.0
    rad = float(np.deg2rad(angle_deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))

    def fn(xy: np.ndarray) -> np.ndarray:
        x = xy[..., 0] - cx
        y = xy[..., 1] - cy
        out = np.empty_like(xy)
        out[..., 0] = c * x + s * y + cx
        out[..., 1] = -s * x + c * y + cy
        return out

    return fn


def _apply_xy_inplace(arr_key: str, results: Dict, map_xy: Callable) -> None:
    """Apply ``map_xy`` to the first 2 channels of ``results[arr_key]`` if it
    exists and has at least 2 trailing channels."""
    val = results.get(arr_key)
    if val is None:
        return
    arr = np.asarray(val)
    if arr.ndim < 2 or arr.shape[-1] < 2:
        return
    arr = arr.astype(np.float32, copy=True)
    arr[..., :2] = map_xy(arr[..., :2])
    results[arr_key] = arr


# ---------------------------------------------------------------------------
# Public transforms
# ---------------------------------------------------------------------------


@TRANSFORMS.register_module()
class JointRandomRotate(BaseTransform):
    """Rotate ``keypoint`` xy and ``global_traj_source`` xy around the image
    centre; update ``img_shape``; record the rotation parameters for
    ``DecodeVideoROI`` so video frames can be rotated to match.

    MUST be placed in the pipeline **before** ``DecodeVideoROI``.

    Args:
        discrete_prob (float): Probability of sampling a lossless
            ``k * 90`` deg rotation (k uniform in ``discrete_ks``).
            Defaults to ``0.4``.
        discrete_ks (Sequence[int]): Choices for ``k`` (each interpreted
            modulo 4).  Defaults to ``(0, 1, 2, 3)`` -- 0/90/180/270 deg.
        continuous_prob (float): Probability of sampling a continuous
            small angle.  Defaults to ``0.4``.
        continuous_max_deg (float): Max absolute angle (degrees) for the
            continuous branch.  Defaults to ``15.0``.

    With the defaults, 20% of samples are left unrotated, 40% get a
    discrete rotation (10% per cardinal multiple), and 40% get a
    continuous small angle in +/- 15 deg.
    """

    def __init__(
        self,
        discrete_prob: float = 0.4,
        discrete_ks: Sequence[int] = (0, 1, 2, 3),
        continuous_prob: float = 0.4,
        continuous_max_deg: float = 15.0,
    ) -> None:
        if discrete_prob < 0 or continuous_prob < 0:
            raise ValueError("Probabilities must be non-negative.")
        if discrete_prob + continuous_prob > 1.0 + 1e-9:
            raise ValueError(
                f"discrete_prob + continuous_prob = "
                f"{discrete_prob + continuous_prob:.3f} must be <= 1"
            )
        self.discrete_prob = float(discrete_prob)
        self.discrete_ks = tuple(int(k) % 4 for k in discrete_ks)
        if not self.discrete_ks:
            raise ValueError("`discrete_ks` must contain at least one value.")
        self.continuous_prob = float(continuous_prob)
        self.continuous_max_deg = float(continuous_max_deg)

    # ------------------------------------------------------------------
    def _sample(self) -> Tuple[str, float, int]:
        """Return ``(mode, angle_deg, k)``; mode is one of
        ``'discrete'``, ``'continuous'``, ``'none'``."""
        r = float(np.random.rand())
        if r < self.discrete_prob:
            k = int(self.discrete_ks[np.random.randint(len(self.discrete_ks))])
            return ("discrete" if k != 0 else "none", float(k * 90), k)
        if r < self.discrete_prob + self.continuous_prob:
            ang = float(
                np.random.uniform(-self.continuous_max_deg, self.continuous_max_deg)
            )
            return ("continuous", ang, 0)
        return ("none", 0.0, 0)

    # ------------------------------------------------------------------
    def transform(self, results: Dict) -> Dict:
        if "img_shape" not in results:
            return results
        mode, angle_deg, k = self._sample()
        if mode == "none":
            results.setdefault("_aug_rotation_angle", 0.0)
            results.setdefault("_aug_rotation_discrete_k", 0)
            return results

        h, w = int(results["img_shape"][0]), int(results["img_shape"][1])
        if mode == "discrete":
            map_xy, (new_h, new_w) = _make_discrete_map(k, h, w)
        else:  # continuous
            map_xy = _make_continuous_map(angle_deg, h, w)
            new_h, new_w = h, w

        _apply_xy_inplace("keypoint", results, map_xy)
        _apply_xy_inplace("global_traj_source", results, map_xy)

        # Pre-stored bbox positions are no longer valid after rotation;
        # force DecodeVideoROI to fall back to the rotated keypoints.
        if results.get("bbox_per_frame") is not None:
            results["bbox_per_frame"] = [None] * len(results["bbox_per_frame"])

        if (new_h, new_w) != (h, w):
            shape_in = results["img_shape"]
            tail: Tuple = ()
            if hasattr(shape_in, "__len__") and len(shape_in) > 2:
                tail = tuple(shape_in[2:])
            results["img_shape"] = (new_h, new_w) + tail

        results["_aug_rotation_angle"] = float(angle_deg)
        results["_aug_rotation_discrete_k"] = int(k)
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"discrete_prob={self.discrete_prob}, "
            f"discrete_ks={self.discrete_ks}, "
            f"continuous_prob={self.continuous_prob}, "
            f"continuous_max_deg={self.continuous_max_deg})"
        )


@TRANSFORMS.register_module()
class JointKeypointJitter(BaseTransform):
    """Add Gaussian noise to ``keypoint`` xy (and optionally
    ``global_traj_source`` xy) to simulate degraded detector accuracy at
    far hand-camera distance.

    Noise std is expressed in **fractions of image diagonal**, so the same
    relative perturbation applies regardless of input resolution.

    Args:
        sigma_keypoint (float): Std of keypoint xy noise as a fraction of
            image diagonal.  ``0.0`` disables.  Defaults to ``0.005``
            (~11 px on a 1080x1920 frame).
        sigma_traj (float): Same, for ``global_traj_source``.  Defaults
            to ``0.0`` (disabled -- the trajectory source is already a
            smoothed signal).
        prob (float): Apply with this probability per sample.  Defaults
            to ``0.5``.
    """

    def __init__(
        self,
        sigma_keypoint: float = 0.005,
        sigma_traj: float = 0.0,
        prob: float = 0.5,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("`prob` must be in [0, 1].")
        if sigma_keypoint < 0 or sigma_traj < 0:
            raise ValueError("Sigmas must be non-negative.")
        self.sigma_keypoint = float(sigma_keypoint)
        self.sigma_traj = float(sigma_traj)
        self.prob = float(prob)

    def transform(self, results: Dict) -> Dict:
        if float(np.random.rand()) >= self.prob:
            return results
        if "img_shape" not in results:
            return results
        h, w = float(results["img_shape"][0]), float(results["img_shape"][1])
        diag = float(np.sqrt(h * h + w * w))

        def _add_noise(arr_key: str, sigma_frac: float) -> None:
            if sigma_frac <= 0:
                return
            val = results.get(arr_key)
            if val is None:
                return
            arr = np.asarray(val)
            if arr.ndim < 2 or arr.shape[-1] < 2:
                return
            arr = arr.astype(np.float32, copy=True)
            noise = np.random.normal(
                loc=0.0,
                scale=sigma_frac * diag,
                size=arr[..., :2].shape,
            ).astype(np.float32)
            arr[..., :2] = arr[..., :2] + noise
            results[arr_key] = arr

        _add_noise("keypoint", self.sigma_keypoint)
        _add_noise("global_traj_source", self.sigma_traj)
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sigma_keypoint={self.sigma_keypoint}, "
            f"sigma_traj={self.sigma_traj}, "
            f"prob={self.prob})"
        )


@TRANSFORMS.register_module()
class RGBQualityJitter(BaseTransform):
    """Randomly downsample-then-upsample ``results['imgs']`` to simulate
    the low-detail RGB ROI we get when the hand is far from the camera.

    Operates on a ``(T, C, H, W)`` float32 array in ``[0, 255]`` (the
    format produced by ``DecodeVideoROI``).

    Args:
        min_scale (float): Lower bound on the downsample factor (relative
            to current ROI side).  Defaults to ``0.33`` (32x32 from 96x96).
        max_scale (float): Upper bound (must be <= 1).  Defaults to
            ``0.66``.
        prob (float): Apply with this probability per sample.  Defaults
            to ``0.5``.
        interpolation_down (int): cv2 interpolation flag for the down
            step.  Defaults to ``cv2.INTER_AREA`` (recommended for shrink).
        interpolation_up (int): cv2 interpolation flag for the up step.
            Defaults to ``cv2.INTER_LINEAR``.
    """

    def __init__(
        self,
        min_scale: float = 0.33,
        max_scale: float = 0.66,
        prob: float = 0.5,
        interpolation_down: int = cv2.INTER_AREA,
        interpolation_up: int = cv2.INTER_LINEAR,
    ) -> None:
        if not (0.0 < min_scale <= max_scale <= 1.0):
            raise ValueError("Expected 0 < min_scale <= max_scale <= 1.")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("`prob` must be in [0, 1].")
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.prob = float(prob)
        self.interp_down = int(interpolation_down)
        self.interp_up = int(interpolation_up)

    def transform(self, results: Dict) -> Dict:
        if float(np.random.rand()) >= self.prob:
            return results
        imgs = results.get("imgs")
        if imgs is None:
            return results
        imgs = np.asarray(imgs)
        if imgs.ndim != 4 or imgs.shape[1] != 3:
            return results  # expect (T, 3, H, W)

        t, _, h, w = imgs.shape
        scale = float(np.random.uniform(self.min_scale, self.max_scale))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if new_h >= h and new_w >= w:
            return results

        out = np.empty_like(imgs)
        for ti in range(t):
            frame_hwc = imgs[ti].transpose(1, 2, 0)
            small = cv2.resize(
                frame_hwc, (new_w, new_h), interpolation=self.interp_down
            )
            back = cv2.resize(small, (w, h), interpolation=self.interp_up)
            out[ti] = back.transpose(2, 0, 1)
        results["imgs"] = out
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"min_scale={self.min_scale}, max_scale={self.max_scale}, "
            f"prob={self.prob})"
        )
