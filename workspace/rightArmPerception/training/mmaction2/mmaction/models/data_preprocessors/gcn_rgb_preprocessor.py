# Copyright (c) OpenMMLab. All rights reserved.
"""Data pre-processor for the skeleton-RGB fusion model.

After DataLoader collation the ``inputs`` field is already a dict of stacked
tensors::

    inputs = {
        'keypoint': Tensor(N, T, V, C+1),   # skeleton, already normalised
        'imgs':     Tensor(N, T, 3, H, W),   # RGB frames, [0, 255] float32
    }

This pre-processor casts everything to the correct device / dtype and applies
ImageNet mean-std normalisation to the ``imgs`` tensor.
"""
import torch
from mmengine.model import BaseDataPreprocessor

from mmaction.registry import MODELS


@MODELS.register_module()
class GCNRGBDataPreprocessor(BaseDataPreprocessor):
    """Minimal data pre-processor for the skeleton-RGB fusion model.

    Responsibilities:
    * Cast all tensors to the target device (inherited from
      ``BaseDataPreprocessor.cast_data``).
    * Apply ImageNet mean / std normalisation to the ``imgs`` modality.

    The ``keypoint`` modality is left unchanged because
    ``PreNormalize2D`` handles skeleton normalisation in the pipeline.

    Args:
        img_mean (list[float]): Per-channel means in RGB order.
            Defaults to ImageNet values ``[123.675, 116.28, 103.53]``.
        img_std (list[float]): Per-channel stds in RGB order.
            Defaults to ImageNet values ``[58.395, 57.12, 57.375]``.
    """

    def __init__(
        self,
        img_mean=(123.675, 116.28, 103.53),
        img_std=(58.395, 57.12, 57.375),
    ) -> None:
        super().__init__()
        mean = torch.tensor(img_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(img_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        # Persistent=False: not saved in state_dict, just moved with .to()
        self.register_buffer('img_mean', mean, False)
        self.register_buffer('img_std', std, False)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Cast data to device and normalise RGB frames.

        Args:
            data (dict): Batch from the DataLoader.  Expected structure::

                {
                    'inputs': {
                        'keypoint': Tensor(N, T, V, C+1),
                        'imgs':     Tensor(N, T, 3, H, W),
                    },
                    'data_samples': [...],
                }

        Returns:
            dict: Same structure with ``imgs`` normalised.
        """
        data = self.cast_data(data)   # moves all tensors to self.device
        inputs = data['inputs']

        # Default collation may leave multi-modal inputs as a list of per-sample
        # tensors instead of an already stacked batch tensor.
        inputs['keypoint'] = self._ensure_batch_tensor(inputs['keypoint'])
        imgs = self._ensure_batch_tensor(inputs['imgs']).float()  # (N, T, 3, H, W)
        imgs = (imgs - self.img_mean) / self.img_std
        inputs['imgs'] = imgs

        data['inputs'] = inputs
        return data

    @staticmethod
    def _ensure_batch_tensor(value):
        """Convert a list/tuple of per-sample tensors into a batch tensor."""
        if torch.is_tensor(value):
            return value

        if isinstance(value, (list, tuple)):
            tensors = [
                item if torch.is_tensor(item) else torch.as_tensor(item)
                for item in value
            ]
            return torch.stack(tensors, dim=0)

        return torch.as_tensor(value)
