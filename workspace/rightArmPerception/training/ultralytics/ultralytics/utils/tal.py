# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy
from .torch_utils import TORCH_1_11


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos




class ConditionalPoseTaskAlignedAssigner(TaskAlignedAssigner):
    """A task-aligned assigner for pose estimation that considers both bbox IoU and keypoint OKS only for specific GT labels.
    
    This assigner only uses OKS (Object Keypoint Similarity) for GTs with a specific label (default is 0).
    For other GTs, it only uses bbox IoU for assignment.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, 
                 eps: float = 1e-9, oks_class_id: int = 0, oks_sigma: float = 0.03):
        """Initialize a ConditionalPoseTaskAlignedAssigner object with customizable hyperparameters.
        
        Args:
            topk (int): The number of top candidates to consider.
            num_classes (int): The number of object classes.
            alpha (float): The alpha parameter for the classification component of the task-aligned metric.
            beta (float): The beta parameter for the localization component of the task-aligned metric.
            eps (float): A small value to prevent division by zero.
            oks_class_id (int): The class ID for which OKS will be considered in assignment.
            oks_sigma (float): The sigma parameter for OKS calculation.
        """
        super().__init__(topk, num_classes, alpha, beta, eps)
        self.oks_class_id = oks_class_id
        self.oks_sigma = oks_sigma

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Compute the task-aligned assignment for pose estimation considering both bbox IoU and keypoint OKS only for specific GT labels.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).
            stride_tensor (torch.Tensor): Stride tensor for scaling coordinates, shape (num_total_anchors, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward_with_conditional_keypoints(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts, gt_kpts, stride_tensor)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in ConditionalPoseTaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            if pred_kpts is not None and gt_kpts is not None:
                result = self._forward_with_conditional_keypoints(*cpu_tensors, pred_kpts.cpu(), gt_kpts.cpu(), stride_tensor.cpu() if stride_tensor is not None else None)
            else:
                result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward_with_conditional_keypoints(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Compute the task-aligned assignment with conditional keypoints consideration."""
        mask_pos, align_metric, overlaps, oks_scores = self.get_pos_mask_with_conditional_keypoints(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, pred_kpts, gt_kpts, stride_tensor
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps_with_conditional_oks(
            mask_pos, overlaps, oks_scores, gt_labels, self.n_max_boxes
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_oks = (oks_scores * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # Combine bbox and keypoint information for normalization where appropriate
        combined_scores = torch.where(
            gt_labels.squeeze(-1).unsqueeze(-1) == self.oks_class_id,
            (pos_overlaps + pos_oks) / 2,  # Average of bbox IoU and keypoint OKS for specific class
            pos_overlaps  # Only bbox IoU for other classes
        )
        norm_align_metric = (align_metric * combined_scores / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask_with_conditional_keypoints(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Get positive mask for each ground truth box considering both bbox IoU and keypoint OKS only for specific GT labels."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric considering both bbox and keypoints conditionally, (b, max_num_obj, h*w)
        align_metric, overlaps, oks_scores = self.get_box_metrics_with_conditional_keypoints(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, pred_kpts, gt_kpts, stride_tensor
        )
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps, oks_scores

    def get_box_metrics_with_conditional_keypoints(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Compute alignment metric given predicted and ground truth bounding boxes, considering keypoints only for specific GT labels.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).
            stride_tensor (torch.Tensor): Stride tensor for scaling coordinates, shape (num_total_anchors, 1).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
            oks_scores (torch.Tensor): OKS scores between predicted and ground truth keypoints.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        oks_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Calculate OKS scores only for GTs with the specific class ID
        if pred_kpts is not None and gt_kpts is not None:
            # Expand gt_labels to match the mask_gt dimensions
            expanded_gt_labels = gt_labels.expand(-1, -1, na)  # (b, n_max_boxes, h*w)
            oks_mask = (expanded_gt_labels == self.oks_class_id).bool() & mask_gt
            
            # Calculate OKS for only the specific class
            if oks_mask.any():
                # Apply stride to predicted keypoints to match the scale of gt_kpts
                if stride_tensor is not None:
                    # Apply stride to predicted keypoints
                    # stride_tensor shape is (na, 1), need to expand for all batches and boxes
                    expanded_stride = stride_tensor.view(1, -1, 1, 1)  # (1, na, 1, 1)
                    scaled_pred_kpts = pred_kpts * expanded_stride  # mul pred_kpts by stride to match gt scale   [bs, num_total_anchors, num_kpts, 3]
                    oks_scores[oks_mask] = self.compute_oks(
                        pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[oks_mask], gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[oks_mask], 
                        scaled_pred_kpts.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1, -1)[oks_mask], gt_kpts.unsqueeze(2).expand(-1, -1, na, -1, -1)[oks_mask]
                    )
                else:
                    raise ValueError("stride_tensor must be provided when using keypoints for OKS calculation.")
                    # oks_scores[oks_mask] = self.compute_oks(
                    #     pd_boxes[oks_mask], gt_boxes[oks_mask], 
                    #     pred_kpts[oks_mask], gt_kpts[oks_mask]
                    # )

        # Combine bbox scores and overlaps with OKS for specific class only
        # For GTs with the specific class ID, use both bbox IoU and OKS
        # For other GTs, only use bbox IoU
        align_metric = torch.where(
            expanded_gt_labels == self.oks_class_id,
            bbox_scores.pow(self.alpha) * ((overlaps + oks_scores) / 2.0).pow(self.beta),  # Use both IoU and OKS
            bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)  # Use only IoU
        )
        
        return align_metric, overlaps, oks_scores

    def compute_oks(self, pred_boxes, gt_boxes, pred_kpts, gt_kpts, eps=1e-9):
        """Compute OKS (Object Keypoint Similarity) between predicted and ground truth keypoints.
        
        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes.
            gt_boxes (torch.Tensor): Ground truth bounding boxes.
            pred_kpts (torch.Tensor): Predicted keypoints (scaled to match gt).
            gt_kpts (torch.Tensor): Ground truth keypoints.
            eps (float): Small value for numerical stability.
            
        Returns:
            oks_scores (torch.Tensor): OKS scores.
        """
        # Calculate the area of the ground truth boxes
        areas = torch.prod(gt_boxes[:, 2:] - gt_boxes[:, :2], dim=1)  # width * height
        
        # Calculate distances between predicted and ground truth keypoints
        # Assuming keypoints format is [x, y, visibility]
        # Only consider visible keypoints (visibility > 0)
        if gt_kpts.dim() == 3:  # (num_samples, num_kpts, 3) where last dim is [x, y, vis]
            gt_kpts_xy = gt_kpts[:, :, :2]  # Extract x, y coordinates
            pred_kpts_xy = pred_kpts[:, :, :2]  # Extract x, y coordinates
            vis_mask = gt_kpts[:, :, 2] > 0  # Extract visibility mask
        else:  # (num_samples, num_kpts, 2) format
            gt_kpts_xy = gt_kpts
            pred_kpts_xy = pred_kpts
            vis_mask = torch.ones_like(gt_kpts[:, :, 0], dtype=torch.bool)  # All keypoints visible

        # Calculate squared distances between predicted and ground truth keypoints
        squared_distances = torch.sum((pred_kpts_xy - gt_kpts_xy) ** 2 / (self.oks_sigma ** 2), dim=2)  # (num_samples, num_kpts)

        # Calculate keypoint scales based on bounding box areas
        # Using sigma values based on the OKS definition
        scale = torch.sqrt(areas)[:, None]  # (num_samples, 1)
        scaled_squared_distances = squared_distances / (scale ** 2 + eps)

        # Calculate OKS for each keypoint
        oks_per_kpt = torch.exp(-scaled_squared_distances / 2)  # (num_samples, num_kpts)


        # Apply visibility mask and calculate average OKS
        oks_per_kpt = oks_per_kpt * vis_mask.float()
        # 计算可见关键点的总数
        vis_count = torch.sum(vis_mask, dim=1)  # (num_samples,)
        # 避免除以零，当没有可见关键点时，oks_scores设为0
        oks_scores = torch.where(vis_count > 0, 
                                torch.sum(oks_per_kpt, dim=1) / (vis_count + eps), 
                                torch.zeros_like(torch.sum(oks_per_kpt, dim=1)))

        return oks_scores

    def select_highest_overlaps_with_conditional_oks(self, mask_pos, overlaps, oks_scores, gt_labels, n_max_boxes):
        """Select anchor boxes with highest combined IoU and OKS when assigned to multiple ground truths, 
        but only consider OKS for specific GT labels."""
        # Expand gt_labels to match the dimensions of overlaps and oks_scores
        expanded_gt_labels = gt_labels.squeeze(-1).unsqueeze(-1).expand(-1, -1, overlaps.shape[-1])  # (b, n_max_boxes, h*w)
        
        # Combine IoU and OKS only for GTs with the specific class ID
        combined_scores = torch.where(
            expanded_gt_labels == self.oks_class_id,
            (overlaps + oks_scores) / 2,  # Average of bbox IoU and keypoint OKS for specific class
            overlaps  # Only bbox IoU for other classes
        )
        
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            # Use combined scores to select the best match, considering OKS only for specific class
            max_overlaps_idx = combined_scores.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos





class RTMO_ConditionalPoseTaskAlignedAssigner(TaskAlignedAssigner):
    """A task-aligned assigner for pose estimation that considers both bbox IoU and keypoint OKS only for specific GT labels.
    
    This assigner only uses OKS (Object Keypoint Similarity) for GTs with a specific label (default is 0).
    For other GTs, it only uses bbox IoU for assignment.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, 
                 eps: float = 1e-9, oks_class_id: int = 0, oks_sigma: float = 0.03):
        """Initialize a ConditionalPoseTaskAlignedAssigner object with customizable hyperparameters.
        
        Args:
            topk (int): The number of top candidates to consider.
            num_classes (int): The number of object classes.
            alpha (float): The alpha parameter for the classification component of the task-aligned metric.
            beta (float): The beta parameter for the localization component of the task-aligned metric.
            eps (float): A small value to prevent division by zero.
            oks_class_id (int): The class ID for which OKS will be considered in assignment.
            oks_sigma (float): The sigma parameter for OKS calculation.
        """
        super().__init__(topk, num_classes, alpha, beta, eps)
        self.oks_class_id = oks_class_id
        self.oks_sigma = oks_sigma

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Compute the task-aligned assignment for pose estimation considering both bbox IoU and keypoint OKS only for specific GT labels.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).
            stride_tensor (torch.Tensor): Stride tensor for scaling coordinates, shape (num_total_anchors, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward_with_conditional_keypoints(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts, gt_kpts, stride_tensor)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in ConditionalPoseTaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            if pred_kpts is not None and gt_kpts is not None:
                result = self._forward_with_conditional_keypoints(*cpu_tensors, pred_kpts.cpu(), gt_kpts.cpu(), stride_tensor.cpu() if stride_tensor is not None else None)
            else:
                result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward_with_conditional_keypoints(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Compute the task-aligned assignment with conditional keypoints consideration."""
        mask_pos, align_metric, overlaps, oks_scores = self.get_pos_mask_with_conditional_keypoints(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, pred_kpts, gt_kpts, stride_tensor
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps_with_conditional_oks(
            mask_pos, overlaps, oks_scores, gt_labels, self.n_max_boxes
        )

        # Assigned target
        target_labels, target_bboxes, target_kpts, target_scores = self.get_targets(gt_labels, gt_bboxes, gt_kpts, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_oks = (oks_scores * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # Combine bbox and keypoint information for normalization where appropriate
        combined_scores = torch.where(
            gt_labels.squeeze(-1).unsqueeze(-1) == self.oks_class_id,
            (pos_overlaps + pos_oks) / 2,  # Average of bbox IoU and keypoint OKS for specific class
            pos_overlaps  # Only bbox IoU for other classes
        )
        norm_align_metric = (align_metric * combined_scores / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx, target_kpts
    

    def get_targets(self, gt_labels, gt_bboxes,  gt_kpts, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_kpts = gt_kpts.view(-1, gt_kpts.shape[-2], gt_kpts.shape[-1])[target_gt_idx]  # (b, h*w, 17, 3)

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_kpts, target_scores




    def get_pos_mask_with_conditional_keypoints(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Get positive mask for each ground truth box considering both bbox IoU and keypoint OKS only for specific GT labels."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric considering both bbox and keypoints conditionally, (b, max_num_obj, h*w)
        align_metric, overlaps, oks_scores = self.get_box_metrics_with_conditional_keypoints(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, pred_kpts, gt_kpts, stride_tensor
        )
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps, oks_scores

    def get_box_metrics_with_conditional_keypoints(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None, stride_tensor=None):
        """Compute alignment metric given predicted and ground truth bounding boxes, considering keypoints only for specific GT labels.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).
            stride_tensor (torch.Tensor): Stride tensor for scaling coordinates, shape (num_total_anchors, 1).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
            oks_scores (torch.Tensor): OKS scores between predicted and ground truth keypoints.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        oks_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Calculate OKS scores only for GTs with the specific class ID
        if pred_kpts is not None and gt_kpts is not None:
            # Expand gt_labels to match the mask_gt dimensions
            expanded_gt_labels = gt_labels.expand(-1, -1, na)  # (b, n_max_boxes, h*w)
            oks_mask = (expanded_gt_labels == self.oks_class_id).bool() & mask_gt
            
            # Calculate OKS for only the specific class
            if oks_mask.any():
                # Apply stride to predicted keypoints to match the scale of gt_kpts
                if stride_tensor is not None:
                    # Apply stride to predicted keypoints
                    # stride_tensor shape is (na, 1), need to expand for all batches and boxes
                    expanded_stride = stride_tensor.view(1, -1, 1, 1)  # (1, na, 1, 1)
                    scaled_pred_kpts = pred_kpts * expanded_stride  # mul pred_kpts by stride to match gt scale   [bs, num_total_anchors, num_kpts, 3]
                    oks_scores[oks_mask] = self.compute_oks(
                        pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[oks_mask], gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[oks_mask], 
                        scaled_pred_kpts.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1, -1)[oks_mask], gt_kpts.unsqueeze(2).expand(-1, -1, na, -1, -1)[oks_mask]
                    )
                else:
                    raise ValueError("stride_tensor must be provided when using keypoints for OKS calculation.")
                    # oks_scores[oks_mask] = self.compute_oks(
                    #     pd_boxes[oks_mask], gt_boxes[oks_mask], 
                    #     pred_kpts[oks_mask], gt_kpts[oks_mask]
                    # )

        # Combine bbox scores and overlaps with OKS for specific class only
        # For GTs with the specific class ID, use both bbox IoU and OKS
        # For other GTs, only use bbox IoU
        align_metric = torch.where(
            expanded_gt_labels == self.oks_class_id,
            bbox_scores.pow(self.alpha) * ((overlaps + oks_scores) / 2.0).pow(self.beta),  # Use both IoU and OKS
            bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)  # Use only IoU
        )
        
        return align_metric, overlaps, oks_scores

    def compute_oks(self, pred_boxes, gt_boxes, pred_kpts, gt_kpts, eps=1e-9):
        """Compute OKS (Object Keypoint Similarity) between predicted and ground truth keypoints.
        
        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes.
            gt_boxes (torch.Tensor): Ground truth bounding boxes.
            pred_kpts (torch.Tensor): Predicted keypoints (scaled to match gt).
            gt_kpts (torch.Tensor): Ground truth keypoints.
            eps (float): Small value for numerical stability.
            
        Returns:
            oks_scores (torch.Tensor): OKS scores.
        """
        # Calculate the area of the ground truth boxes
        areas = torch.prod(gt_boxes[:, 2:] - gt_boxes[:, :2], dim=1)  # width * height
        
        # Calculate distances between predicted and ground truth keypoints
        # Assuming keypoints format is [x, y, visibility]
        # Only consider visible keypoints (visibility > 0)
        if gt_kpts.dim() == 3:  # (num_samples, num_kpts, 3) where last dim is [x, y, vis]
            gt_kpts_xy = gt_kpts[:, :, :2]  # Extract x, y coordinates
            pred_kpts_xy = pred_kpts[:, :, :2]  # Extract x, y coordinates
            vis_mask = gt_kpts[:, :, 2] > 0  # Extract visibility mask
        else:  # (num_samples, num_kpts, 2) format
            gt_kpts_xy = gt_kpts
            pred_kpts_xy = pred_kpts
            vis_mask = torch.ones_like(gt_kpts[:, :, 0], dtype=torch.bool)  # All keypoints visible

        # Calculate squared distances between predicted and ground truth keypoints
        squared_distances = torch.sum((pred_kpts_xy - gt_kpts_xy) ** 2 / (self.oks_sigma ** 2), dim=2)  # (num_samples, num_kpts)

        # Calculate keypoint scales based on bounding box areas
        # Using sigma values based on the OKS definition
        scale = torch.sqrt(areas)[:, None]  # (num_samples, 1)
        scaled_squared_distances = squared_distances / (scale ** 2 + eps)

        # Calculate OKS for each keypoint
        oks_per_kpt = torch.exp(-scaled_squared_distances / 2)  # (num_samples, num_kpts)


        # Apply visibility mask and calculate average OKS
        oks_per_kpt = oks_per_kpt * vis_mask.float()
        # 计算可见关键点的总数
        vis_count = torch.sum(vis_mask, dim=1)  # (num_samples,)
        # 避免除以零，当没有可见关键点时，oks_scores设为0
        oks_scores = torch.where(vis_count > 0, 
                                torch.sum(oks_per_kpt, dim=1) / (vis_count + eps), 
                                torch.zeros_like(torch.sum(oks_per_kpt, dim=1)))

        return oks_scores

    def select_highest_overlaps_with_conditional_oks(self, mask_pos, overlaps, oks_scores, gt_labels, n_max_boxes):
        """Select anchor boxes with highest combined IoU and OKS when assigned to multiple ground truths, 
        but only consider OKS for specific GT labels."""
        # Expand gt_labels to match the dimensions of overlaps and oks_scores
        expanded_gt_labels = gt_labels.squeeze(-1).unsqueeze(-1).expand(-1, -1, overlaps.shape[-1])  # (b, n_max_boxes, h*w)
        
        # Combine IoU and OKS only for GTs with the specific class ID
        combined_scores = torch.where(
            expanded_gt_labels == self.oks_class_id,
            (overlaps + oks_scores) / 2,  # Average of bbox IoU and keypoint OKS for specific class
            overlaps  # Only bbox IoU for other classes
        )
        
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            # Use combined scores to select the best match, considering OKS only for specific class
            max_overlaps_idx = combined_scores.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos





# 没有验证，使用时要自己检查code
class PoseTaskAlignedAssigner(TaskAlignedAssigner):
    """A task-aligned assigner for pose estimation that considers both bbox IoU and keypoint OKS."""

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """Initialize a PoseTaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__(topk, num_classes, alpha, beta, eps)

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None):
        """Compute the task-aligned assignment for pose estimation considering both bbox IoU and keypoint OKS.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward_with_keypoints(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts, gt_kpts)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in PoseTaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            if pred_kpts is not None and gt_kpts is not None:
                result = self._forward_with_keypoints(*cpu_tensors, pred_kpts.cpu(), gt_kpts.cpu())
            else:
                result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward_with_keypoints(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None):
        """Compute the task-aligned assignment with keypoints consideration."""
        mask_pos, align_metric, overlaps, oks_scores = self.get_pos_mask_with_keypoints(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, pred_kpts, gt_kpts
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps_with_oks(mask_pos, overlaps, oks_scores, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_oks = (oks_scores * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # Combine bbox and keypoint information for normalization
        combined_scores = (overlaps + oks_scores) / 2  # average of bbox IoU and keypoint OKS
        norm_align_metric = (align_metric * combined_scores / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask_with_keypoints(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, pred_kpts=None, gt_kpts=None):
        """Get positive mask for each ground truth box considering both bbox IoU and keypoint OKS."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric considering both bbox and keypoints, (b, max_num_obj, h*w)
        align_metric, overlaps, oks_scores = self.get_box_metrics_with_keypoints(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, pred_kpts, gt_kpts
        )
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps, oks_scores
    
    def get_box_metrics_with_keypoints(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, pred_kpts=None, gt_kpts=None):
        """Compute alignment metric given predicted and ground truth bounding boxes and keypoints.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification, localization and keypoint similarity.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
            oks_scores (torch.Tensor): OKS scores between predicted and ground truth keypoints.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        
        # 计算bbox IoU
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # 如果提供了关键点信息，计算OKS
        oks_scores = torch.ones_like(overlaps)  # 初始化为1，如果没有关键点则不影响结果
        if pred_kpts is not None and gt_kpts is not None:
            oks_scores = self.compute_oks_for_assignment(pred_kpts, gt_kpts, mask_gt)

        # 结合bbox IoU和OKS
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta * 0.5) * oks_scores.pow(self.beta * 0.5)
        return align_metric, overlaps, oks_scores

    def compute_oks_for_assignment(self, pred_kpts, gt_kpts, mask_gt):
        """Compute OKS (Object Keypoint Similarity) for assignment.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints with shape (bs, num_total_anchors, num_kpts, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints with shape (bs, n_max_boxes, num_kpts, 3).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            oks_scores (torch.Tensor): OKS scores with shape (bs, n_max_boxes, h*w).
        """
        bs, n_max_boxes, na = mask_gt.shape
        device = pred_kpts.device
        oks_scores = torch.zeros([bs, n_max_boxes, na], dtype=pred_kpts.dtype, device=device)

        # 计算每个anchor的bbox面积用于OKS归一化
        # 这里简化处理，使用gt bbox的面积
        gt_areas = torch.abs((gt_kpts[..., 0].max(dim=2, keepdim=True)[0] - gt_kpts[..., 0].min(dim=2, keepdim=True)[0]) * 
                            (gt_kpts[..., 1].max(dim=2, keepdim=True)[0] - gt_kpts[..., 1].min(dim=2, keepdim=True)[0]))
        
        # 为避免除零错误，设置最小面积
        gt_areas = torch.clamp(gt_areas, min=1.0)

        for b in range(bs):
            for obj_idx in range(n_max_boxes):
                if mask_gt[b, obj_idx].any():  # 如果该gt object存在
                    # 获取gt keypoints
                    gt_kpt = gt_kpts[b, obj_idx]  # (num_kpts, 3)
                    
                    # 获取对应anchor的预测关键点
                    valid_anchors = mask_gt[b, obj_idx]
                    if valid_anchors.any():
                        pred_kpt_batch = pred_kpts[b, valid_anchors]  # (num_valid_anchors, num_kpts, 3)
                        
                        # 计算OKS
                        # OKS = exp(-d²/2s²k²)
                        # 其中d是关键点距离，s是目标尺度，k是关键点特定常数
                        gt_kpt_xy = gt_kpt[:, :2]  # (num_kpts, 2)
                        gt_kpt_vis = gt_kpt[:, 2] if gt_kpt.shape[1] == 3 else torch.ones(gt_kpt.shape[0], device=device)
                        
                        # 只考虑可见的关键点
                        visible_mask = gt_kpt_vis > 0
                        if visible_mask.any():
                            # 计算距离
                            dists = torch.norm(
                                pred_kpt_batch[:, visible_mask, :2] - 
                                gt_kpt_xy[visible_mask].unsqueeze(0), 
                                dim=-1
                            )  # (num_valid_anchors, num_visible_kpts)
                            
                            # 使用目标尺度归一化
                            area = gt_areas[b, obj_idx].unsqueeze(0).unsqueeze(-1)  # (1, 1)
                            
                            # OKS计算，使用COCO的sigma值或自定义sigma
                            sigmas = torch.tensor(OKS_SIGMA, device=device) if gt_kpt.shape[0] == 17 else torch.ones(visible_mask.sum(), device=device) * 0.02
                            sigmas = sigmas[visible_mask] if len(sigmas) == len(visible_mask) else torch.ones(visible_mask.sum(), device=device) * 0.02
                            
                            # 计算OKS
                            variances = (2 * sigmas ** 2).unsqueeze(0) * area * 2
                            oks_vals = torch.exp(-dists ** 2 / variances)  # (num_valid_anchors, num_visible_kpts)
                            
                            # 平均OKS
                            oks_vals = oks_vals.mean(dim=-1)  # (num_valid_anchors,)
                            
                            # 将OKS值赋给对应位置
                            oks_scores[b, obj_idx, valid_anchors] = oks_vals
        
        return oks_scores

    def select_highest_overlaps_with_oks(self, mask_pos, overlaps, oks_scores, n_max_boxes):
        """Select anchor boxes with highest combined IoU and OKS when assigned to multiple ground truths."""
        # Combine IoU and OKS for better assignment
        combined_scores = (overlaps + oks_scores) / 2  # Average of bbox IoU and keypoint OKS
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            # Use combined scores to select the best match
            max_overlaps_idx = combined_scores.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos












class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)










