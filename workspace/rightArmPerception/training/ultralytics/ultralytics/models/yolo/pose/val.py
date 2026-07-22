# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import json

import torch.distributed as dist

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import RTMO_PoseModel
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis, nms
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model


class PoseValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized
    metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints.
        kpt_shape (list[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        _prepare_batch: Prepare a batch for processing by converting keypoints to float and scaling to original
            dimensions.
        _prepare_pred: Prepare and scale keypoints in predictions for pose processing.
        _process_batch: Return correct prediction matrix by computing Intersection over Union (IoU) between detections
            and ground truth.
        plot_val_samples: Plot and save validation set samples with ground truth bounding boxes and keypoints.
        plot_predictions: Plot and save model predictions with bounding boxes and keypoints.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate object detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (list, optional): List of callback functions to be executed during validation.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseValidator
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
            >>> validator = PoseValidator(args=args)
            >>> validator()

        Notes:
            This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
            for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS
            due to a known bug with pose models.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> dict[str, torch.Tensor]:
        """Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

        This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of
        predictions and reshaping them according to the keypoint shape configuration. The keypoints are reshaped from a
        flattened format to the proper dimensional structure (typically [N, 17, 3] for COCO pose format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing bounding boxes, confidence
                scores, class predictions, and keypoint data.

        Returns:
            (dict[torch.Tensor]): Dict of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

        Notes:
            If no keypoints are present in a prediction (empty keypoints), that prediction is skipped and continues
            to the next one. The keypoints are extracted from the 'extra' field which contains additional
            task-specific data beyond basic detection.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # remove extra if exists
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to original image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground
        truth.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels, 'bboxes'
                for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for pose true
                positives across 10 IoU levels.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls' and 'keypoints.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format to COCO
        format, and appends the results to the internal JSON dictionary (self.jdict).

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'keypoints'
                tensors.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "kpts": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate object detection model using COCO JSON format."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])






class RTMO_PoseValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized
    metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints.
        kpt_shape (list[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        _prepare_batch: Prepare a batch for processing by converting keypoints to float and scaling to original
            dimensions.
        _prepare_pred: Prepare and scale keypoints in predictions for pose processing.
        _process_batch: Return correct prediction matrix by computing Intersection over Union (IoU) between detections
            and ground truth.
        plot_val_samples: Plot and save validation set samples with ground truth bounding boxes and keypoints.
        plot_predictions: Plot and save model predictions with bounding boxes and keypoints.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate object detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (list, optional): List of callback functions to be executed during validation.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseValidator
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
            >>> validator = PoseValidator(args=args)
            >>> validator()

        Notes:
            This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
            for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS
            due to a known bug with pose models.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "RTMO_Pose"
        self.metrics = PoseMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt


    def nms_postprocess(self, preds: torch.Tensor, pos_feature: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains 'bboxes', 'conf',
                'cls', and 'extra' tensors.
        """
        outputs, pos_feats = nms.non_max_suppression_with_pos_feature(
            preds,
            pos_feature,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs], pos_feats



    def postprocess(self, preds: torch.Tensor, pose_feats: torch.Tensor, head_dcc_pose_test) -> dict[str, torch.Tensor]:
        """Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

        This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of
        predictions and reshaping them according to the keypoint shape configuration. The keypoints are reshaped from a
        flattened format to the proper dimensional structure (typically [N, 17, 3] for COCO pose format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing bounding boxes, confidence
                scores, class predictions, and keypoint data.

        Returns:
            (dict[torch.Tensor]): Dict of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

        Notes:
            If no keypoints are present in a prediction (empty keypoints), that prediction is skipped and continues
            to the next one. The keypoints are extracted from the 'extra' field which contains additional
            task-specific data beyond basic detection.
        """
        preds, pos_feats = self.nms_postprocess(preds, pose_feats)   # box的nms； nms要选出box对应的feature
        kpts = head_dcc_pose_test(pos_feats, preds)  #(pose_feats, bbox_cs)

        visibility = torch.zeros_like(kpts[..., :1])  # 创建与原张量相同设备和类型的全0张量
        # cls为1的visibility设为1
        visibility[torch.cat([i['cls'] for i in preds], dim=0) == 0] = 1
        
        kpts_expanded = torch.cat([kpts, visibility], dim=-1)




        # 将kpts_expanded与preds的维度对齐
        bbox_lengths = [pred['bboxes'].shape[0] for pred in preds]
        kpts_split_list = torch.split(kpts_expanded, bbox_lengths, dim=0)



        for pred, kpts in zip(preds, kpts_split_list):
            pred["keypoints"] = kpts  # remove extra if exists
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to original image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground
        truth.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels, 'bboxes'
                for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for pose true
                positives across 10 IoU levels.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        hand_gt_mask = (gt_cls == 0)
        hand_pred_mask = (preds["cls"] == 0)

        hand_gt_cls = gt_cls[hand_gt_mask]
        pred_cls = preds["cls"]#[hand_pred_mask]

        if hand_gt_cls.shape[0] == 0 or pred_cls.shape[0] == 0:
            tp_p = np.zeros((pred_cls.shape[0], self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"][hand_gt_mask])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"][hand_gt_mask], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(pred_cls, hand_gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp



    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls' and 'keypoints.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format to COCO
        format, and appends the results to the internal JSON dictionary (self.jdict).

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'keypoints'
                tensors.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "kpts": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate object detection model using COCO JSON format."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
    
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # validate non-compiled original model to avoid issues
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds, pose_feats = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                # if self.training:
                if False:   # 挑过val 中的loss计算
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:  # 处理dcc的预测
                if isinstance(model, RTMO_PoseModel):  # 训练过程中的val
                    preds = self.postprocess(preds, pose_feats, model.model[21].forward_dcc_test)
                else:  # 非训练过程的val
                    preds = self.postprocess(preds, pose_feats, model.model.model[21].forward_dcc_test)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            # Reduce loss across all GPUs
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {**stats, **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats


