# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops, colorstr, emojis, nms
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode
from ultralytics.utils.files import increment_path

import torch
from pathlib import Path
import cv2

class PosePredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific warnings
        for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.pose import PosePredictor
            >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
            >>> predictor = PosePredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction, including keypoints.

        Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
                the number of detections, K is the number of keypoints, and D is the keypoint dimension.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and
                keypoints.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result



class RTMO_PosePredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific warnings
        for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.pose import PosePredictor
            >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
            >>> predictor = PosePredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "RTMO_Pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )
    
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
            nc=0 if self.args.task == "detect" else self.model.model.model[-1].nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.model.model.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs], pos_feats


    def postprocess(self, preds, pose_feats: torch.Tensor, head_dcc_pose_test, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        # preds = nms.non_max_suppression(
        #     preds,
        #     self.args.conf,
        #     self.args.iou,
        #     self.args.classes,
        #     self.args.agnostic_nms,
        #     max_det=self.args.max_det,
        #     nc=0 if self.args.task == "detect" else len(self.model.names),
        #     end2end=getattr(self.model, "end2end", False),
        #     rotated=self.args.task == "obb",
        #     return_idxs=save_feats,
        # )

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


        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results
    

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
    
    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Stream real-time inference on camera feed and save results to file.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, self.model.ch, *self.imgsz)
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds, pose_feats = self.inference(im, *args, **kwargs)
                    # preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, pose_feats, self.model.model.model[-1].forward_dcc_test, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # close any open windows

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")





    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction, including keypoints.

        Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
                the number of detections, K is the number of keypoints, and D is the keypoint dimension.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and
                keypoints.
        """
        result = super().construct_result(torch.cat([pred['bboxes'], pred['conf'][:, None], pred['cls'][:, None]], dim=1), img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred['keypoints']#[:, 6:].view(pred['keypoints'].shape[0], *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result

