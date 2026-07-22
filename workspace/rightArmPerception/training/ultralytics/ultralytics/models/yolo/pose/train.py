# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel, RTMO_PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER



import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)





class PoseTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )


    def fc_load_weights(self, weights: dict[str, Any], pose_model: PoseModel) -> PoseModel:
        """Load fully connected layer weights from a checkpoint file.

        Args:
            weights (dict): Dictionary containing the weights, with "fc_ckpt" key pointing to the checkpoint file.
            model (PoseModel): The pose estimation model to load the weights into.

        Returns:
            (PoseModel): The model with loaded fully connected layer weights.
        """
        import torch
        detection_state_dict = torch.load(weights["fc_ckpt"], weights_only=False)["model"].state_dict()
        print("开始迁移权重...")
        pose_state_dict = pose_model.state_dict()
        
        
        transferred = 0
        skipped = 0
        
        for key, value in pose_state_dict.items():
            if key in detection_state_dict:
                # 确保形状匹配  
                if detection_state_dict[key].shape == value.shape:
                    pose_state_dict[key] = detection_state_dict[key]
                    transferred += 1
                else:
                    print(f"形状不匹配, {key}, 检测模型形状: {detection_state_dict[key].shape}, Pose模型形状: {value.shape}")
                    if key == "model.model.1.conv.weight" or key == "model.model.3.conv.weight":
                        print(f"注意: 该权重形状不匹配，是因为groups!=1导致的。")

                        if key == "model.model.1.conv.weight":
                            # 由groups=2的权重构造groups=1的权重
                            old_weight = pose_state_dict[key]
                            c_out, c_in, k_h, k_w = old_weight.shape
                            G = 2  # 原始groups数量
                            weight_g2 = detection_state_dict[key]  # [32, 8, 3, 3]
                            weight_g1 = torch.zeros(c_out, c_in, k_h, k_w)  # [32, 16, 3, 3]
                            for g in range(G):
                                in_start = g * (c_in // G)          # 0, 8
                                in_end   = (g + 1) * (c_in // G)    # 8, 16

                                out_start = g * (c_out // G)        # 0, 16
                                out_end   = (g + 1) * (c_out // G)  # 16, 32

                                # 从 weight_g2 中取出第 g 组的权重（连续存储）
                                # 注意：weight_g2 的输出通道是连续的 0~31，所以第 g 组就是 [out_start:out_end]
                                weight_g1[out_start:out_end, in_start:in_end, :, :] = weight_g2[out_start:out_end, :, :, :]

                            pose_state_dict[key] = weight_g1
                            transferred += 1
                        elif key == "model.model.3.conv.weight":
                            # 由groups=4的权重构造groups=1的权重
                            old_weight = pose_state_dict[key]
                            c_out, c_in, k_h, k_w = old_weight.shape
                            G = 4  # 原始groups数量
                            weight_g4 = detection_state_dict[key]  # [64, 8, 3, 3]
                            weight_g1 = torch.zeros(c_out, c_in, k_h, k_w)  # [64, 32, 3, 3]
                            for g in range(G):
                                in_start = g * (c_in // G)          # 0, 8, 16, 24
                                in_end   = (g + 1) * (c_in // G)    # 8, 16, 24, 32

                                out_start = g * (c_out // G)        # 0, 16, 32, 48
                                out_end   = (g + 1) * (c_out // G)  # 16, 32, 48, 64

                                # 从 weight_g4 中取出第 g 组的权重（连续存储）
                                weight_g1[out_start:out_end, in_start:in_end, :, :] = weight_g4[out_start:out_end, :, :, :]

                            pose_state_dict[key] = weight_g1
                            transferred += 1
                        else:
                            raise ValueError("不应到达此处")
                    else:
                        skipped += 1
                        print(f"形状不匹配，跳过权重: {key}, 检测模型形状: {detection_state_dict[key].shape}, Pose模型形状: {value.shape}")
                            
                    
            else:
                print(f"Pose模型中不存在该权重，跳过: {key}")  # model.model.1.conv.weight model.model.3.conv.weight
                skipped += 1

        pose_model.load_state_dict(pose_state_dict)
        print(f"权重迁移完成: 成功迁移 {transferred} 个权重，跳过 {skipped} 个权重")
        return pose_model
        


    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if isinstance(weights, dict) and "fc_ckpt" in weights:
            model = self.fc_load_weights(weights, model)
        elif weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data




class RTMO_PoseTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (RTMO_PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "RTMO_Pose"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )


    def fc_load_weights(self, weights: dict[str, Any], pose_model: PoseModel) -> PoseModel:
        """Load fully connected layer weights from a checkpoint file.

        Args:
            weights (dict): Dictionary containing the weights, with "fc_ckpt" key pointing to the checkpoint file.
            model (PoseModel): The pose estimation model to load the weights into.

        Returns:
            (PoseModel): The model with loaded fully connected layer weights.
        """
        import torch
        detection_state_dict = torch.load(weights["fc_ckpt"], weights_only=False)["model"].state_dict()
        print("开始迁移权重...")
        pose_state_dict = pose_model.state_dict()
        
        
        transferred = 0
        skipped = 0
        
        for key, value in pose_state_dict.items():
            if key in detection_state_dict:
                # 确保形状匹配  
                if detection_state_dict[key].shape == value.shape:
                    pose_state_dict[key] = detection_state_dict[key]
                    transferred += 1
                else:
                    print(f"形状不匹配, {key}, 检测模型形状: {detection_state_dict[key].shape}, Pose模型形状: {value.shape}")
                    if key == "model.model.1.conv.weight" or key == "model.model.3.conv.weight":
                        print(f"注意: 该权重形状不匹配，是因为groups!=1导致的。")

                        if key == "model.model.1.conv.weight":
                            # 由groups=2的权重构造groups=1的权重
                            old_weight = pose_state_dict[key]
                            c_out, c_in, k_h, k_w = old_weight.shape
                            G = 2  # 原始groups数量
                            weight_g2 = detection_state_dict[key]  # [32, 8, 3, 3]
                            weight_g1 = torch.zeros(c_out, c_in, k_h, k_w)  # [32, 16, 3, 3]
                            for g in range(G):
                                in_start = g * (c_in // G)          # 0, 8
                                in_end   = (g + 1) * (c_in // G)    # 8, 16

                                out_start = g * (c_out // G)        # 0, 16
                                out_end   = (g + 1) * (c_out // G)  # 16, 32

                                # 从 weight_g2 中取出第 g 组的权重（连续存储）
                                # 注意：weight_g2 的输出通道是连续的 0~31，所以第 g 组就是 [out_start:out_end]
                                weight_g1[out_start:out_end, in_start:in_end, :, :] = weight_g2[out_start:out_end, :, :, :]

                            pose_state_dict[key] = weight_g1
                            transferred += 1
                        elif key == "model.model.3.conv.weight":
                            # 由groups=4的权重构造groups=1的权重
                            old_weight = pose_state_dict[key]
                            c_out, c_in, k_h, k_w = old_weight.shape
                            G = 4  # 原始groups数量
                            weight_g4 = detection_state_dict[key]  # [64, 8, 3, 3]
                            weight_g1 = torch.zeros(c_out, c_in, k_h, k_w)  # [64, 32, 3, 3]
                            for g in range(G):
                                in_start = g * (c_in // G)          # 0, 8, 16, 24
                                in_end   = (g + 1) * (c_in // G)    # 8, 16, 24, 32

                                out_start = g * (c_out // G)        # 0, 16, 32, 48
                                out_end   = (g + 1) * (c_out // G)  # 16, 32, 48, 64

                                # 从 weight_g4 中取出第 g 组的权重（连续存储）
                                weight_g1[out_start:out_end, in_start:in_end, :, :] = weight_g4[out_start:out_end, :, :, :]

                            pose_state_dict[key] = weight_g1
                            transferred += 1
                        else:
                            raise ValueError("不应到达此处")
                    else:
                        skipped += 1
                        print(f"形状不匹配，跳过权重: {key}, 检测模型形状: {detection_state_dict[key].shape}, Pose模型形状: {value.shape}")
                            
                    
            else:
                print(f"Pose模型中不存在该权重，跳过: {key}")  # model.model.1.conv.weight model.model.3.conv.weight
                skipped += 1

        pose_model.load_state_dict(pose_state_dict)
        print(f"权重迁移完成: 成功迁移 {transferred} 个权重，跳过 {skipped} 个权重")
        return pose_model
        


    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> RTMO_PoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = RTMO_PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if isinstance(weights, dict) and "fc_ckpt" in weights:
            model = self.fc_load_weights(weights, model)
        elif weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "mle_loss", "dcc_oks_loss"
        return yolo.pose.RTMO_PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data





    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _setup_train(self):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers, ".dfl"是原始的
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point and self.fc_mode is None:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
            elif v.requires_grad and "cv4" not in k and hasattr(self, 'fc_mode') and self.fc_mode == "pose_train":  # only floating point Tensor can require gradients  and self.fc_mode == "pose_train"  不训练pose模型时要删除
                LOGGER.warning(
                    f"train mode: train fc_pose, setting 'requires_grad=True' for trainable layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of trainable layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)

        model = self.model
        # 验证可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"\n参数统计:")
        print(f"- 冻结层名称列表: {freeze_layer_names}")
        print(f"- 可训练参数数量: {trainable_params}")
        print(f"- 总参数数量: {total_params}")



        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")


    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )





    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                # Backward
                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                # break   # 开始训练要删除

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")




