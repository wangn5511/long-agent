# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union, Dict

from mmengine.fileio import exists, list_from_file
from mmengine.logging import MMLogger

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class RawframeDataset(BaseActionDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        filename_tmpl (str): Template for each filename.
            Defaults to ``img_{:05}.jpg``.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Defaults to False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Defaults to False.
        num_classes (int, optional): Number of classes in the dataset.
            Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 1, since raw frames count from 1.
            Defaults to 1.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(img=''),
                 filename_tmpl: str = 'img_{:05}.jpg',
                 with_offset: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 1,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            video_info = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]

            # XXX: sample name
            video_info['sample_name'] = frame_dir
            
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)
            video_info['frame_dir'] = frame_dir
            idx += 1
            if self.with_offset:
                # idx for offset and total_frames
                video_info['offset'] = int(line_split[idx])
                video_info['total_frames'] = int(line_split[idx + 1])
                idx += 2
            else:
                # idx for total_frames
                video_info['total_frames'] = int(line_split[idx])
                idx += 1
            # idx for label[s]
            label = [int(x) for x in line_split[idx:]]
            # add fake label for inference datalist without label
            if not label:
                label = [-1]
            if self.multi_class:
                assert self.num_classes is not None
                video_info['label'] = label
            else:
                assert len(label) == 1
                video_info['label'] = label[0]
            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)  # retrieve item from self.data_list and apply pipeline
        data_info['filename_tmpl'] = self.filename_tmpl
        return data_info


@DATASETS.register_module()
class RawframeDatasetWithLabelFix(RawframeDataset):
    """Rawframe dataset for action recognition with label fix."""
    def __init__(self, 
        class_mapping: Dict[int, str] = None,
        abandon_list: List[str] = None,
        label_fix_list: Dict[str, str] = None,
        **kwargs
    ):
        self.abandon_list = abandon_list
        self.label_fix_list = label_fix_list
        self.class_mapping = class_mapping
        self.label_name_to_id = {v: k for k, v in class_mapping.items()}

        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()

        logger = MMLogger.get_current_instance()

        # 1. 过滤abandon_list中的数据
        filter_indices = []
        for index, video_info in enumerate(data_list):
            if video_info["sample_name"] in self.abandon_list:
                logger.info(f"Abandon video: {video_info['sample_name']}")
                filter_indices.append(index)

        data_list = [x for i, x in enumerate(data_list) if i not in filter_indices]

        # 2. 修正label_fix_list中的数据
        for video_info in data_list:
            if video_info["sample_name"] in self.label_fix_list:
                logger.info(f"Fix label for video: {video_info['sample_name']} from {self.class_mapping[video_info['label']]} to {self.label_fix_list[video_info['sample_name']]}")
                video_info["label"] = self.label_name_to_id[self.label_fix_list[video_info["sample_name"]]]
        
        return data_list
