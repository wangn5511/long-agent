# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from enum import Enum
import os.path as osp
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.logging import MMLogger

from mmaction.registry import RUNNERS


class Gesture(Enum):
    """Gesture types for hand gesture recognition.
    
    Defines the standard gesture categories used in the gesture
    detection dataset.
    """
    NONE = 0  # 没有手势
    OK = 1  # OK 手势
    WAVE_IN = 2  # 内挥
    WAVE_OUT = 3  # 外挥
    WAVE_UP = 4  # 上挥
    WAVE_DOWN = 5  # 下挥
    WAVE_LEFT = 6  # 左挥
    WAVE_RIGHT = 7  # 右挥
    BYEBYE = 8  # "byebye" 挥
    THUMBS_UP = 9  # 点赞

    FIST_MOVE = 10  # 握拳 + 前后移动
    NUMBER = 11  # 数字手势 1–5
    SWIPE_FORWARD = 12  # 前滑动（桌面视角）
    SWIPE_BACKWARD = 13  # 后滑动（桌面视角）
    SWIPE_LEFT = 14  # 左滑动（桌面视角）
    SWIPE_RIGHT = 15  # 右滑动（桌面视角）


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # NOTE: Need to set work_dir to the ConfusionMatrix's save_path
    for evaluator in cfg.test_evaluator:
        if evaluator['type'] == 'ConfusionMatrix':
            evaluator.save_path = osp.join(cfg.work_dir, evaluator.save_path)
            evaluator.misclassified_save_path = osp.join(cfg.work_dir, evaluator.misclassified_save_path) if evaluator.misclassified_save_path is not None else None
            evaluator.classified_save_path = osp.join(cfg.work_dir, evaluator.classified_save_path) if evaluator.classified_save_path is not None else None

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()

    # NOTE: Use output of confusion matrix to get detailed evaluation results
    for evaluator in cfg.test_evaluator:
        if evaluator['type'] == 'ConfusionMatrix':
            summary_confusion_matrix(
                evaluator.save_path,
                evaluator.get('class_mapping'),
                evaluator.get('summary_save_path'))


def metrics_from_confusion(conf_mat):
    """
    输入:
        conf_mat: (K, K) 的混淆矩阵, M[i,j] = 真实类别 i 被预测为 j 的数量
    输出:
        per_class_acc: 每个类别的 accuracy (这里按类内正确率统计)
        per_class_recall: 每个类别的 recall
        per_class_precision: 每个类别的 precision
        per_class_f1: 每个类别的 f1
    """
    conf_mat = np.array(conf_mat)
    # 对角线：预测正确的数量
    diag = np.diag(conf_mat)
    
    # 每一行和：真实类别 i 的样本数
    real_per_class = conf_mat.sum(axis=1)
    
    # 每一列和：预测为 j 的样本数
    pred_per_class = conf_mat.sum(axis=0)
    
    # per-class acc（与 recall 一致，类内正确率）
    per_class_acc = diag / (real_per_class + 1e-12)
    
    # recall（等同于 per-class acc，这里单独列出）
    per_class_recall = diag / (real_per_class + 1e-12)
    
    # precision
    per_class_precision = diag / (pred_per_class + 1e-12)

    # f1
    per_class_f1 = 2 * per_class_precision * per_class_recall / (
        per_class_precision + per_class_recall + 1e-12)

    return per_class_acc, per_class_recall, per_class_precision, per_class_f1


def resolve_class_name(class_mapping, index: int) -> str:
    if class_mapping is not None:
        if index in class_mapping:
            return str(class_mapping[index])
        if str(index) in class_mapping:
            return str(class_mapping[str(index)])
    if index in Gesture._value2member_map_:
        return Gesture(index).name
    return f'CLASS_{index}'


def summary_confusion_matrix(confusion_matrix_path: str,
                             class_mapping: dict = None,
                             summary_save_path: str = None):
    """
    Summary the confusion matrix and save the per-class metrics.
    """
    confusion_matrix = np.load(confusion_matrix_path)
    acc, recall, precision, f1 = metrics_from_confusion(confusion_matrix)

    logger = MMLogger.get_current_instance()
    logger.info(f"=============>> Confusion matrix\n: {confusion_matrix}")
    lines = []
    for i in range(len(confusion_matrix)):
        class_name = resolve_class_name(class_mapping, i)
        lines.append(
            f"Class {class_name:>10}: acc={acc[i]:.3f}, recall={recall[i]:.3f}, "
            f"precision={precision[i]:.3f}, f1={f1[i]:.3f}")
    summary_text = '\n'.join(lines)
    logger.info("=============>> Per-class metrics summary:\n" + summary_text)
    logger.info(f"=============>> End of confusion matrix summary")

    if summary_save_path is None:
        summary_save_path = osp.splitext(confusion_matrix_path)[0] + '_summary.txt'
    with open(summary_save_path, 'w', encoding='utf-8') as f:
        f.write("Confusion matrix:\n")
        f.write(np.array2string(confusion_matrix))
        f.write("\n\nPer-class metrics:\n")
        f.write(summary_text)
        f.write('\n')


if __name__ == '__main__':
    main()
