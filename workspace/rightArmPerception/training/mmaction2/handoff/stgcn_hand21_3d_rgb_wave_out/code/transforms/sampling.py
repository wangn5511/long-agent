from typing import Dict, Optional, Tuple

import numpy as np

from mmaction.datasets.transforms import UniformSampleFrames
from mmaction.registry import TRANSFORMS
from ..utils.window_track_policy import (
    evaluate_window_track_policy,
    summarize_window_track_stats,
    window_policy_rank_key,
)


def _apply_transition_aware_indices(results: Dict, inds: np.ndarray, clip_len: int, num_clips: int) -> Dict:
    num_frames = results["total_frames"]
    inds = np.mod(inds, num_frames)
    start_index = results.get("start_index", 0)

    if "keypoint" in results:
        kp = results["keypoint"]
        assert num_frames == kp.shape[1]
        num_person = kp.shape[0]
        num_persons = [num_person] * num_frames
        for i in range(num_frames):
            j = num_person - 1
            while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                j -= 1
            num_persons[i] = j + 1
        transitional = [False] * num_frames
        for i in range(1, num_frames - 1):
            if num_persons[i] != num_persons[i - 1]:
                transitional[i] = transitional[i - 1] = True
            if num_persons[i] != num_persons[i + 1]:
                transitional[i] = transitional[i + 1] = True
        inds_int = inds.astype(np.int64)
        coeff = np.array([transitional[i] for i in inds_int])
        inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

    results["frame_inds"] = (inds + start_index).astype(np.int32)
    results["clip_len"] = clip_len
    results["frame_interval"] = None
    results["num_clips"] = num_clips
    return results


@TRANSFORMS.register_module()
class MixedTSNSampleFrames(UniformSampleFrames):
    """Training-time sampler aligned with deployment-style 16-frame windows.

    During training it mixes three sampling modes on the source clip:

    - continuous 16-frame slice as the main path
    - local-window TSN over a short contiguous sub-interval
    - full-clip TSN as a low-probability long-span augmentation

    Validation and testing should use a separate deterministic sampler so that
    clip construction stays stable across runs.
    """

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = False,
        seed: int = 255,
        continuous_ratio: float = 0.6,
        local_ratio: float = 0.3,
        full_clip_ratio: float = 0.1,
        local_min_coverage: int = 20,
        local_max_coverage: int = 30,
        local_max_coverage_ratio: float = 0.8,
        continuous_start_min_ratio: float = 0.2,
    ) -> None:
        super().__init__(
            clip_len=clip_len,
            num_clips=num_clips,
            test_mode=test_mode,
            seed=seed,
        )
        total_ratio = continuous_ratio + local_ratio + full_clip_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError("continuous_ratio + local_ratio + full_clip_ratio must equal 1.")
        if continuous_ratio < 0.0 or local_ratio < 0.0 or full_clip_ratio < 0.0:
            raise ValueError("Sampling ratios must be non-negative.")
        if local_min_coverage < clip_len:
            raise ValueError("local_min_coverage must be >= clip_len.")
        if local_max_coverage < local_min_coverage:
            raise ValueError("local_max_coverage must be >= local_min_coverage.")
        if not 0.0 < local_max_coverage_ratio <= 1.0:
            raise ValueError("local_max_coverage_ratio must be in (0, 1].")
        if not 0.0 <= continuous_start_min_ratio <= 1.0:
            raise ValueError("continuous_start_min_ratio must be in [0, 1].")

        self.continuous_ratio = continuous_ratio
        self.local_ratio = local_ratio
        self.full_clip_ratio = full_clip_ratio
        self.local_min_coverage = local_min_coverage
        self.local_max_coverage = local_max_coverage
        self.local_max_coverage_ratio = local_max_coverage_ratio
        self.continuous_start_min_ratio = continuous_start_min_ratio

    def _sample_continuous_range(self, start: int, clip_len: int) -> np.ndarray:
        return np.arange(start, start + clip_len, dtype=np.int64)

    def _get_usable_sampling_start(self, num_frames: int, clip_len: int) -> int:
        if num_frames <= clip_len:
            return 0
        max_start = num_frames - clip_len
        return min(int(np.floor(self.continuous_start_min_ratio * max_start)), max_start)

    def _get_usable_sampling_span(self, num_frames: int, clip_len: int) -> Tuple[int, int]:
        start = self._get_usable_sampling_start(num_frames, clip_len)
        return start, num_frames - start

    def _get_full_clip_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        usable_start, usable_len = self._get_usable_sampling_span(num_frames, clip_len)
        if usable_start <= 0:
            return self._get_train_clips(num_frames, clip_len)
        inds = self._get_train_clips(usable_len, clip_len)
        return inds + usable_start

    def _sample_tsn_range(self, start: int, span_len: int, clip_len: int) -> np.ndarray:
        if span_len <= 0:
            raise ValueError("span_len must be positive.")
        if span_len < clip_len:
            offset = np.random.randint(0, span_len)
            return start + np.mod(np.arange(offset, offset + clip_len), span_len)

        segment_boundaries = np.array(
            [i * span_len // clip_len for i in range(clip_len + 1)],
            dtype=np.int64,
        )
        inds = []
        for left, right in zip(segment_boundaries[:-1], segment_boundaries[1:]):
            if right > left:
                sampled = np.random.randint(left, right)
            else:
                sampled = left
            inds.append(start + sampled)
        return np.array(inds, dtype=np.int64)

    def _get_continuous_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        if num_frames <= clip_len:
            return self._get_train_clips(num_frames, clip_len)

        all_inds = []
        max_start = num_frames - clip_len
        min_start = self._get_usable_sampling_start(num_frames, clip_len)
        for _ in range(self.num_clips):
            start = np.random.randint(min_start, max_start + 1)
            inds = self._sample_continuous_range(start=start, clip_len=clip_len)
            all_inds.append(inds)

        return np.concatenate(all_inds)

    def _get_local_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        usable_start, usable_len = self._get_usable_sampling_span(num_frames, clip_len)
        max_span = min(
            self.local_max_coverage,
            int(np.floor(usable_len * self.local_max_coverage_ratio)),
        )
        min_span = self.local_min_coverage

        if usable_len < min_span or max_span < min_span:
            return self._get_continuous_train_clips(num_frames, clip_len)

        all_inds = []
        for _ in range(self.num_clips):
            span_len = np.random.randint(min_span, max_span + 1)
            max_start = num_frames - span_len
            start = usable_start if max_start <= usable_start else np.random.randint(usable_start, max_start + 1)
            inds = self._sample_tsn_range(start=start, span_len=span_len, clip_len=clip_len)
            all_inds.append(inds)

        return np.concatenate(all_inds)

    def transform(self, results: Dict) -> Dict:
        num_frames = results["total_frames"]
        if self.test_mode:
            inds = self._get_train_clips(num_frames, self.clip_len)
        elif num_frames < self.clip_len:
            inds = self._get_train_clips(num_frames, self.clip_len)
        elif num_frames < self.local_min_coverage:
            inds = self._get_continuous_train_clips(num_frames, self.clip_len)
        else:
            r = np.random.rand()
            if r < self.continuous_ratio:
                inds = self._get_continuous_train_clips(num_frames, self.clip_len)
            elif r < self.continuous_ratio + self.local_ratio:
                inds = self._get_local_train_clips(num_frames, self.clip_len)
            else:
                inds = self._get_full_clip_train_clips(num_frames, self.clip_len)

        return _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"test_mode={self.test_mode}, "
            f"seed={self.seed}, "
            f"continuous_ratio={self.continuous_ratio}, "
            f"local_ratio={self.local_ratio}, "
            f"full_clip_ratio={self.full_clip_ratio}, "
            f"local_min_coverage={self.local_min_coverage}, "
            f"local_max_coverage={self.local_max_coverage}, "
            f"local_max_coverage_ratio={self.local_max_coverage_ratio}, "
            f"continuous_start_min_ratio={self.continuous_start_min_ratio})"
        )


@TRANSFORMS.register_module()
class ClickAwareMixedTSNSampleFrames(MixedTSNSampleFrames):
    """Mixed sampler with class-specific continuous-only override.

    Samples whose label is listed in ``force_continuous_labels`` always use the
    continuous 16-frame branch. All other labels keep the default mixed
    strategy from ``MixedTSNSampleFrames``.
    """

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = False,
        seed: int = 255,
        continuous_ratio: float = 0.6,
        local_ratio: float = 0.3,
        full_clip_ratio: float = 0.1,
        local_min_coverage: int = 20,
        local_max_coverage: int = 30,
        local_max_coverage_ratio: float = 0.8,
        continuous_start_min_ratio: float = 0.2,
        force_continuous_labels: tuple = (2,),
    ) -> None:
        super().__init__(
            clip_len=clip_len,
            num_clips=num_clips,
            test_mode=test_mode,
            seed=seed,
            continuous_ratio=continuous_ratio,
            local_ratio=local_ratio,
            full_clip_ratio=full_clip_ratio,
            local_min_coverage=local_min_coverage,
            local_max_coverage=local_max_coverage,
            local_max_coverage_ratio=local_max_coverage_ratio,
            continuous_start_min_ratio=continuous_start_min_ratio,
        )
        self.force_continuous_labels = tuple(int(label) for label in force_continuous_labels)

    def transform(self, results: Dict) -> Dict:
        label = results.get("label", None)
        if not self.test_mode and label is not None and int(label) in self.force_continuous_labels:
            num_frames = results["total_frames"]
            if num_frames < self.clip_len:
                inds = self._get_train_clips(num_frames, self.clip_len)
            else:
                inds = self._get_continuous_train_clips(num_frames, self.clip_len)
            return _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)
        return super().transform(results)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"test_mode={self.test_mode}, "
            f"seed={self.seed}, "
            f"continuous_ratio={self.continuous_ratio}, "
            f"local_ratio={self.local_ratio}, "
            f"full_clip_ratio={self.full_clip_ratio}, "
            f"local_min_coverage={self.local_min_coverage}, "
            f"local_max_coverage={self.local_max_coverage}, "
            f"local_max_coverage_ratio={self.local_max_coverage_ratio}, "
            f"continuous_start_min_ratio={self.continuous_start_min_ratio}, "
            f"force_continuous_labels={self.force_continuous_labels})"
        )


@TRANSFORMS.register_module()
class CoreAwareMixedSampleFrames(MixedTSNSampleFrames):
    """Mixed sampler with class-aware core-anchored sampling.

    Labels in ``core_classes`` can use core-aware continuous or local TSN
    sampling when ``core_start/core_end/core_peak`` are present in the sample
    metadata. Labels in ``force_continuous_labels`` always use the core-aware
    continuous branch. Samples without core metadata automatically fall back to
    the base mixed sampling logic.
    """

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = False,
        seed: int = 255,
        continuous_ratio: float = 0.7,
        local_ratio: float = 0.2,
        full_clip_ratio: float = 0.1,
        local_min_coverage: int = 20,
        local_max_coverage: int = 30,
        local_max_coverage_ratio: float = 0.8,
        continuous_start_min_ratio: float = 0.2,
        core_classes: tuple = (0, 2, 3),
        force_continuous_labels: tuple = (2,),
        core_local_tsn_labels: tuple = (0, 3),
        core_local_margin: int = 5,
        core_jitter: int = 3,
    ) -> None:
        super().__init__(
            clip_len=clip_len,
            num_clips=num_clips,
            test_mode=test_mode,
            seed=seed,
            continuous_ratio=continuous_ratio,
            local_ratio=local_ratio,
            full_clip_ratio=full_clip_ratio,
            local_min_coverage=local_min_coverage,
            local_max_coverage=local_max_coverage,
            local_max_coverage_ratio=local_max_coverage_ratio,
            continuous_start_min_ratio=continuous_start_min_ratio,
        )
        self.core_classes = tuple(int(label) for label in core_classes)
        self.force_continuous_labels = tuple(int(label) for label in force_continuous_labels)
        self.core_local_tsn_labels = tuple(int(label) for label in core_local_tsn_labels)
        self.core_local_margin = int(core_local_margin)
        self.core_jitter = int(core_jitter)

    def _get_core_meta(self, results: Dict) -> Tuple[int, int, int]:
        return (
            int(results.get("core_start", -1)),
            int(results.get("core_end", -1)),
            int(results.get("core_peak", -1)),
        )

    def _sample_core_anchored(self, results: Dict) -> np.ndarray:
        total_frames = results["total_frames"]
        core_start, core_end, core_peak = self._get_core_meta(results)
        if core_peak < 0 or total_frames <= self.clip_len:
            return self._get_train_clips(total_frames, self.clip_len)

        if core_start < 0 or core_end < 0:
            ideal_start = core_peak - self.clip_len // 2
            jitter = 0 if self.core_jitter <= 0 else np.random.randint(-self.core_jitter, self.core_jitter + 1)
            win_start = ideal_start + jitter
            win_start = max(win_start, core_peak - self.clip_len + 1)
            win_start = min(win_start, core_peak)
            win_start = int(np.clip(win_start, 0, total_frames - self.clip_len))
            return self._sample_continuous_range(start=win_start, clip_len=self.clip_len)

        core_len = core_end - core_start + 1
        if core_len < self.clip_len:
            ideal_start = core_peak - self.clip_len // 2
            jitter = 0 if self.core_jitter <= 0 else np.random.randint(-self.core_jitter, self.core_jitter + 1)
            win_start = ideal_start + jitter
            win_start = max(win_start, core_peak - self.clip_len + 1)
            win_start = min(win_start, core_peak)
        else:
            core_center = (core_start + core_end) // 2
            ideal_start = core_center - self.clip_len // 2
            jitter = 0 if self.core_jitter <= 0 else np.random.randint(-self.core_jitter, self.core_jitter + 1)
            win_start = ideal_start + jitter

        win_start = int(np.clip(win_start, 0, total_frames - self.clip_len))
        return self._sample_continuous_range(start=win_start, clip_len=self.clip_len)

    def _get_core_local_train_clips(self, results: Dict) -> np.ndarray:
        total_frames = results["total_frames"]
        core_start, core_end, core_peak = self._get_core_meta(results)
        if core_peak < 0 or core_start < 0 or core_end < 0:
            return self._get_local_train_clips(total_frames, self.clip_len)
        span_start = max(0, core_start - self.core_local_margin)
        span_end = min(total_frames - 1, core_end + self.core_local_margin)
        span_len = span_end - span_start + 1
        if span_len < self.clip_len:
            return self._sample_core_anchored(results)
        all_inds = []
        for _ in range(self.num_clips):
            inds = self._sample_tsn_range(start=span_start, span_len=span_len, clip_len=self.clip_len)
            all_inds.append(inds)
        return np.concatenate(all_inds)

    def transform(self, results: Dict) -> Dict:
        label = results.get("label", None)
        num_frames = results["total_frames"]
        if self.test_mode or label is None or int(label) not in self.core_classes:
            return super().transform(results)

        core_start, core_end, core_peak = self._get_core_meta(results)
        if core_peak < 0:
            return super().transform(results)

        label = int(label)
        if num_frames < self.clip_len:
            inds = self._get_train_clips(num_frames, self.clip_len)
        elif label in self.force_continuous_labels:
            inds = self._sample_core_anchored(results)
        else:
            r = np.random.rand()
            if r < self.continuous_ratio:
                inds = self._sample_core_anchored(results)
            elif r < self.continuous_ratio + self.local_ratio and label in self.core_local_tsn_labels:
                inds = self._get_core_local_train_clips(results)
            elif r < self.continuous_ratio + self.local_ratio:
                inds = self._sample_core_anchored(results)
            else:
                inds = self._get_train_clips(num_frames, self.clip_len)

        return _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"test_mode={self.test_mode}, "
            f"seed={self.seed}, "
            f"continuous_ratio={self.continuous_ratio}, "
            f"local_ratio={self.local_ratio}, "
            f"full_clip_ratio={self.full_clip_ratio}, "
            f"core_classes={self.core_classes}, "
            f"force_continuous_labels={self.force_continuous_labels}, "
            f"core_local_tsn_labels={self.core_local_tsn_labels}, "
            f"core_local_margin={self.core_local_margin}, "
            f"core_jitter={self.core_jitter})"
        )


@TRANSFORMS.register_module()
class CoreAwareSampleFrames(MixedTSNSampleFrames):
    """Online onset-anchored continuous sampler for CLICK; mixed strategy otherwise.

    For labels in ``core_classes`` (default: CLICK=2):
    - Compute wrist-relative motion energy online from the full keypoint sequence:

          kp_rel[t] = kp[t] - kp[t, wrist]          # remove wrist translation
          disp[t, k] = ||kp_rel[t, k] - kp_rel[t-1, k]||_2
          E[t] = max_k disp[t, k]
          E_smooth = moving_average(E, window=peak_smooth_window)

      Wrist-centering isolates finger/hand-shape motion from global wrist
      translation, making the approach phase (hand moving toward target) appear
      low-energy while the tap and retraction stand out clearly.

    - Find the motion onset: first frame where E_smooth >= onset_threshold * max(E_smooth).
      Falls back to continuous sampling when the clip energy is below abs_energy_floor.

    - Place the window so onset sits at position ``pre_context`` from the left:

          win_start = onset - pre_context + jitter

      With ``pre_context=4`` the window shows 4 frames of approach before the
      onset, then 12 frames covering the full tap + retraction. The retraction
      (hand shape returning to extended cursor pose) is the most discriminative
      feature against CURSOR, so it receives the majority of the window.

    - Falls back to ordinary continuous sampling when keypoints are absent or
      the clip energy is too low.

    All other labels use the base MixedTSNSampleFrames strategy unchanged.
    """

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = False,
        seed: int = 255,
        continuous_ratio: float = 0.6,
        local_ratio: float = 0.3,
        full_clip_ratio: float = 0.1,
        local_min_coverage: int = 20,
        local_max_coverage: int = 30,
        local_max_coverage_ratio: float = 0.8,
        continuous_start_min_ratio: float = 0.2,
        core_classes: tuple = (2,),
        peak_smooth_window: int = 3,
        onset_threshold: float = 0.5,
        abs_energy_floor: float = 0.003,
        pre_context: int = 4,
        jitter_range: int = 2,
    ) -> None:
        super().__init__(
            clip_len=clip_len,
            num_clips=num_clips,
            test_mode=test_mode,
            seed=seed,
            continuous_ratio=continuous_ratio,
            local_ratio=local_ratio,
            full_clip_ratio=full_clip_ratio,
            local_min_coverage=local_min_coverage,
            local_max_coverage=local_max_coverage,
            local_max_coverage_ratio=local_max_coverage_ratio,
            continuous_start_min_ratio=continuous_start_min_ratio,
        )
        self.core_classes = tuple(int(c) for c in core_classes)
        self.peak_smooth_window = int(peak_smooth_window)
        self.onset_threshold = float(onset_threshold)
        self.abs_energy_floor = float(abs_energy_floor)
        self.pre_context = int(pre_context)
        self.jitter_range = int(jitter_range)

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or values.size == 0:
            return values.astype(np.float32, copy=True)
        pad = window // 2
        padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    def _find_onset_window_start(self, results: Dict, *, jitter: Optional[int] = None) -> Optional[int]:
        total_frames = results["total_frames"]
        keypoint = results.get("keypoint")
        if keypoint is None:
            return None

        kp = np.asarray(keypoint, dtype=np.float32)
        if kp.ndim != 4 or kp.shape[0] < 1 or kp.shape[1] != total_frames:
            return None

        kp = kp[0]  # (T, 21, 3)
        kp_rel = kp - kp[:, 0:1, :]  # wrist-relative: removes global wrist translation
        disp = np.linalg.norm(kp_rel[1:] - kp_rel[:-1], axis=-1)
        energy = np.concatenate([[0.0], disp.max(axis=-1).astype(np.float32)], axis=0)
        energy_smooth = self._moving_average(energy, self.peak_smooth_window)

        max_e = float(energy_smooth.max())
        if max_e < self.abs_energy_floor:
            return None

        threshold_abs = self.onset_threshold * max_e
        onset = next((i for i, e in enumerate(energy_smooth) if e >= threshold_abs), -1)
        if onset < 0:
            return None

        if jitter is None:
            jitter = np.random.randint(-self.jitter_range, self.jitter_range + 1) if self.jitter_range > 0 else 0
        win_start = onset - self.pre_context + jitter
        win_start = int(np.clip(win_start, 0, total_frames - self.clip_len))
        return win_start

    def _sample_onset_anchored(self, results: Dict) -> np.ndarray:
        total_frames = results["total_frames"]
        win_start = self._find_onset_window_start(results)
        if win_start is None:
            return self._get_continuous_train_clips(total_frames, self.clip_len)
        return np.arange(win_start, win_start + self.clip_len, dtype=np.int64)

    def transform(self, results: Dict) -> Dict:
        label = results.get("label")
        if not self.test_mode and label is not None and int(label) in self.core_classes:
            num_frames = results["total_frames"]
            if num_frames < self.clip_len:
                inds = self._get_train_clips(num_frames, self.clip_len)
            else:
                inds = self._sample_onset_anchored(results)
            return _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)
        return super().transform(results)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"test_mode={self.test_mode}, "
            f"seed={self.seed}, "
            f"continuous_ratio={self.continuous_ratio}, "
            f"local_ratio={self.local_ratio}, "
            f"full_clip_ratio={self.full_clip_ratio}, "
            f"local_min_coverage={self.local_min_coverage}, "
            f"local_max_coverage={self.local_max_coverage}, "
            f"local_max_coverage_ratio={self.local_max_coverage_ratio}, "
            f"continuous_start_min_ratio={self.continuous_start_min_ratio}, "
            f"core_classes={self.core_classes}, "
            f"peak_smooth_window={self.peak_smooth_window}, "
            f"onset_threshold={self.onset_threshold}, "
            f"abs_energy_floor={self.abs_energy_floor}, "
            f"pre_context={self.pre_context}, "
            f"jitter_range={self.jitter_range})"
        )


@TRANSFORMS.register_module()
class ClickDeterministicOnsetSampleFrames(CoreAwareSampleFrames):
    """Deterministic validation/test sampler with CLICK-only onset anchoring."""

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = True,
        seed: int = 255,
        crop_mode: str = "center",
        core_classes: tuple = (2,),
        peak_smooth_window: int = 3,
        onset_threshold: float = 0.5,
        abs_energy_floor: float = 0.003,
        pre_context: int = 0,
    ) -> None:
        super().__init__(
            clip_len=clip_len,
            num_clips=num_clips,
            test_mode=test_mode,
            seed=seed,
            continuous_ratio=1.0,
            local_ratio=0.0,
            full_clip_ratio=0.0,
            core_classes=core_classes,
            peak_smooth_window=peak_smooth_window,
            onset_threshold=onset_threshold,
            abs_energy_floor=abs_energy_floor,
            pre_context=pre_context,
            jitter_range=0,
        )
        if crop_mode not in {"center", "left"}:
            raise ValueError("crop_mode must be one of {'center', 'left'}.")
        self.crop_mode = crop_mode

    def _get_deterministic_start(self, num_frames: int) -> int:
        if num_frames <= self.clip_len:
            return 0
        if self.crop_mode == "left":
            return 0
        return (num_frames - self.clip_len) // 2

    def transform(self, results: Dict) -> Dict:
        num_frames = results["total_frames"]
        label = results.get("label")
        if label is not None and int(label) in self.core_classes and num_frames > self.clip_len:
            win_start = self._find_onset_window_start(results, jitter=0)
            if win_start is not None:
                inds = np.arange(win_start, win_start + self.clip_len, dtype=np.int64)
                return _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)

        start = self._get_deterministic_start(num_frames)
        inds = np.arange(start, start + self.clip_len, dtype=np.int64)
        all_inds = [inds.copy() for _ in range(self.num_clips)]
        return _apply_transition_aware_indices(results, np.concatenate(all_inds), self.clip_len, self.num_clips)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"test_mode={self.test_mode}, "
            f"seed={self.seed}, "
            f"crop_mode={self.crop_mode!r}, "
            f"core_classes={self.core_classes}, "
            f"peak_smooth_window={self.peak_smooth_window}, "
            f"onset_threshold={self.onset_threshold}, "
            f"abs_energy_floor={self.abs_energy_floor}, "
            f"pre_context={self.pre_context})"
        )


@TRANSFORMS.register_module()
class DeterministicContinuousSampleFrames(UniformSampleFrames):
    """Deterministic continuous clip sampler for validation and clip-level test.

    When the source clip is longer than ``clip_len``, this transform takes a
    fixed contiguous window. By default the crop is centred so validation stays
    deterministic and closer to deployment-style fixed-length buffering.
    """

    def __init__(
        self,
        clip_len: int,
        num_clips: int = 1,
        test_mode: bool = True,
        seed: int = 255,
        crop_mode: str = "center",
    ) -> None:
        super().__init__(
            clip_len=clip_len,
            num_clips=num_clips,
            test_mode=test_mode,
            seed=seed,
        )
        if crop_mode not in {"center", "left"}:
            raise ValueError("crop_mode must be one of {'center', 'left'}.")
        self.crop_mode = crop_mode

    def _get_deterministic_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        if num_frames <= clip_len:
            start = 0
        elif self.crop_mode == "left":
            start = 0
        else:
            start = (num_frames - clip_len) // 2

        inds = self._sample_continuous_clip(start, clip_len)
        all_inds = [inds.copy() for _ in range(self.num_clips)]
        return np.concatenate(all_inds)

    def _sample_continuous_clip(self, start: int, clip_len: int) -> np.ndarray:
        return np.arange(start, start + clip_len, dtype=np.int64)

    def transform(self, results: Dict) -> Dict:
        inds = self._get_deterministic_clips(results["total_frames"], self.clip_len)
        return _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"test_mode={self.test_mode}, "
            f"seed={self.seed}, "
            f"crop_mode={self.crop_mode!r})"
        )


class _TrackAwareWindowMixin:
    def __init__(
        self,
        *,
        track_valid_score_threshold: float = 0.0,
        primary_valid_ratio_thr: float = 0.75,
        primary_min_longest_run: int = 8,
        secondary_max_ratio_thr: float = 0.25,
        secondary_total_ratio_thr: float = 0.375,
        primary_margin_frames_thr: int = 2,
    ) -> None:
        self.track_valid_score_threshold = float(track_valid_score_threshold)
        self.primary_valid_ratio_thr = float(primary_valid_ratio_thr)
        self.primary_min_longest_run = int(primary_min_longest_run)
        self.secondary_max_ratio_thr = float(secondary_max_ratio_thr)
        self.secondary_total_ratio_thr = float(secondary_total_ratio_thr)
        self.primary_margin_frames_thr = int(primary_margin_frames_thr)

    def _evaluate_track_window(self, results: Dict, frame_inds: np.ndarray) -> Dict[str, object]:
        track_ids = results.get("track_ids")
        track_frame_score = results.get("track_frame_score")
        if track_ids is None or track_frame_score is None:
            return {
                "selected_track_id": None,
                "selected_track_index": None,
                "selected_track_stats": None,
                "secondary_max_ratio": 0.0,
                "secondary_total_ratio": 0.0,
                "secondary_best_track_id": None,
                "secondary_best_valid_count": 0,
                "primary_margin_frames": -1,
                "passed": True,
                "reasons": [],
                "track_candidates": [],
            }

        stats = summarize_window_track_stats(
            track_ids=track_ids,
            track_frame_score=np.asarray(track_frame_score, dtype=np.float32),
            frame_inds=np.asarray(frame_inds, dtype=np.int64),
            valid_score_threshold=self.track_valid_score_threshold,
        )
        return evaluate_window_track_policy(
            track_stats=stats,
            window_len=int(max(1, np.asarray(frame_inds).size)),
            primary_valid_ratio_thr=self.primary_valid_ratio_thr,
            primary_min_longest_run=self.primary_min_longest_run,
            secondary_max_ratio_thr=self.secondary_max_ratio_thr,
            secondary_total_ratio_thr=self.secondary_total_ratio_thr,
            primary_margin_frames_thr=self.primary_margin_frames_thr,
        )

    def _attach_track_window_meta(
        self,
        results: Dict,
        summary: Dict[str, object],
        *,
        fallback_used: bool,
    ) -> Dict:
        results["selected_window_track_id"] = summary.get("selected_track_id")
        results["selected_window_track_stats"] = summary
        results["window_track_quality_passed"] = bool(summary.get("passed", False))
        results["window_track_rejection_reasons"] = list(summary.get("reasons", []))
        results["window_track_fallback_used"] = bool(fallback_used)
        return results


@TRANSFORMS.register_module()
class TrackAwareCoreAwareSampleFrames(_TrackAwareWindowMixin, CoreAwareSampleFrames):
    """Core-aware sampler with window-level local primary-track selection."""

    def __init__(
        self,
        *args,
        max_resample_attempts: int = 8,
        fallback_policy: str = "best_quality",
        **kwargs,
    ) -> None:
        track_kwargs = {
            "track_valid_score_threshold": kwargs.pop("track_valid_score_threshold", 0.0),
            "primary_valid_ratio_thr": kwargs.pop("primary_valid_ratio_thr", 0.75),
            "primary_min_longest_run": kwargs.pop("primary_min_longest_run", 8),
            "secondary_max_ratio_thr": kwargs.pop("secondary_max_ratio_thr", 0.25),
            "secondary_total_ratio_thr": kwargs.pop("secondary_total_ratio_thr", 0.375),
            "primary_margin_frames_thr": kwargs.pop("primary_margin_frames_thr", 2),
        }
        CoreAwareSampleFrames.__init__(self, *args, **kwargs)
        _TrackAwareWindowMixin.__init__(self, **track_kwargs)
        self.max_resample_attempts = int(max_resample_attempts)
        if fallback_policy not in {"best_quality", "error"}:
            raise ValueError("fallback_policy must be one of {'best_quality', 'error'}.")
        self.fallback_policy = fallback_policy

    def _sample_candidate_frame_inds(self, results: Dict) -> np.ndarray:
        probe = dict(results)
        sampled = CoreAwareSampleFrames.transform(self, probe)
        start_index = int(results.get("start_index", 0))
        inds = np.asarray(sampled["frame_inds"], dtype=np.int64).reshape(-1) - start_index
        return inds

    def transform(self, results: Dict) -> Dict:
        best_inds: Optional[np.ndarray] = None
        best_summary: Optional[Dict[str, object]] = None
        best_rank: Optional[tuple] = None

        for _ in range(max(1, self.max_resample_attempts)):
            inds = self._sample_candidate_frame_inds(results)
            summary = self._evaluate_track_window(results, inds)
            rank = window_policy_rank_key(summary)
            if best_rank is None or rank > best_rank:
                best_inds = inds
                best_summary = summary
                best_rank = rank
            if bool(summary.get("passed", False)):
                out = _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)
                return self._attach_track_window_meta(out, summary, fallback_used=False)

        if self.fallback_policy == "error" or best_inds is None or best_summary is None:
            raise RuntimeError("Track-aware sampling failed to find a valid window.")

        out = _apply_transition_aware_indices(results, best_inds, self.clip_len, self.num_clips)
        return self._attach_track_window_meta(out, best_summary, fallback_used=True)


@TRANSFORMS.register_module()
class TrackAwareClickDeterministicOnsetSampleFrames(_TrackAwareWindowMixin, ClickDeterministicOnsetSampleFrames):
    """Deterministic sampler that also selects a local primary track per window."""

    def __init__(
        self,
        *args,
        enforce_quality_gate: bool = False,
        **kwargs,
    ) -> None:
        track_kwargs = {
            "track_valid_score_threshold": kwargs.pop("track_valid_score_threshold", 0.0),
            "primary_valid_ratio_thr": kwargs.pop("primary_valid_ratio_thr", 0.75),
            "primary_min_longest_run": kwargs.pop("primary_min_longest_run", 8),
            "secondary_max_ratio_thr": kwargs.pop("secondary_max_ratio_thr", 0.25),
            "secondary_total_ratio_thr": kwargs.pop("secondary_total_ratio_thr", 0.375),
            "primary_margin_frames_thr": kwargs.pop("primary_margin_frames_thr", 2),
        }
        ClickDeterministicOnsetSampleFrames.__init__(self, *args, **kwargs)
        _TrackAwareWindowMixin.__init__(self, **track_kwargs)
        self.enforce_quality_gate = bool(enforce_quality_gate)

    def _sample_candidate_frame_inds(self, results: Dict) -> np.ndarray:
        probe = dict(results)
        sampled = ClickDeterministicOnsetSampleFrames.transform(self, probe)
        start_index = int(results.get("start_index", 0))
        inds = np.asarray(sampled["frame_inds"], dtype=np.int64).reshape(-1) - start_index
        return inds

    def transform(self, results: Dict) -> Dict:
        inds = self._sample_candidate_frame_inds(results)
        summary = self._evaluate_track_window(results, inds)
        if self.enforce_quality_gate and not bool(summary.get("passed", False)):
            summary = dict(summary)
        out = _apply_transition_aware_indices(results, inds, self.clip_len, self.num_clips)
        return self._attach_track_window_meta(out, summary, fallback_used=not bool(summary.get("passed", False)))
