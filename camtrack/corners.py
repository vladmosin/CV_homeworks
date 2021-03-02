#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from scipy.spatial import distance_matrix
from copy import deepcopy


from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    BLOCK_SIZE = 6
    MIN_DIST = 10
    QUALITY = 0.01
    MAX_CORNERS = 1000

    img0 = frame_sequence[0]
    p0 = cv2.goodFeaturesToTrack(img0, MAX_CORNERS, QUALITY, MIN_DIST, blockSize=BLOCK_SIZE).squeeze(1)

    last_id = len(p0)
    ids = np.arange(last_id)
    frame_corners = FrameCorners(ids, p0, np.ones(len(ids)) * BLOCK_SIZE)

    builder.set_corners_at_frame(0, frame_corners)
    for frame, img1 in enumerate(frame_sequence[1:], 1):
        p0, st, _ = cv2.calcOpticalFlowPyrLK(np.uint8(img0 * 255), np.uint8(img1 * 255), p0,
                                             None, winSize=(12, 12), maxLevel=3, minEigThreshold=0.001)
        st = st.reshape(-1)

        p0 = p0[st == 1]
        ids = ids[st == 1]

        p1 = cv2.goodFeaturesToTrack(img1, MAX_CORNERS, QUALITY, MIN_DIST, blockSize=BLOCK_SIZE).squeeze(1)

        distances = distance_matrix(p1, p0).min(axis=1)
        p1 = p1[distances >= MIN_DIST, :]

        p0 = np.concatenate([p0, p1])
        ids = np.concatenate([ids, np.arange(last_id, last_id + len(p1))])
        last_id += len(p1)

        frame_corners = FrameCorners(ids, p0, np.ones(len(ids)) * BLOCK_SIZE)
        builder.set_corners_at_frame(frame, frame_corners)
        img0 = img1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
