#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    calc_inlier_indices,
    eye3x4,
    Correspondences
)

max_reprojection_error = 5
min_triangulation_angle_deg = 2
min_depth = 0.1
min_intersection = 8
max_error = 20
threshold = 0.1


def get_best_view(points_1, points_2, ids, intrinsic_mat, R1, R2, t):
    view_mat_1 = eye3x4()
    triangulation_params = TriangulationParameters(
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        min_depth=min_depth
    )

    correspondence = Correspondences(ids, points_1, points_2)

    _, corr_ids1, _ = triangulate_correspondences(
        correspondence, view_mat_1, np.hstack((R1, t)), intrinsic_mat, triangulation_params
    )

    _, corr_ids2, _ = triangulate_correspondences(
        correspondence, view_mat_1, np.hstack((R2, t)), intrinsic_mat, triangulation_params
    )

    if len(corr_ids1) > len(corr_ids2):
        return R1, len(corr_ids1)
    else:
        return R2, len(corr_ids2)


def collect_views(i, j, R, t):
    return (i, view_mat3x4_to_pose(eye3x4())), (j, view_mat3x4_to_pose(np.hstack((R, t))))


def find_known_views(corner_storage: CornerStorage, intrinsic_mat: np.ndarray):
    n = len(corner_storage)

    scores = []

    for i in range(n):
        for j in range(i + 5, n):
            intersection, indices = snp.intersect(corner_storage[i].ids.reshape(-1), corner_storage[j].ids.reshape(-1), indices=True)

            if len(intersection) < 10:
                break

            points1 = corner_storage[i].points[indices[0]]
            points2 = corner_storage[j].points[indices[1]]
            essential_mat, inliers = cv2.findEssentialMat(points1, points2, intrinsic_mat, method=cv2.RANSAC, threshold=threshold)

            mask = inliers.reshape(-1) == 1

            R1, R2, t = cv2.decomposeEssentialMat(essential_mat)
            R, val = get_best_view(points1[mask], points2[mask], intersection[mask], intrinsic_mat, R1, R2, t)

            scores.append([i, j, R, t, val])

            if val > 60:
                return collect_views(i, j, R, t)

    i, j, R, t, _ = max(scores, key=lambda x: x[-1])

    return collect_views(i, j, R, t)


def first_not_none(view_mats, i):
    n = len(view_mats)
    for j in range(n):
        if i + j < n and view_mats[i + j] is not None:
            return view_mats[i + j]
        if i - j >= 0 and view_mats[i - j] is not None:
            return view_mats[i - j]


def remove_none(view_mats):
    n = len(view_mats)
    for i in range(n):
        if view_mats[i] is not None:
            continue
        else:
            view_mats[i] = first_not_none(view_mats, i)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_known_views(corner_storage, intrinsic_mat)

    triangulation_parameters = TriangulationParameters(
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        min_depth=min_depth
    )

    fid = known_view_1[0]
    sid = known_view_2[0]

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count

    view_mats[fid] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[sid] = pose_to_view_mat3x4(known_view_2[1])

    correspondence = build_correspondences(corner_storage[fid], corner_storage[sid])
    points3d, corr_ids, _ = triangulate_correspondences(
        correspondence, view_mats[fid], view_mats[sid], intrinsic_mat, triangulation_parameters
    )

    triangulation_parameters = TriangulationParameters(
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=min_triangulation_angle_deg,
        min_depth=min_depth
    )

    point_cloud_builder = PointCloudBuilder(corr_ids, points3d)

    updated = True
    while updated:
        updated = False

        for i in range(frame_count):
            if view_mats[i] is None:
                intersection, indices = snp.intersect(point_cloud_builder.ids.squeeze(1),
                                                      corner_storage[i].ids.squeeze(1), indices=True)
                points_cloud = point_cloud_builder.points[indices[0]]
                points_corners = corner_storage[i].points[indices[1]]

                if len(intersection) < min_intersection:
                    continue

                camera_found, R, t, inliers = cv2.solvePnPRansac(
                    points_cloud, points_corners.reshape(-1, 1, 2), intrinsic_mat,
                    np.array([]), flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=300,
                    reprojectionError=max_reprojection_error, confidence=0.995
                )

                if not camera_found:
                    continue

                view_mats[i] = rodrigues_and_translation_to_view_mat3x4(R, t)
                updated = True

                outliers_ids = np.delete(intersection, inliers)
                for j in range(frame_count):
                    if view_mats[j] is None or i == j:
                        continue

                    correspondence = build_correspondences(
                        corner_storage[i], corner_storage[j],
                        np.concatenate([point_cloud_builder.ids, outliers_ids[:, None]])
                    )

                    points3d, corr_ids, _ = triangulate_correspondences(
                        correspondence, view_mats[i], view_mats[j], intrinsic_mat, triangulation_parameters)
                    point_cloud_builder.add_points(corr_ids, points3d)

                for j in range(frame_count):
                    if view_mats[j] is None:
                        continue

                    intersection, indices = snp.intersect(point_cloud_builder.ids.squeeze(1),
                                                          corner_storage[j].ids.squeeze(1), indices=True)
                    points_cloud = point_cloud_builder.points[indices[0]]
                    points_corners = corner_storage[j].points[indices[1]]
                    inliers = calc_inlier_indices(
                        points_cloud, points_corners, intrinsic_mat @ view_mats[j], max_error
                    )
                    point_cloud_builder.remove_points(np.delete(intersection, inliers))

    remove_none(view_mats)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
