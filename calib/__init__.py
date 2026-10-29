# -*- coding: utf-8 -*-
"""
calib —— 相机内参标定核心包（KB 鱼眼 / Brown 模型；可选 OpenCV 3.2.0 C API 位级后端）。

公共 API 在此统一导出，辅助脚本可 `from calib import ...` 或 `import calib as ci`。
"""
from .exceptions import CalibError, ConfigError, DataError, CalibrationError
from .config import Config
from .backends import OCV320Backend, CV2Backend, get_backend
from .models import (CameraModel, KBModel, BrownModel, make_model,
                     select_candidates, solve_calibration, check_pose_diversity,
                     estimate_focal_px, horizontal_fov_deg)
from .imageio import (VIDEO_EXT, IMAGE_EXT, glob_sorted, extract_frames,
                      prepare_images)
from .detect import (to_gray_rgb2gray, detect_corners, check_grid_consistency,
                     draw_detection_vis, save_scaled_jpeg, check_chessboard,
                     load_txt_corners)
from .geometry import build_object_points
from .output import (fmt_fs_values, write_intrinsic_xml, write_point_txt,
                     make_cover_image)
from .calibration import (_check_calib_result, _per_frame_reproj_error,
                          reprojection_outlier_filter, run)
from .cli import parse_intrinsic_xml, parse_intrinsic_model, compare, main

__all__ = [
    "CalibError", "ConfigError", "DataError", "CalibrationError",
    "Config",
    "OCV320Backend", "CV2Backend", "get_backend",
    "CameraModel", "KBModel", "BrownModel", "make_model",
    "select_candidates", "solve_calibration", "check_pose_diversity",
    "estimate_focal_px", "horizontal_fov_deg",
    "VIDEO_EXT", "IMAGE_EXT", "glob_sorted", "extract_frames", "prepare_images",
    "to_gray_rgb2gray", "detect_corners", "check_grid_consistency",
    "draw_detection_vis", "save_scaled_jpeg", "check_chessboard", "load_txt_corners",
    "build_object_points",
    "fmt_fs_values", "write_intrinsic_xml", "write_point_txt", "make_cover_image",
    "_check_calib_result", "_per_frame_reproj_error", "reprojection_outlier_filter",
    "run",
    "parse_intrinsic_xml", "parse_intrinsic_model", "compare", "main",
]
