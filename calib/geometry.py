# -*- coding: utf-8 -*-
"""世界坐标（object points）：Z=0 平面上的棋盘格角点。"""
import numpy as np


def build_object_points(cfg):
    """wordPoint：x = j * square_size_width, y = i * square_size_height, z = 0"""
    bw, bh = cfg.board_size_width, cfg.board_size_height
    pts = np.zeros((bw * bh, 3), np.float64)
    for i in range(bh):
        for j in range(bw):
            pts[i * bw + j, 0] = j * cfg.square_size_width
            pts[i * bw + j, 1] = i * cfg.square_size_height
    return pts
