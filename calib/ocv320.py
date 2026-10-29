# -*- coding: utf-8 -*-
"""
ocv320.py —— 通过 ctypes 调用 opencv_world320.dll（OpenCV 3.2.0）的 C API

目的：findChessboardCorners / cornerSubPix 在 OpenCV 不同版本间结果有差异。
为获得稳定、可复现的检测结果，本模块绕过 pip 的 cv2，直接调用 3.2.0 的
cvFindChessboardCorners / cvFindCornerSubPix（需 lib/ 下有对应 DLL，否则降级 cv2）。
"""
import ctypes
import os

import numpy as np

# DLL 位置：工程根 lib/ 优先，其次 OPENCV320_DIR（见 dllpath.py）
from .dllpath import resolve as _resolve_dll

DLL_DIR, DLL = _resolve_dll()

os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ.get("PATH", "")

_CV_8UC1 = 0
CV_TERMCRIT_ITER = 1
CV_TERMCRIT_EPS = 2


# 字段布局按 OpenCV 3.2.0 C API 定义（refcount/hdr_refcount 排列与 4.x 头文件不同），勿按新版头文件臆改
class CvMat(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("step", ctypes.c_int),
        ("refcount", ctypes.POINTER(ctypes.c_int)),
        ("hdr_refcount", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
    ]


class CvSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int)]


class CvPoint2D32f(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class CvTermCriteria(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("max_iter", ctypes.c_int),
                ("epsilon", ctypes.c_double)]


try:
    _dll = ctypes.CDLL(DLL)
except OSError as e:
    raise ImportError("加载 OpenCV 3.2.0 失败（%s）: %s。请确认 lib/ 下有 "
                      "opencv_world320.dll 及其依赖" % (DLL, e))

_dll.cvCreateMat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_dll.cvCreateMat.restype = ctypes.POINTER(CvMat)
_dll.cvReleaseMat.argtypes = [ctypes.POINTER(ctypes.POINTER(CvMat))]

_dll.cvFindChessboardCorners.argtypes = [
    ctypes.c_void_p, CvSize, ctypes.POINTER(CvPoint2D32f),
    ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_dll.cvFindChessboardCorners.restype = ctypes.c_int

_dll.cvFindCornerSubPix.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CvPoint2D32f), ctypes.c_int,
    CvSize, CvSize, CvTermCriteria]

_dll.cvAdaptiveThreshold.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]


def adaptive_threshold_320(gray: np.ndarray, block_size: int, c: float = 20.0,
                           method: int = 0, thresh_type: int = 0,
                           max_value: float = 255.0) -> np.ndarray:
    """用 OpenCV 3.2.0 的 cvAdaptiveThreshold 做自适应二值化。

    method: 0=MEAN_C, 1=GAUSSIAN_C;  thresh_type: 0=BINARY, 1=BINARY_INV
    """
    h, w = gray.shape
    src = _dll.cvCreateMat(h, w, _CV_8UC1)
    dst = _dll.cvCreateMat(h, w, _CV_8UC1)
    try:
        ctypes.memmove(src.contents.data, gray.ctypes.data,
                       np.ascontiguousarray(gray).nbytes)
        _dll.cvAdaptiveThreshold(ctypes.cast(src, ctypes.c_void_p),
                                 ctypes.cast(dst, ctypes.c_void_p),
                                 ctypes.c_double(max_value),
                                 method, thresh_type, block_size, ctypes.c_double(c))
        out = np.empty((h, w), np.uint8)
        ctypes.memmove(out.ctypes.data, dst.contents.data, out.nbytes)
        return out
    finally:
        _dll.cvReleaseMat(ctypes.pointer(src))
        _dll.cvReleaseMat(ctypes.pointer(dst))


# CALIB_CB_* 与 cv2 同名常量取值一致
CALIB_CB_ADAPTIVE_THRESH = 1
CALIB_CB_NORMALIZE_IMAGE = 2
CALIB_CB_FILTER_QUADS = 4
CALIB_CB_FAST_CHECK = 8


def find_corners_320(gray: np.ndarray, bw: int, bh: int, flags=None,
                     subpix_win=(11, 11), subpix_iters=30, subpix_eps=0.001,
                     do_subpix=True):
    """用 OpenCV 3.2.0 做棋盘格检测 + 亚像素细化。

    gray : uint8 单通道 ndarray (H, W)
    返回 (ok, corners[N,2] float64) 或 (False, None)
    """
    if flags is None:
        flags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE
    assert gray.dtype == np.uint8 and gray.ndim == 2
    h, w = gray.shape
    gray = np.ascontiguousarray(gray)

    mat = _dll.cvCreateMat(h, w, _CV_8UC1)
    try:
        ctypes.memmove(mat.contents.data, gray.ctypes.data, gray.nbytes)
        n = bw * bh
        buf = (CvPoint2D32f * n)()
        cnt = ctypes.c_int(0)
        ok = _dll.cvFindChessboardCorners(
            ctypes.cast(mat, ctypes.c_void_p), CvSize(bw, bh),
            buf, ctypes.byref(cnt), flags)
        if not ok:
            return False, None
        count = cnt.value
        if count != n:
            return False, None
        if do_subpix:
            _dll.cvFindCornerSubPix(
                ctypes.cast(mat, ctypes.c_void_p), buf, count,
                CvSize(subpix_win[0], subpix_win[1]), CvSize(-1, -1),
                CvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                               subpix_iters, subpix_eps))
        pts = np.array([[buf[i].x, buf[i].y] for i in range(count)],
                       dtype=np.float64)
        return True, pts
    finally:
        _dll.cvReleaseMat(ctypes.pointer(mat))


def corner_subpix_320(gray: np.ndarray, pts: np.ndarray,
                      subpix_win=(9, 9), subpix_iters=30, subpix_eps=0.1):
    """用 OpenCV 3.2.0 的 cvFindCornerSubPix 在**灰度图**上做亚像素细化。

    默认参数（9x9 窗口、30 次迭代、eps=0.1）为实测能稳定复现棋盘角点的组合。
    注意：检测在二值图上做，细化在原始灰度图上做 —— 二者不是同一张图。
    """
    assert gray.dtype == np.uint8 and gray.ndim == 2
    h, w = gray.shape
    gray = np.ascontiguousarray(gray)
    n = len(pts)
    mat = _dll.cvCreateMat(h, w, _CV_8UC1)
    try:
        ctypes.memmove(mat.contents.data, gray.ctypes.data, gray.nbytes)
        buf = (CvPoint2D32f * n)()
        for i, (x, y) in enumerate(pts):
            buf[i].x = float(x); buf[i].y = float(y)
        _dll.cvFindCornerSubPix(
            ctypes.cast(mat, ctypes.c_void_p), buf, n,
            CvSize(subpix_win[0], subpix_win[1]), CvSize(-1, -1),
            CvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                           subpix_iters, subpix_eps))
        return np.array([[buf[i].x, buf[i].y] for i in range(n)], dtype=np.float64)
    finally:
        _dll.cvReleaseMat(ctypes.pointer(mat))
