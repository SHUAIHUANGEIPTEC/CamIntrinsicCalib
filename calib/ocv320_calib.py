# -*- coding: utf-8 -*-
"""
ocv320_calib.py —— 通过 ctypes 调用 opencv_world320.dll 中的 cv::fisheye::calibrate。

用途：需要与 OpenCV 3.2.0 位级一致的标定求解时使用（pip 的 cv2 为 4.x/5.x，
求解行为存在差异）。结构体布局按 MSVC x64 ABI / OpenCV 3.2.0 内部定义：
  _InputArray : { int flags; void* obj; Size sz; }        24 字节
  cv::Mat     : 96 字节 (flags,dims,rows,cols,data,...,size.p,step.p,step.buf[2])
  std::vector : { T* first; T* last; T* end; }            24 字节 (MSVC x64)
  TermCriteria: { int type; int maxCount; double eps; }   16 字节 -> 按指针传
"""
import ctypes
import os

import numpy as np

# DLL 位置：工程根 lib/ 优先，其次 OPENCV320_DIR（见 dllpath.py）
from .dllpath import resolve as _resolve_dll

DLL_DIR, DLL = _resolve_dll()
os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ.get("PATH", "")

try:
    os.add_dll_directory(DLL_DIR)
except Exception:
    pass

try:
    dll = ctypes.CDLL(DLL)
except OSError as e:
    raise ImportError("加载 OpenCV 3.2.0 失败（%s）: %s" % (DLL, e))

MANGLED = ("?calibrate@fisheye@cv@@YANAEBV_InputArray@2@0AEBV?$Size_@H@2@"
           "AEBV_InputOutputArray@2@2AEBV_OutputArray@2@3HVTermCriteria@2@@Z")

# _InputArray.flags 常量（OpenCV 3.2.0 内部定义）
IA_OBJPTS = 0x81040015   # STD_VECTOR_VECTOR + CV_32FC3
IA_IMGPTS = 0x8104000d   # STD_VECTOR_VECTOR + CV_32FC2
IA_MAT64 = 0x3010000     # MAT + CV_64F
IA_VEC64C3 = 0x82030016  # STD_VECTOR + CV_64FC3


class Size2i(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int)]


class TermCriteria(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("maxCount", ctypes.c_int),
                ("epsilon", ctypes.c_double)]


def make_ia(flags, obj_addr):
    """构造 _InputArray：{int flags; void* obj; Size sz{0,0}}"""
    buf = ctypes.create_string_buffer(24)
    ctypes.cast(buf, ctypes.POINTER(ctypes.c_int))[0] = flags
    ctypes.cast(ctypes.byref(buf, 8), ctypes.POINTER(ctypes.c_void_p))[0] = obj_addr
    return buf


def make_empty_mat():
    """96 字节全零的空 cv::Mat"""
    return ctypes.create_string_buffer(96)


def read_mat(buf):
    """从 Mat 结构里读出矩阵（CV_64F）；偏移量布局见文件头（data 指针在 +16）"""
    p = ctypes.cast(buf, ctypes.POINTER(ctypes.c_int))
    flags, dims, rows, cols = p[0], p[1], p[2], p[3]
    data = ctypes.cast(ctypes.byref(buf, 16), ctypes.POINTER(ctypes.c_void_p))[0]
    if not data:
        return None
    depth = flags & 7
    cn = ((flags >> 3) & 63) + 1
    if depth != 6:      # CV_64F
        return None
    arr = np.ctypeslib.as_array(
        ctypes.cast(data, ctypes.POINTER(ctypes.c_double)),
        shape=(rows, cols * cn))
    return arr.copy(), rows, cols, cn


def make_vector_of_vector(n, per, itemsize, data_ptr_of):
    """构造 vector<vector<T>>：外层 24B，n 个内层各 24B"""
    inner = ctypes.create_string_buffer(n * 24)
    for i in range(n):
        base = data_ptr_of(i)
        off = i * 24
        ctypes.cast(ctypes.byref(inner, off + 0), ctypes.POINTER(ctypes.c_void_p))[0] = base
        ctypes.cast(ctypes.byref(inner, off + 8), ctypes.POINTER(ctypes.c_void_p))[0] = base + per * itemsize
        ctypes.cast(ctypes.byref(inner, off + 16), ctypes.POINTER(ctypes.c_void_p))[0] = base + per * itemsize
    outer = ctypes.create_string_buffer(24)
    addr_inner = ctypes.addressof(inner)
    ctypes.cast(ctypes.byref(outer, 0), ctypes.POINTER(ctypes.c_void_p))[0] = addr_inner
    ctypes.cast(ctypes.byref(outer, 8), ctypes.POINTER(ctypes.c_void_p))[0] = addr_inner + n * 24
    ctypes.cast(ctypes.byref(outer, 16), ctypes.POINTER(ctypes.c_void_p))[0] = addr_inner + n * 24
    return outer, inner


def make_vector(n, itemsize):
    """构造一个可写的 vector<T>（先分配 n*itemsize 的连续缓冲）"""
    buf = ctypes.create_string_buffer(n * itemsize)
    v = ctypes.create_string_buffer(24)
    a = ctypes.addressof(buf)
    ctypes.cast(ctypes.byref(v, 0), ctypes.POINTER(ctypes.c_void_p))[0] = a
    ctypes.cast(ctypes.byref(v, 8), ctypes.POINTER(ctypes.c_void_p))[0] = a + n * itemsize
    ctypes.cast(ctypes.byref(v, 16), ctypes.POINTER(ctypes.c_void_p))[0] = a + n * itemsize
    return v, buf


def fisheye_calibrate_320(obj3f, img2f, image_size, flags=14,
                          max_count=100, eps=1e-6, k_init=None):
    """obj3f: (N,P,3) float32 ; img2f: (N,P,2) float32"""
    N, P, _ = obj3f.shape
    obj3c = np.ascontiguousarray(obj3f, dtype=np.float32)
    img2c = np.ascontiguousarray(img2f, dtype=np.float32)

    outer_o, inner_o = make_vector_of_vector(
        N, P, 12, lambda i: obj3c.ctypes.data + i * P * 12)
    outer_i, inner_i = make_vector_of_vector(
        N, P, 8, lambda i: img2c.ctypes.data + i * P * 8)

    ia_o = make_ia(IA_OBJPTS, ctypes.addressof(outer_o))
    ia_i = make_ia(IA_IMGPTS, ctypes.addressof(outer_i))

    matK = make_empty_mat()
    matD = make_empty_mat()
    if k_init is not None:
        # 手工构造一个 3x3 CV_64F 的 Mat：data 指向外部 double 缓冲
        kbuf = np.ascontiguousarray(k_init, dtype=np.float64)
        ctypes.cast(matK, ctypes.POINTER(ctypes.c_int))[0] = (6) + ((1 - 1) << 3) | 0x10000  # CV_64FC1
        ctypes.cast(matK, ctypes.POINTER(ctypes.c_int))[1] = 2      # dims
        ctypes.cast(matK, ctypes.POINTER(ctypes.c_int))[2] = 3      # rows
        ctypes.cast(matK, ctypes.POINTER(ctypes.c_int))[3] = 3      # cols
        ctypes.cast(ctypes.byref(matK, 16), ctypes.POINTER(ctypes.c_void_p))[0] = kbuf.ctypes.data
        ctypes.cast(ctypes.byref(matK, 24), ctypes.POINTER(ctypes.c_void_p))[0] = kbuf.ctypes.data
        ctypes.cast(ctypes.byref(matK, 32), ctypes.POINTER(ctypes.c_void_p))[0] = kbuf.ctypes.data + 72
        ctypes.cast(ctypes.byref(matK, 40), ctypes.POINTER(ctypes.c_void_p))[0] = kbuf.ctypes.data + 72
        # size.p -> &rows
        ctypes.cast(ctypes.byref(matK, 64), ctypes.POINTER(ctypes.c_void_p))[0] = \
            ctypes.addressof(matK) + 8
        # step.p -> &step.buf[0]
        ctypes.cast(ctypes.byref(matK, 72), ctypes.POINTER(ctypes.c_void_p))[0] = \
            ctypes.addressof(matK) + 80
        step = ctypes.cast(ctypes.byref(matK, 80), ctypes.POINTER(ctypes.c_size_t))
        step[0] = 24; step[1] = 0

    ia_K = make_ia(IA_MAT64, ctypes.addressof(matK))
    ia_D = make_ia(IA_MAT64, ctypes.addressof(matD))

    v_r, buf_r = make_vector(N, 24)
    v_t, buf_t = make_vector(N, 24)
    ia_r = make_ia(IA_VEC64C3, ctypes.addressof(v_r))
    ia_t = make_ia(IA_VEC64C3, ctypes.addressof(v_t))

    sz = Size2i(image_size[0], image_size[1])
    crit = TermCriteria(3, max_count, eps)

    proto = ctypes.WINFUNCTYPE(
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p)
    try:
        fn = proto((MANGLED, dll))
    except AttributeError as e:
        raise RuntimeError("opencv_world320.dll 中找不到符号 %s（DLL 不完整或版本不符）"
                           % MANGLED)

    try:
        rms = fn(ctypes.addressof(ia_o), ctypes.addressof(ia_i), ctypes.addressof(sz),
                 ctypes.addressof(ia_K), ctypes.addressof(ia_D),
                 ctypes.addressof(ia_r), ctypes.addressof(ia_t),
                 flags, ctypes.addressof(crit))
    except (OSError, ValueError) as e:
        raise RuntimeError("fisheye::calibrate 调用失败: %s" % e)

    K = read_mat(matK)
    D = read_mat(matD)
    if K is None or D is None:
        raise RuntimeError("fisheye::calibrate 未返回有效的 K/D（可能内存布局不匹配）")
    rvecs = np.frombuffer(buf_r, dtype=np.float64).reshape(N, 3).copy()
    tvecs = np.frombuffer(buf_t, dtype=np.float64).reshape(N, 3).copy()
    # 保持引用，防止缓冲区被回收
    return rms, K, D, rvecs, tvecs, (outer_o, inner_o, outer_i, inner_i, matK, matD)

