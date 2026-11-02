# -*- coding: utf-8 -*-
"""后端抽象：优先 OpenCV 3.2.0（位级一致），否则 pip 的 cv2（近似）。

【职责边界】本模块只提供 KB（cv::fisheye）标定路径与角点检测原语；
模型分发（KB / Brown / auto）在 models.solve_calibration() 中完成：
- KB 模型 → 调用本模块的 calibrate（位级一致路径）；
- Brown 模型 → 直接走 pip cv2.calibrateCamera（见 models.BrownModel），
  不经过本模块。
"""
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ---- OpenCV 3.2.0 直连后端（位级一致）；导入失败则降级到 pip 的 cv2 ----
try:
    from . import ocv320
    from . import ocv320_calib
    HAS_OCV320 = True
except Exception:
    ocv320 = ocv320_calib = None
    HAS_OCV320 = False

# 检测/求解的固定参数（经参考数据集验证，勿凭直觉调参）
THRESH_C = 20.0                                  # 自适应阈值 C
CB_FLAGS = 0xB                                   # ADAPTIVE_THRESH|NORMALIZE_IMAGE|FILTER_QUADS
SUBPIX_WIN, SUBPIX_ITERS, SUBPIX_EPS = (9, 9), 30, 0.1   # 灰度图亚像素细化
FISHEYE_FLAGS = 14                               # RECOMPUTE_EXTRINSIC|CHECK_COND|FIX_SKEW
FISHEYE_MAX_COUNT, FISHEYE_EPS = 100, 1e-6       # 求解迭代终止条件


class OCV320Backend(object):
    name = "opencv_world320.dll (OpenCV 3.2.0)"

    def adaptive_threshold(self, gray, bs):
        return ocv320.adaptive_threshold_320(gray, bs, THRESH_C, 0, 0, 255.0)

    def find_corners(self, image, bw, bh):
        return ocv320.find_corners_320(image, bw, bh, CB_FLAGS, do_subpix=False)

    def corner_subpix(self, gray, pts):
        return ocv320.corner_subpix_320(gray, pts, SUBPIX_WIN, SUBPIX_ITERS, SUBPIX_EPS)

    def calibrate(self, obj, img, size):
        # 返回 (rms, K, D, rvecs, tvecs)；K/D 可能带 keep-alive 元组，取 [0]
        rms, K, D, rv, tv, _k = ocv320_calib.fisheye_calibrate_320(
            obj, img, size, flags=FISHEYE_FLAGS,
            max_count=FISHEYE_MAX_COUNT, eps=FISHEYE_EPS)
        Km = K[0] if isinstance(K, tuple) else np.zeros((3, 3))
        Dm = (D[0] if isinstance(D, tuple) else np.zeros((4, 1))).ravel()[:4]
        return float(rms), np.asarray(Km, np.float64), np.asarray(Dm, np.float64), rv, tv


class CV2Backend(object):
    """降级后端：pip 的 OpenCV。

    差异点：OpenCV 4.x 的 fisheye::calibrate 在 K 为空时会崩
    （InitExtrinsics 断言 fabs(norm_u1) > 0），所以这里改为
    flags |= USE_INTRINSIC_GUESS 并自行给出初值。
    初值用 Zhang 闭式解粗估焦距（适配任意相机），再围绕估计值做
    邻域对数扫描兜底收敛；Zhang 失败时退回 1.5W 的通用默认。
    """
    name = "cv2 %s" % (cv2.__version__ if cv2 else "?")

    def adaptive_threshold(self, gray, bs):
        return cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, bs, 20.0)

    def find_corners(self, image, bw, bh):
        ok, c = cv2.findChessboardCorners(image, (bw, bh), CB_FLAGS)
        if not ok:
            return False, None
        return True, c.reshape(-1, 2).astype(np.float64)

    def corner_subpix(self, gray, pts):
        c = pts.astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, c, SUBPIX_WIN, (-1, -1),
                         (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
                          SUBPIX_ITERS, SUBPIX_EPS))
        return c.reshape(-1, 2).astype(np.float64)

    @staticmethod
    def _estimate_f0(obj, img, size):
        """Zhang 单应分解粗估焦距（px）；失败返回 None。"""
        try:
            Kz = cv2.initCameraMatrix2D(
                np.asarray(obj, np.float32).reshape(len(obj), -1, 3),
                np.asarray(img, np.float32).reshape(len(img), -1, 2), size)
            f = float(min(Kz[0, 0], Kz[1, 1]))
            return f if (np.isfinite(f) and f > 0) else None
        except Exception:
            return None

    def calibrate(self, obj, img, size):
        F = cv2.fisheye
        fl = (F.CALIB_RECOMPUTE_EXTRINSIC | F.CALIB_CHECK_COND |
              F.CALIB_FIX_SKEW | F.CALIB_USE_INTRINSIC_GUESS)
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
                FISHEYE_MAX_COUNT, FISHEYE_EPS)
        op = obj.astype(np.float32).reshape(len(obj), 1, -1, 3)
        ip = img.astype(np.float32).reshape(len(img), 1, -1, 2)
        f0 = self._estimate_f0(obj, img, size) or (1.5 * size[0])
        # 初值序列分两阶段：
        # 1) 围绕 Zhang 估计值的邻域 scale 快速路径（普通相机首选初值即命中）；
        # 2) 宽域对数网格兜底 —— Zhang 是针孔假设，对强鱼眼系统性偏大
        #    （实测本工程实例：f0=2568 而真值 552，邻域下界 0.6*f0=1541 仍差
        #    ~2.8 倍，全落空导致误降级 Brown）。fisheye 收敛带窄（约
        #    [1.0, 1.36]×f_true），故用 1.12 细网格保证必有点落入带内。
        seq = [f0 * s for s in (1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.5, 0.6)]
        f, hi = 0.04 * size[0], 30.0 * size[0]     # FOV≈176° ~ 超长焦
        while f <= hi:
            seq.append(f)
            f *= 1.12
        best = None
        for f in seq:
            K = np.array([[f, 0, size[0] / 2.], [0, f, size[1] / 2.], [0, 0, 1.]])
            D = np.zeros((4, 1))
            try:
                rms, Ko, Do, rv, tv = cv2.fisheye.calibrate(
                    op, ip, size, K, D, flags=fl, criteria=crit)
            except Exception:
                continue
            if not np.isfinite(Ko).all():
                continue
            if best is None or rms < best[0]:
                best = (float(rms), np.array(Ko, np.float64),
                        np.array(Do, np.float64).ravel()[:4], rv, tv)
            # rms 已达亚像素量级（<1px）即认为收敛，不再扫后续初值
            if best is not None and best[0] < 1.0:
                break
        if best is None:
            raise RuntimeError("fisheye::calibrate 失败：请检查角点质量")
        return best


def get_backend(prefer320=True):
    if prefer320 and HAS_OCV320:
        try:
            return OCV320Backend()
        except Exception:
            pass
    if cv2 is None:
        raise RuntimeError("既没有 opencv_world320.dll 也没有 cv2")
    return CV2Backend()
