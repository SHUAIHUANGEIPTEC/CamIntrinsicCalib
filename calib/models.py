# -*- coding: utf-8 -*-
"""相机模型抽象层：KB（Kannala-Brandt / cv::fisheye 4 参数）与 Brown（径向切向）。

职责划分：
- 标定 / 投影 / 求外参 / 去畸变 / 写出 五个模型相关操作统一到 CameraModel 接口；
- KBModel 复用既有后端（OCV320 DLL 位级一致；或 pip cv2 fisheye 降级），
  输出格式固定（见 output.write_intrinsic_xml）；
- BrownModel 走 pip 的 cv2.calibrateCamera（5 参数 k1 k2 p1 p2 k3，
  或 8 参数 rational k1..k6 p1 p2）；
- select_candidates() 实现 auto 策略：按粗估 FOV 决定候选模型（中间带双标定）；
- solve_calibration() 串起候选试算、参数惩罚选优、KB 失败自动降级 Brown；
- 长焦稳健性：initCameraMatrix2D 初值 + f0 对数扫描、
  小畸变 fix_k3/zero_tangent 裁剪重标、标定板姿态多样性警告。
"""
import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .exceptions import CalibrationError

# camera_model 合法取值与 Brown 系数个数
VALID_MODELS = ("kb", "brown", "auto")
VALID_BROWN_COEFFS = (5, 8)

# auto 选择的 FOV 阈值（水平全视场角，度）：
#   ≥ FOV_WIDE_DEG   广角/鱼眼 → KB 模型；
#   ≤ FOV_NARROW_DEG 普通/长焦 → Brown 模型（小 FOV 下 fisheye 参数退化）；
#   中间带           双标定，按带参数惩罚的 RMS 选优。
FOV_WIDE_DEG = 70.0
FOV_NARROW_DEG = 45.0

# 双标定选优时每个额外畸变系数的 RMS 惩罚（px），避免简单模型与复杂模型打平时
# 倾向参数更多的模型。
MODEL_PENALTY_PER_COEFF = 0.01


def _require_cv2():
    if cv2 is None:
        raise CalibrationError("该操作需要 pip 的 OpenCV（cv2），当前环境未安装")


def _as_K(K):
    return np.asarray(K, np.float64).reshape(3, 3)


def _as_D(D, n):
    """畸变系数统一为 (n,1) float64（KB 传 4，Brown 传 5/8）。"""
    d = np.asarray(D, np.float64).ravel()
    if d.size < n:
        d = np.concatenate([d, np.zeros(n - d.size)])
    return d[:n].reshape(n, 1)


def _flags_str(flags):
    """把 cv2 标定 flags 缩写成人能看的短名（用于错误报告）。"""
    names = []
    for name in ("CALIB_USE_INTRINSIC_GUESS", "CALIB_RATIONAL_MODEL",
                 "CALIB_FIX_K3", "CALIB_ZERO_TANGENT_DIST", "CALIB_FIX_K1",
                 "CALIB_FIX_K2", "CALIB_FIX_K4", "CALIB_FIX_K5", "CALIB_FIX_K6"):
        v = getattr(cv2, name, None)
        if v is not None and flags & v:
            names.append(name.replace("CALIB_", ""))
    return "|".join(names) if names else "flags=0"


# ==============================================================================
# 模型抽象
# ==============================================================================
class CameraModel(object):
    """相机模型统一接口：标定 / 投影 / 求外参 / 去畸变 / 写出。

    子类实现各模型相关操作；与模型无关的流程（角点检测、筛选、IO）
    不经过本类。
    """

    name = "?"
    n_coeffs = 0

    def calibrate(self, obj, img, size):
        """标定。返回 (rms, K(3,3), D(n,), rvecs, tvecs)。"""
        raise NotImplementedError

    def project_points(self, obj, rvec, tvec, K, D, alpha=None):
        """把 (P,3) 物点投影到像素平面，返回 (P,2)。

        alpha 仅 KB 模型有意义（fisheye::projectPoints 的 newCameraMatrix
        混合系数，覆盖图用 0.1，重投影计算不用），Brown 忽略。
        """
        raise NotImplementedError

    def solve_pnp(self, obj, img, K, D):
        """求单帧外参，返回 (rvec, tvec)；失败抛异常。"""
        raise NotImplementedError

    def undistort_image(self, src, K, D, Knew, size):
        """整幅图去畸变，返回新图。"""
        raise NotImplementedError

    def distortion_for_xml(self, D):
        """写出 intrinsic.xml 时的畸变系数列表。"""
        return list(np.asarray(D, np.float64).ravel())

    def score(self, rms):
        """双标定选优分数：RMS + 参数数量惩罚（越小越优）。

        n_coeffs - 4：以 KB 基线 4 系数（k1..k4）为免罚基数，Brown 扩展系数逐个惩罚。
        """
        return float(rms) + MODEL_PENALTY_PER_COEFF * max(0, self.n_coeffs - 4)

    def __repr__(self):
        return self.name


# ==============================================================================
# KB 模型：cv::fisheye（Kannala-Brandt 4 参数 k1..k4）
# ==============================================================================
class KBModel(CameraModel):
    """鱼眼 KB 模型（Kannala-Brandt 4 参数 k1..k4）。

    - 标定走构造注入的 backend.calibrate（优先 opencv_world320.dll 位级一致路径）；
    - 投影走 fisheye::projectPoints(..., alpha=0.1)；
    - XML 写 5 元素布局（前 4 个系数 + 1 个占位 0）。
    """

    name = "kb"
    n_coeffs = 4

    def __init__(self, backend=None):
        self.backend = backend

    def calibrate(self, obj, img, size):
        if self.backend is None:
            raise CalibrationError("KB 标定需要标定后端（backend）")
        return self.backend.calibrate(obj, img, size)

    def project_points(self, obj, rvec, tvec, K, D, alpha=None):
        _require_cv2()
        K64 = _as_K(K)
        D4 = _as_D(D, 4)
        o = np.asarray(obj, np.float64).reshape(1, -1, 3)
        try:
            if alpha is None:
                prj, _ = cv2.fisheye.projectPoints(o, rvec, tvec, K64, D4)
            else:
                prj, _ = cv2.fisheye.projectPoints(
                    o, rvec, tvec, K64, D4, alpha=alpha)
        except TypeError:
            # 旧版 cv2 的 projectPoints 不支持 alpha 关键字
            prj, _ = cv2.fisheye.projectPoints(o, rvec, tvec, K64, D4)
        return np.asarray(prj).reshape(-1, 2)

    def solve_pnp(self, obj, img, K, D):
        _require_cv2()
        K64 = _as_K(K)
        D4 = _as_D(D, 4)
        o = np.asarray(obj, np.float64).reshape(1, -1, 3)
        ip = np.asarray(img, np.float64).reshape(1, -1, 2)
        if hasattr(cv2.fisheye, "solvePnP"):
            r = cv2.fisheye.solvePnP(o, ip, K64, D4)
            return np.asarray(r[1]).reshape(3), np.asarray(r[2]).reshape(3)
        # 部分 cv2 构建（如 4.5.x 的 manylinux wheel）未导出 fisheye.solvePnP：
        # 降级为「undistortPoints 到理想针孔像平面 + 普通 solvePnP」，几何等价。
        # 【坑】undistortPoints 缺省 P=单位阵输出的是归一化坐标，必须 P=K 才回到像素系
        und = cv2.fisheye.undistortPoints(ip, K64, D4, P=K64)
        ok, rvec, tvec = cv2.solvePnP(
            o.reshape(-1, 3), und.reshape(-1, 2), K64, None)
        if not ok:
            raise CalibrationError("KB solvePnP 失败（undistortPoints 降级路径）")
        return np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)

    def undistort_image(self, src, K, D, Knew, size):
        _require_cv2()
        return cv2.fisheye.undistortImage(
            distorted=src, K=_as_K(K), D=_as_D(D, 4),
            Knew=_as_K(Knew), new_size=size)

    def distortion_for_xml(self, D):
        # XML 统一写 5 元素布局：fisheye 只有 4 个系数，第 5 个写 0 占位（格式兼容）
        return list(np.asarray(D, np.float64).ravel()[:4]) + [0.0]


# ==============================================================================
# Brown 模型：k1 k2 p1 p2 k3（5 参数）或 rational k1..k6 p1 p2（8 参数）
# ==============================================================================
class BrownModel(CameraModel):
    """径向切向 Brown 模型，走 pip 的 cv2.calibrateCamera。

    长焦稳健求解策略：
    1. 初值候选：cv2.initCameraMatrix2D（Zhang 单应分解）、内部初始化、
       f0 ∈ [0.8W, 40W] 对数扫描（覆盖普通到长焦的焦距范围），全部试算取最优；
    2. 小畸变精度：解出后若 k3 / 切向项相对径向项可忽略，
       用 FIX_K3 + ZERO_TANGENT_DIST 裁剪重标，重投影不变差才接受。
    """

    name = "brown"

    def __init__(self, n_coeffs=5):
        if n_coeffs not in VALID_BROWN_COEFFS:
            raise CalibrationError(
                "Brown 畸变系数个数只能是 %s，当前 %r"
                % ("/".join(map(str, VALID_BROWN_COEFFS)), n_coeffs))
        self.n_coeffs = int(n_coeffs)

    @property
    def calib_flags(self):
        base = 0
        if self.n_coeffs == 8:
            base |= cv2.CALIB_RATIONAL_MODEL
        return base

    def calibrate(self, obj, img, size):
        _require_cv2()
        # calibrateCamera 严格要求 vector<vector<Point3f>>（float32），类型必须转对
        op = [np.asarray(o, np.float32).reshape(-1, 1, 3) for o in obj]
        ip = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in img]
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

        best = None
        errs = []
        for K0, flags in self._initial_guesses(op, ip, size):
            try:
                rms, K, D, rv, tv = cv2.calibrateCamera(
                    op, ip, size, K0, np.zeros((self.n_coeffs, 1)),
                    flags=flags, criteria=crit)
            except Exception as e:
                errs.append("%s: %s" % (_flags_str(flags), e))
                continue
            if not (np.isfinite(K).all() and np.isfinite(D).all()):
                errs.append("%s: 结果含 NaN/Inf" % _flags_str(flags))
                continue
            if best is None or rms < best[0]:
                best = (float(rms), np.asarray(K, np.float64),
                        np.asarray(D, np.float64).ravel(), rv, tv, flags)
        if best is None:
            raise CalibrationError(
                "cv2.calibrateCamera 失败（所有初值均未收敛）。原因: %s"
                % (errs[0] if errs else "无候选初值"))

        rms, K, D, rv, tv, flags = best
        # 小畸变精度优化：k3 / 切向项可忽略时裁剪重标
        if self.n_coeffs == 5:
            refit = self._refit_small_distortion(op, ip, size, K, D, rms, flags, crit)
            if refit is not None:
                rms, K, D, rv, tv = refit
        return rms, K, D, rv, tv

    def _initial_guesses(self, op, ip, size):
        """初值候选序列 (K0, flags)：Zhang 粗解、内部初始化、f0 对数扫描。"""
        w, h = size
        out = []
        # 1) Zhang 单应分解初值（对长焦最稳健的闭式解）
        try:
            # op/ip 是 (N,1,3)/(N,1,2) 列表，initCameraMatrix2D 要 (M,N,3)
            Kz = cv2.initCameraMatrix2D(
                np.asarray(op, np.float32).reshape(len(op), -1, 3),
                np.asarray(ip, np.float32).reshape(len(ip), -1, 2), size)
            if np.isfinite(Kz).all() and Kz[0, 0] > 0:
                out.append((Kz, self.calib_flags | cv2.CALIB_USE_INTRINSIC_GUESS))
        except Exception:
            pass
        # 2) 不带初值（OpenCV 内部按视图初始化）
        out.append((np.zeros((3, 3)), self.calib_flags))
        # 3) f0 对数扫描：0.8W ~ 40W（覆盖广角到长焦）
        for s in (0.8, 1.5, 3.0, 6.0, 10.0, 20.0, 40.0):
            f = w * s
            K0 = np.array([[f, 0., w / 2.], [0., f, h / 2.], [0., 0., 1.]])
            out.append((K0, self.calib_flags | cv2.CALIB_USE_INTRINSIC_GUESS))
        return out

    def _refit_small_distortion(self, op, ip, size, K, D, rms0, base_flags, crit):
        """k3 / 切向项可忽略时用 FIX_K3 + ZERO_TANGENT_DIST 重标（防过拟合）。

        仅当裁剪后重投影不变差（相对 0.1% + 绝对 0.02px）才接受，否则返回原解。
        """
        k1, k2, p1, p2, k3 = [float(x) for x in np.asarray(D).ravel()[:5]]
        # 参考半径取图像对角对应的归一化半径（长焦 f 大 → r_max 很小，
        # k3/k6 高阶项在此范围内天然可忽略，正是要裁剪的过拟合噪声）
        f = max(abs(float(K[0, 0])), abs(float(K[1, 1])), 1e-9)
        r = min(math.hypot(size[0] / 2.0, size[1] / 2.0) / f, 1.0)
        radial = max(abs(k1) * r * r, abs(k2) * r ** 4, abs(k3) * r ** 6, 1e-12)
        if abs(k3) * r ** 6 > 0.01 * radial:
            return None                      # k3 贡献不可忽略，保留
        if max(abs(p1), abs(p2)) > 0.01 * radial:
            return None                      # 切向项不可忽略，保留
        flags = base_flags | cv2.CALIB_USE_INTRINSIC_GUESS | \
            cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        # 【坑】CALIB_FIX_K3 的语义是"固定为初值"而非置零——
        # 必须把 D 初值的 k3/p1/p2 清零，否则过拟合的 k3 会被原样保留。
        D0 = np.asarray(D, np.float64).reshape(-1, 1).copy()
        D0[2:4] = 0.0                        # p1, p2（ZERO_TANGENT_DIST 也固定为初值）
        D0[4] = 0.0                          # k3
        try:
            rms, K2, D2, rv, tv = cv2.calibrateCamera(
                op, ip, size, np.asarray(K, np.float64), D0,
                flags=flags, criteria=crit)
        except Exception:
            return None
        if not (np.isfinite(K2).all() and np.isfinite(D2).all()):
            return None
        # 容差 = 相对 0.1% + 绝对 0.02px：
        # 参数贡献已判 <1% radial（几何位移上界远小于 0.02px），
        # 合成/优质数据 rms0 极小时靠绝对项兜底，避免拒绝无害的裁剪。
        if rms > rms0 * 1.001 + 0.02:
            return None                      # 裁剪后明显变差，保留原解
        return float(rms), np.asarray(K2, np.float64), \
            np.asarray(D2, np.float64).ravel(), rv, tv

    def project_points(self, obj, rvec, tvec, K, D, alpha=None):
        _require_cv2()
        # Brown 模型无 alpha 语义，忽略
        o = np.asarray(obj, np.float64).reshape(1, -1, 3)
        prj, _ = cv2.projectPoints(
            o, np.asarray(rvec, np.float64).reshape(3, 1),
            np.asarray(tvec, np.float64).reshape(3, 1),
            _as_K(K), np.asarray(D, np.float64).ravel()[:self.n_coeffs])
        return np.asarray(prj).reshape(-1, 2)

    def solve_pnp(self, obj, img, K, D):
        _require_cv2()
        ok, rvec, tvec = cv2.solvePnP(
            np.asarray(obj, np.float64).reshape(-1, 3),
            np.asarray(img, np.float64).reshape(-1, 2),
            _as_K(K), np.asarray(D, np.float64).ravel()[:self.n_coeffs])
        if not ok:
            raise CalibrationError("cv2.solvePnP 失败")
        return np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)

    def undistort_image(self, src, K, D, Knew, size):
        _require_cv2()
        return cv2.undistort(src, _as_K(K),
                             np.asarray(D, np.float64).ravel()[:self.n_coeffs],
                             None, _as_K(Knew))


def make_model(name, brown_coeffs=5, backend=None):
    """按名字构造模型实例（kb/brown）；auto 由 select_candidates 解析。

    backend 仅供 KB 标定路径使用（Brown 不需要）。
    """
    name = (name or "kb").lower()
    if name == "kb":
        return KBModel(backend)
    if name == "brown":
        return BrownModel(brown_coeffs)
    raise CalibrationError("未知相机模型 %r（可选 kb / brown / auto）" % name)


# ==============================================================================
# auto 模型选择
# ==============================================================================
def estimate_focal_px(obj, img, size):
    """用 Zhang 单应分解粗估焦距（px），失败返回 None。"""
    if cv2 is None:
        return None
    try:
        K0 = cv2.initCameraMatrix2D(
            np.asarray(obj, np.float32), np.asarray(img, np.float32), size)
        f = float(min(K0[0, 0], K0[1, 1]))
        return f if (np.isfinite(f) and f > 0) else None
    except Exception:
        return None


def horizontal_fov_deg(f, width):
    """由焦距与图像宽估计水平全视场角（度）。"""
    return math.degrees(2.0 * math.atan(width / (2.0 * f)))


def select_candidates(cfg, model_override=None, obj=None, img=None, size=None,
                      backend=None):
    """决定候选模型列表。

    返回 (candidates, decision_lines)：
      - 显式 kb / brown → 单候选；
      - auto → 按粗估 FOV：≥70° KB、≤45° Brown、中间带双标定比较；
      - 无法粗估 → 双标定比较。
    decision_lines 是给人看的决策日志。backend 注入 KB 候选供标定使用。
    """
    requested = (model_override or getattr(cfg, "camera_model", "kb")).lower()
    n = int(getattr(cfg, "distortion_coeffs", 5))
    if requested not in VALID_MODELS:
        raise CalibrationError(
            "未知 camera_model %r（可选 %s）" % (requested, "/".join(VALID_MODELS)))
    if requested == "kb":
        return [KBModel(backend)], ["camera_model = kb（显式指定）"]
    if requested == "brown":
        return [BrownModel(n)], ["camera_model = brown（显式指定，%d 参数）" % n]

    # ---- auto ----
    f0 = None
    if obj is not None and img is not None and size is not None:
        f0 = estimate_focal_px(obj, img, size)
    if f0 is None:
        return ([KBModel(backend), BrownModel(n)],
                ["camera_model = auto：无法粗估 FOV，KB/Brown 双标定比较"])
    fov = horizontal_fov_deg(f0, size[0])
    head = "camera_model = auto：粗估 f0 = %.1f px，水平 FOV ≈ %.1f°" % (f0, fov)
    if fov >= FOV_WIDE_DEG:
        return [KBModel(backend)], ["%s（≥ %.0f°）→ KB" % (head, FOV_WIDE_DEG)]
    if fov <= FOV_NARROW_DEG:
        return [BrownModel(n)], ["%s（≤ %.0f°）→ Brown" % (head, FOV_NARROW_DEG)]
    return ([KBModel(backend), BrownModel(n)],
            ["%s（%.0f°~%.0f° 中间带）→ KB/Brown 双标定比较"
             % (head, FOV_NARROW_DEG, FOV_WIDE_DEG)])


def solve_calibration(backend, obj_all, img_pts, size, cfg,
                      model_override=None, verbose=True):
    """按候选策略标定，返回 dict(model, rms, K, D, rvecs, tvecs, decisions)。

    - KB 失败（ill-conditioned / 不收敛）自动降级 Brown 并记录原因；
    - auto 中间带双标定按带惩罚 RMS 选优；
    - 全部候选失败抛 CalibrationError。
    """
    candidates, lines = select_candidates(
        cfg, model_override=model_override, obj=obj_all, img=img_pts, size=size,
        backend=backend)
    brown_coeffs = int(getattr(cfg, "distortion_coeffs", 5))
    results, errors = [], []

    def _try(m):
        try:
            rms, K, D, rv, tv = m.calibrate(
                np.asarray(obj_all, np.float32),
                np.asarray(img_pts, np.float32), size)
            if not (np.isfinite(K).all() and np.isfinite(D).all()):
                raise CalibrationError("结果含 NaN/Inf")
            results.append((m, float(rms), np.asarray(K, np.float64),
                            np.asarray(D, np.float64).ravel(), rv, tv))
            lines.append("%s 标定成功：rms = %.6f px" % (m.name, rms))
        except Exception as e:
            errors.append("%s：%s" % (m.name, e))
            lines.append("%s 标定失败：%s" % (m.name, e))

    for m in candidates:
        _try(m)

    # KB 失败自动降级 Brown（候选中尚未包含 brown 时）
    tried_brown = any(m.name == "brown" for m in candidates)
    if not results and any(m.name == "kb" for m in candidates) and not tried_brown:
        fb = BrownModel(brown_coeffs)
        lines.append("KB 失败 → 自动降级 %d 参数 Brown 模型" % brown_coeffs)
        _try(fb)

    if not results:
        raise CalibrationError("所有候选模型均标定失败：" + "；".join(errors))

    if len(results) == 1:
        chosen = results[0]
        if len(candidates) > 1:
            lines.append("仅 %s 成功，直接采用" % chosen[0].name)
    else:
        chosen = min(results, key=lambda t: t[0].score(t[1]))
        detail = "，".join("%s rms=%.6f（score=%.6f）"
                          % (m.name, r, m.score(r)) for m, r, *_ in results)
        lines.append("候选比较：%s → 选 %s" % (detail, chosen[0].name))

    model, rms, K, D, rvecs, tvecs = chosen
    if verbose:
        for ln in lines:
            print("  " + ln)
    return dict(model=model, rms=rms, K=K, D=D,
                rvecs=rvecs, tvecs=tvecs, decisions=lines)


# ==============================================================================
# 长焦专项：标定板姿态多样性检查
# ==============================================================================
def check_pose_diversity(rvecs, tvecs,
                         min_tilt_std_deg=5.0, min_depth_rel_range=0.10):
    """检查各帧标定板姿态的多样性，返回警告文本列表（只警告不报错）。

    长焦 FOV 小，若各帧倾角/景深位移都太接近，焦距与畸变强耦合、解不稳。
      - 倾角（棋盘法线与光轴夹角）标准差 < min_tilt_std_deg 提示补倾斜姿态；
      - 景深相对极差 < min_depth_rel_range 提示补前后位移。
    """
    if cv2 is None or rvecs is None or tvecs is None:
        return []
    tilts, depths = [], []
    for rv, tv in zip(rvecs, tvecs):
        if rv is None or tv is None:
            continue
        try:
            R, _ = cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3, 1))
        except Exception:
            continue
        # 世界 Z（棋盘法线）映射到相机系；与光轴夹角
        nz = float(np.clip(R[2, 2], -1.0, 1.0))
        tilts.append(math.degrees(math.acos(nz)))
        depths.append(float(np.asarray(tv, np.float64).reshape(3)[2]))

    warns = []
    if len(tilts) >= 3:
        std = float(np.std(tilts))
        if std < min_tilt_std_deg:
            warns.append(
                "标定板倾角分布过窄：std = %.1f° < %.0f°，"
                "长焦标定建议增加倾斜姿态" % (std, min_tilt_std_deg))
    if len(depths) >= 3:
        depths = np.asarray(depths, np.float64)
        med = float(np.median(np.abs(depths))) + 1e-9
        rel = float(np.max(depths) - np.min(depths)) / med
        if rel < min_depth_rel_range:
            warns.append(
                "景深方向位移过小：相对极差 %.2f < %.2f，"
                "建议增加标定板前后位移" % (rel, min_depth_rel_range))
    return warns
