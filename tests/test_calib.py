# -*- coding: utf-8 -*-
"""
test_calib.py —— 内参标定的测试用例

可直接运行：
    python tests/test_calib.py
或用 pytest 收集：
    pytest tests/test_calib.py -v

测试数据来自项目内 data/（config、reference、img）；数据缺失时相关用例
自动跳过（SKIP），其余纯逻辑用例仍会执行。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # 工程根
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                    # 未 pip install 时也能 import calib

import calib as ci

# 测试数据根目录：默认项目内 data/，可用环境变量 CALIB_TEST_ROOT 指向外部数据根
LOCAL = os.path.join(ROOT, "data")
TEST_ROOT = os.environ.get("CALIB_TEST_ROOT", LOCAL)
IMG_DIR = os.path.join(TEST_ROOT, "img")        # 标定图集（完整集 62 帧）
REF_DIR = os.path.join(LOCAL, "reference")      # 参考真值（内参/角点/去畸变图）


def _cfg_path():
    """返回 config.xml 的路径"""
    local = os.path.join(LOCAL, "config", "config.xml")
    if os.path.isfile(local):
        return local
    return os.path.join(TEST_ROOT, "config.xml")

# ---- 参考真值（来自 data/reference/intrinsic.xml） ----
K_REF = np.array([[5.5219979841369570e+02, 0., 9.5204054754574918e+02],
                  [0., 5.5229128589051504e+02, 7.6366494927417023e+02],
                  [0., 0., 1.]])
D_REF = np.array([-2.1060946099353429e-03, 2.8784203172450558e-05,
                  -2.9754884314735813e-03, 6.8325818294171013e-04])

# 参考保留集（字典序下保留的 52 帧）
KEPT = {'0', '1', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22',
        '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36',
        '37', '38', '39', '4', '40', '41', '43', '45', '46', '47', '48', '49', '5',
        '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '7', '8'}


class _SkipTest(Exception):
    """用例跳过（数据/环境缺失）：独立运行器识别为 SKIP；pytest 下由 _skip 转换"""


def _skip(reason):
    """数据/环境缺失时跳过用例：pytest 记 SKIP，独立运行器打印后短路"""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason)
    raise _SkipTest(reason)


def has_data():
    """标定图集（data/img）存在即可"""
    return _img0() is not None


def _img0():
    p = os.path.join(IMG_DIR, "0.png")
    return p if os.path.isfile(p) else None


def _ref_txt(name):
    p = os.path.join(REF_DIR, name)
    return p if os.path.isfile(p) else None


def _ref_xml():
    return _ref_txt("intrinsic.xml")


def _ref_d():
    return _ref_txt("0_d.png")


def load_points(path, dim):
    with open(path, encoding="utf-8") as f:
        v = [float(t) for t in f.read().split()]
    n, bw, bh = int(v[0]), int(v[1]), int(v[2]); per = bw * bh
    return n, bw, bh, np.array(v[3:], np.float64).reshape(n, per, dim)


# ==============================================================================
# 纯逻辑用例（不依赖外部数据）
# ==============================================================================
def test_config_parse():
    cfg = ci.Config(_cfg_path())
    assert cfg.W == 1920 and cfg.H == 1536
    assert cfg.board_size_width == 11 and cfg.board_size_height == 8
    assert cfg.square_size_width == 40.0 and cfg.square_size_height == 40.0
    assert cfg.r_board == 960.0
    assert cfg.PER == 88


def test_object_points():
    cfg = ci.Config(_cfg_path())
    obj = ci.build_object_points(cfg)          # (88, 3)
    assert obj.shape == (88, 3)
    assert np.allclose(obj[0], [0, 0, 0])
    assert np.allclose(obj[1], [40, 0, 0])      # 第二个：j=1
    assert np.allclose(obj[11], [0, 40, 0])     # 第 12 个：i=1,j=0 → (0,40,0)
    assert np.allclose(obj[87], [10 * 40, 7 * 40, 0])
    assert np.allclose(obj[:, 2], 0.0)          # z 全为 0


def test_fmt_fs_values_integer():
    # 整数写成 "%d."（0→"0."、1→"1."），非整数用 %.16e
    assert "1." in ci.fmt_fs_values([1.0])
    assert "0." in ci.fmt_fs_values([0.0])


def test_parse_intrinsic_xml_roundtrip():
    # 用参考 K/D 回写，再解析，应一致（且不把 <?xml version="1.0"?> 抓进来）
    import tempfile
    p = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    ci.write_intrinsic_xml(p, K_REF, D_REF)
    K2, D2 = ci.parse_intrinsic_xml(p)
    os.remove(p)
    assert np.allclose(K2, K_REF, atol=1e-9)
    assert np.allclose(D2, D_REF, atol=1e-9)


def test_xml_matches_reference_bytes():
    # 用参考 K/D 回写，<intrinsic_matrix> 块应与 data/reference/intrinsic.xml 逐字节一致
    ref = _ref_xml()
    if not ref:
        return _skip("无参考 intrinsic.xml 备份")
    import re, tempfile
    p = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    ci.write_intrinsic_xml(p, K_REF, D_REF)
    with open(p, encoding="utf-8") as f:
        gen = f.read()
    os.remove(p)
    with open(ref, encoding="utf-8") as f:
        ref_txt = f.read()

    def blk(t, tag):
        return re.search(r'<%s>(.*?)</%s>' % (tag, tag), t, re.S).group(1).strip()
    assert blk(gen, "intrinsic_matrix") == blk(ref_txt, "intrinsic_matrix")
    # 前 4 项一致（第 5 项参考是未初始化残留值，本次写 0.）
    assert blk(gen, "distortion_coeffs").split()[:4] == \
        blk(ref_txt, "distortion_coeffs").split()[:4]


# ==============================================================================
# 依赖外部数据的用例（无数据自动跳过）
# ==============================================================================
def test_rgb2gray_fixedpoint():
    if not has_data():
        return _skip("无标定图数据")
    import cv2
    bgr = cv2.imread(_img0())
    ref = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
    got = ci.to_gray_rgb2gray(bgr)
    assert np.array_equal(got, ref)             # 逐像素一致


def test_frame_filter_exact():
    # 完整检测+筛选，保留集应与参考一致（较慢，约 2 分钟）
    # 需要完整的 62 帧 data/img 数据
    if not (os.path.isdir(IMG_DIR) and len(ci.glob_sorted(
            [os.path.join(IMG_DIR, "*.png")])) >= 62):
        return _skip("无全量 62 帧 img/ 数据")
    cfg = ci.Config(_cfg_path())
    backend = ci.get_backend(prefer320=True)
    paths = ci.glob_sorted([os.path.join(IMG_DIR, "*.png")])
    kept, corners, dd, dr, dh = ci.check_chessboard(backend, cfg, paths, verbose=False)
    assert set(kept) == KEPT, "保留集不一致: 缺 %s 多 %s" % (
        sorted(KEPT - set(kept)), sorted(set(kept) - KEPT))


def test_corners_vs_reference():
    if not has_data():
        return _skip("无标定图数据")
    ref = _ref_txt("imgPoint_intrinsic.txt")
    if not ref:
        return _skip("无参考角点 txt")
    got_path = os.path.join(ROOT, "artifacts", "got_corners.npy")
    if not os.path.isfile(got_path):
        return _skip("无自检测角点缓存 artifacts/got_corners.npy")
    n, bw, bh, ref_pts = load_points(ref, 2)
    got = np.load(got_path)
    assert got.shape == ref_pts.shape
    d = np.abs(got - ref_pts)
    assert d.mean() < 0.01                     # 实测 0.0027 px
    assert d.max() < 0.5


def test_calibration_reference_matches():
    # 用参考角点做标定，应与参考内参位级一致
    ref = _ref_txt("imgPoint_intrinsic.txt")
    if not ref:
        return _skip("无参考角点 txt")
    try:
        from calib.ocv320_calib import fisheye_calibrate_320
    except ImportError as e:
        return _skip("无 OpenCV 3.2.0 DLL 环境: %s" % e)
    n, bw, bh, img = load_points(ref, 2)
    wref = _ref_txt("wordPoint_intrinsic.txt")
    if not wref:
        return _skip("无参考物点 txt")
    _, _, _, obj = load_points(wref, 3)
    rms, K, D, rv, tv, _k = fisheye_calibrate_320(
        obj.astype(np.float32), img.astype(np.float32), (1920, 1536),
        flags=14, max_count=100, eps=1e-6)
    Km = K[0] if isinstance(K, tuple) else K
    Dm = (D[0] if isinstance(D, tuple) else D).ravel()[:4]
    assert np.abs(Km - K_REF).max() < 1e-5
    assert np.abs(Dm - D_REF).max() < 1e-9


def test_undistort_matches_reference():
    # 用参考 K/D 复算去畸变，应与 data/reference 的 *_d.png 逐像素一致
    import cv2
    if not has_data():
        return _skip("无标定图数据")
    src_p = _img0()
    ref_p = _ref_d()
    if not (os.path.isfile(src_p) and os.path.isfile(ref_p)):
        return _skip("无参考去畸变图 0_d.png")
    src = cv2.imread(src_p)
    ref = cv2.imread(ref_p)
    D4 = D_REF.reshape(4, 1)
    got = cv2.fisheye.undistortImage(distorted=src, K=K_REF, D=D4,
                                     Knew=K_REF, new_size=(1920, 1536))
    assert np.array_equal(got, ref)            # 逐像素一致


# ==============================================================================
# 稳健性增强的用例（网格一致性校验 + 重投影离群剔除）
# ==============================================================================
def test_grid_consistency_normal():
    # 正常角点（参考 52 帧）不应被网格校验误杀
    ref = _ref_txt("imgPoint_intrinsic.txt")
    if not ref:
        return _skip("无参考角点 txt")
    n, bw, bh, img = load_points(ref, 2)
    for k in range(n):
        ok, reason = ci.check_grid_consistency(img[k], bw, bh)
        assert ok, "帧 %d 被误杀: %s" % (k, reason)


def test_grid_consistency_shuffled():
    # 角点乱序应被拦截
    ref = _ref_txt("imgPoint_intrinsic.txt")
    if not ref:
        return _skip("无参考角点 txt")
    n, bw, bh, img = load_points(ref, 2)
    c = img[0].copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(bw * bh, 10, replace=False)
    c[idx] = img[0][idx[::-1]]                 # 打乱 10 个点
    ok, reason = ci.check_grid_consistency(c, bw, bh)
    assert not ok, "乱序角点未被拦截"


def test_grid_consistency_row_swap():
    # 两行交换应被拦截（列单调性破坏）
    ref = _ref_txt("imgPoint_intrinsic.txt")
    if not ref:
        return _skip("无参考角点 txt")
    n, bw, bh, img = load_points(ref, 2)
    c = img[0].copy()
    c[0:bw] = img[0][bw:2 * bw]
    c[bw:2 * bw] = img[0][0:bw]
    ok, reason = ci.check_grid_consistency(c, bw, bh)
    assert not ok, "行交换未被拦截"


def test_reprojection_filter_mechanism():
    # 验证剔除机制：注入一个「角点顺序错乱」的坏帧（真实世界中的错误检出），
    # 其逐帧重投影误差应显著高于正常帧，并被 MAD 离群剔除识别。
    # 注意：单纯的棋盘整体平移会被外参（rvec/tvec）吸收，误差不会变大——这是
    # 正确行为，因为平移后的棋盘仍是合法标定视角，不属于「错误检出」。
    ref = _ref_txt("imgPoint_intrinsic.txt")
    if not ref:
        return _skip("无参考角点 txt")
    n, bw, bh, img = load_points(ref, 2)
    _, _, _, obj = load_points(_ref_txt("wordPoint_intrinsic.txt"), 3)
    # 把第 0 帧的角点顺序打乱（模拟错误检出导致的点序错乱）
    bad = img.copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(bw * bh, 30, replace=False)
    bad[0, idx] = img[0, idx[::-1]]            # 打乱 30 个点的顺序
    errs, _, _ = ci._per_frame_reproj_error(obj, bad, K_REF, D_REF)
    # 坏帧误差应显著大于其余正常帧
    assert errs[0] > 10 * np.median(errs[1:]), \
        "注入坏帧未被识别：坏帧=%.2f 其余中位=%.2f" % (errs[0], np.median(errs[1:]))
    # 进一步：整条剔除链路应把第 0 帧判为离群
    inlier, per_frame, out = ci.reprojection_outlier_filter(
        obj, bad, K_REF, D_REF)
    assert 0 in out, "MAD 剔除未识别坏帧（out=%s）" % out


# ==============================================================================
# 相机模型扩展用例（KB / Brown / 长焦 / auto）
# ==============================================================================
def _need_cv2():
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _synth_views(K, D, size, n_views=15, square=25.0, bw=11, bh=8,
                 tilt_deg=30.0, z_range=(500., 700.), seed=0):
    """生成带 Brown 畸变的合成角点 (obj_all, img_all)，越界帧自动剔除。"""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = size
    obj = np.stack(np.meshgrid(np.arange(bw) * square, np.arange(bh) * square,
                               np.zeros(1)), -1).reshape(-1, 3)
    obj_all, img_all = [], []
    for _ in range(n_views):
        ax = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
        ay = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
        Rm, _ = cv2.Rodrigues(np.array([ax, ay, 0.0]))
        t = np.array([rng.uniform(-0.2, 0.2) * W / 2,
                      rng.uniform(-0.2, 0.2) * H / 2,
                      rng.uniform(*z_range)])
        prj, _ = cv2.projectPoints(obj.astype(np.float64), Rm, t, K, D)
        p = prj.reshape(-1, 2)
        if not ((p[:, 0] > 5).all() and (p[:, 0] < W - 5).all()
                and (p[:, 1] > 5).all() and (p[:, 1] < H - 5).all()):
            continue
        obj_all.append(obj)
        img_all.append(p)
    return np.asarray(obj_all), np.asarray(img_all)


class _FakeCfg(object):
    """select_candidates 用的最小配置对象"""

    def __init__(self, model="auto", coeffs=5):
        self.camera_model = model
        self.distortion_coeffs = coeffs


def test_config_optional_model_fields():
    # 旧 config.xml（无新字段）应回落到默认 kb / 5
    cfg = ci.Config(_cfg_path())
    assert cfg.camera_model == "kb" and cfg.distortion_coeffs == 5
    # 带新字段的临时 config 应正确解析
    import tempfile
    with open(_cfg_path(), encoding="utf-8") as f:
        txt = f.read()
    txt += "\n<camera_model>brown</camera_model>\n<distortion_coeffs>8</distortion_coeffs>\n"
    p = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    cfg2 = ci.Config(p)
    os.remove(p)
    assert cfg2.camera_model == "brown" and cfg2.distortion_coeffs == 8
    # 非法取值应抛 ConfigError
    txt_bad = txt.replace("brown", "banana")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt_bad)
    try:
        ci.Config(p)
        assert False, "非法 camera_model 未报错"
    except ci.ConfigError:
        pass
    os.remove(p)


def test_brown_synthetic_roundtrip():
    # 普通视场 Brown 合成数据：fx/k1 恢复精度
    if not _need_cv2():
        return _skip("无 cv2 环境")
    W, H = 1024, 768
    K = np.array([[1000., 0, W / 2], [0, 1000., H / 2], [0, 0, 1]])
    D = np.array([-0.2, 0.03, 5e-4, -4e-4, 0.0])
    obj, img = _synth_views(K, D, (W, H), z_range=(550., 750.))
    assert len(obj) >= 8, "合成有效帧太少"
    rms, K2, D2, _, _ = ci.BrownModel(5).calibrate(obj, img, (W, H))
    assert abs(K2[0, 0] - 1000) / 1000 < 0.01, "fx 恢复偏差过大: %s" % K2[0, 0]
    assert abs(K2[0, 2] - W / 2) < 5, "cx 恢复偏差过大"
    assert abs(D2[0] - (-0.2)) < 0.02, "k1 恢复偏差过大: %s" % D2[0]
    assert rms < 0.5, "rms 过大: %s" % rms


def test_brown_reproj_error_consistency():
    # Brown 模型的逐帧重投影误差链路（含剔除机制）
    if not _need_cv2():
        return _skip("无 cv2 环境")
    W, H = 1024, 768
    K = np.array([[1000., 0, W / 2], [0, 1000., H / 2], [0, 0, 1]])
    D = np.array([-0.2, 0.03, 5e-4, -4e-4, 0.0])
    obj, img = _synth_views(K, D, (W, H), z_range=(550., 750.), seed=3)
    n = len(obj)
    obj_all = obj
    model = ci.BrownModel(5)
    errs, _, _ = ci._per_frame_reproj_error(obj_all, img, K, D, model=model)
    good = errs[np.isfinite(errs)]
    assert len(good) >= 6 and np.median(good) < 0.3, \
        "正常帧重投影过大: %.3f" % np.median(good)
    # 注入乱序坏帧 → 离群剔除应识别
    bad = img.copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(obj.shape[1], 30, replace=False)
    bad[0, idx] = img[0, idx[::-1]]
    inlier, _, out = ci.reprojection_outlier_filter(
        obj_all, bad, K, D, model=model)
    assert 0 in out, "MAD 剔除未识别 Brown 坏帧（out=%s）" % out


def test_outlier_filter_wrong_model_inflates_error():
    # 回归（B1）：Brown 标定结果若按 KB 鱼眼算逐帧误差会严重失真——
    # 生产路径必须把当前 model 传入 reprojection_outlier_filter。
    if not _need_cv2():
        return _skip("无 cv2 环境")
    W, H = 1024, 768
    K = np.array([[1000., 0, W / 2], [0, 1000., H / 2], [0, 0, 1]])
    D = np.array([-0.2, 0.03, 5e-4, -4e-4, 0.0])
    obj, img = _synth_views(K, D, (W, H), z_range=(550., 750.), seed=5)
    assert len(obj) >= 8, "合成有效帧太少"
    errs_ok, _, _ = ci._per_frame_reproj_error(
        obj, img, K, D, model=ci.BrownModel(5))
    errs_bad, _, _ = ci._per_frame_reproj_error(
        obj, img, K, D[:4], model=ci.KBModel())
    med_ok = np.median(errs_ok[np.isfinite(errs_ok)])
    n_fail_bad = int(np.sum(~np.isfinite(errs_bad)))
    finite_bad = errs_bad[np.isfinite(errs_bad)]
    med_bad = np.median(finite_bad) if len(finite_bad) else float("inf")
    # 用错模型时要么部分帧求解直接失败(inf)，要么中位误差显著膨胀
    assert n_fail_bad > 0 or med_bad > 5 * med_ok, \
        "错模型误差未失真（好=%.4f 坏=%.4f 失败帧=%d）" % (med_ok, med_bad, n_fail_bad)


def test_telephoto_small_distortion():
    # 长焦（f=6000，小畸变）：fx 恢复 + k3 过拟合被裁剪为 0
    if not _need_cv2():
        return _skip("无 cv2 环境")
    W, H = 1024, 768
    K = np.array([[6000., 0, W / 2], [0, 6000., H / 2], [0, 0, 1]])
    D = np.array([-0.02, 0.0, 0.0, 0.0, 0.0])
    obj, img = _synth_views(K, D, (W, H), n_views=18, square=40.,
                            tilt_deg=20., z_range=(3500., 6000.), seed=1)
    assert len(obj) >= 8, "合成有效帧太少"
    rms, K2, D2, rv, tv = ci.BrownModel(5).calibrate(obj, img, (W, H))
    assert abs(K2[0, 0] - 6000) / 6000 < 0.015, \
        "长焦 fx 恢复偏差过大: %s" % K2[0, 0]
    assert rms < 0.5, "长焦 rms 过大: %s" % rms
    # 小畸变下 k3 不应保留过拟合噪声（真值 0）
    assert abs(D2[4]) < 1e-3, "k3 未被裁剪（过拟合残留）: %s" % D2[4]
    # 姿态有分布时不应误报；单一姿态应报警
    assert not ci.check_pose_diversity(rv, tv), "合成正常数据误报姿态警告"
    warns = ci.check_pose_diversity([rv[0]] * len(rv), [tv[0]] * len(tv))
    assert warns, "姿态多样性检查未生效"


def test_auto_model_selection():
    # auto：广角 → KB、长焦 → Brown
    if not _need_cv2():
        return _skip("无 cv2 环境")
    import cv2
    W, H = 1024, 768
    cfg = _FakeCfg("auto")
    # 广角 f=400（FOV≈116°）：fisheye 投影生成
    K = np.array([[400., 0, W / 2], [0, 400., H / 2], [0, 0, 1]])
    D4 = np.array([-0.05, 0.01, -0.003, 0.0005]).reshape(4, 1)
    obj0 = np.stack(np.meshgrid(np.arange(11) * 25., np.arange(8) * 25.,
                                np.zeros(1)), -1).reshape(-1, 3)
    rng = np.random.default_rng(2)
    obj3, img3 = [], []
    for _ in range(15):
        rvec = np.array([np.deg2rad(rng.uniform(-35, 35)),
                         np.deg2rad(rng.uniform(-35, 35)), 0.0])
        t = np.array([rng.uniform(-150, 150), rng.uniform(-120, 120),
                      rng.uniform(400., 700.)])
        prj, _ = cv2.fisheye.projectPoints(
            obj0.astype(np.float64).reshape(1, -1, 3), rvec, t, K, D4)
        p = prj.reshape(-1, 2)
        if not ((p[:, 0] > 5).all() and (p[:, 0] < W - 5).all()
                and (p[:, 1] > 5).all() and (p[:, 1] < H - 5).all()):
            continue
        obj3.append(obj0)
        img3.append(p)
    cands, _ = ci.select_candidates(cfg, obj=np.asarray(obj3),
                                    img=np.asarray(img3), size=(W, H))
    assert [m.name for m in cands] == ["kb"], "广角 auto 应选 KB: %s" % cands
    # 长焦（f=6000，FOV≈10°）
    K2 = np.array([[6000., 0, W / 2], [0, 6000., H / 2], [0, 0, 1]])
    obj2, img2 = _synth_views(K2, np.array([-0.02, 0, 0, 0, 0]), (W, H),
                              n_views=18, square=40., tilt_deg=20.,
                              z_range=(3500., 6000.), seed=1)
    cands, _ = ci.select_candidates(cfg, obj=obj2, img=img2, size=(W, H))
    assert [m.name for m in cands] == ["brown"], "长焦 auto 应选 Brown: %s" % cands
    # solve_calibration 全链路（长焦 auto → brown）
    res = ci.solve_calibration(None, obj2, img2, (W, H), cfg, verbose=False)
    assert res["model"].name == "brown"
    assert abs(res["K"][0, 0] - 6000) / 6000 < 0.015
    assert res["decisions"], "未产生模型决策日志"


def test_brown_xml_roundtrip_and_model_tag():
    # brown XML：5/8 系数完整往返 + camera_model 标签；kb 保持旧格式
    import tempfile
    K = np.array([[6000., 0, 512.], [0, 6000., 384.], [0, 0, 1.]])
    D5 = np.array([-0.02, 3e-4, 1e-7, 2e-7, -0.05])
    D8 = np.array([-0.02, 3e-4, 1e-7, 2e-7, -0.05, 1e-9, 2e-9, 3e-12])
    p = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
    ci.write_intrinsic_xml(p, K, D5, model="brown")
    K2, D2 = ci.parse_intrinsic_xml(p)
    assert ci.parse_intrinsic_model(p) == "brown"
    assert np.allclose(D2, D5), "brown 5 系数未完整往返: %s" % D2
    ci.write_intrinsic_xml(p, K, D8, model="brown")
    K3, D3 = ci.parse_intrinsic_xml(p)
    assert np.allclose(D3, D8), "brown 8 系数未完整往返: %s" % D3
    os.remove(p)
    # kb 缺省写法：5 元素布局、无标签（历史格式）
    Dr = np.array([-2.106e-3, 2.878e-5, -2.975e-3, 6.833e-4])
    ci.write_intrinsic_xml(p, K, Dr)
    assert ci.parse_intrinsic_model(p) == "kb"
    K4, D4 = ci.parse_intrinsic_xml(p)
    assert np.allclose(D4, Dr), "kb 解析应回到 4 系数"
    os.remove(p)


# ==============================================================================
# 简易测试运行器（无 pytest 时也能跑）
# ==============================================================================
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print("PASS  %s" % name)
            passed += 1
        except _SkipTest as e:
            print("SKIP  %s: %s" % (name, e))
            skipped += 1
        except AssertionError as e:
            print("FAIL  %s: %s" % (name, e))
            failed += 1
        except Exception as e:
            print("ERROR %s: %r" % (name, e))
            failed += 1
    print("\n%d passed, %d skipped, %d failed" % (passed, skipped, failed))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
