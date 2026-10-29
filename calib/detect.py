# -*- coding: utf-8 -*-
"""角点检测 + 帧筛选：灰度化、bs 循环二值化检测、网格校验、可视化、txt 角点。"""
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .exceptions import CalibError, DataError


def to_gray_rgb2gray(bgr):
    """定点实现的灰度化（与 cvtColor(..., COLOR_RGB2GRAY) 逐像素一致）。

    注意通道语义：本函数对 BGR 数据使用 RGB2GRAY 权重——即把 B 通道按 R、
    R 通道按 B 的权重加权。对灰白棋盘格影响极小，但会改变 adaptiveThreshold
    的结果，是检测链路的既定行为，须保持不变。

    关键：OpenCV 只认「通道下标」，不认通道语义。COLOR_RGB2GRAY 把
    **channel0 当作 R、channel2 当作 B**，而 imread 给的是 BGR，于是
    channel0 实际是 B、channel2 实际是 R —— 权重是被"调反"了的。

    【坑1】必须用固定点实现，浮点 0.299/0.587/0.114 有 ±1 舍入差，
           实测会让个别帧二值化结果不同、多丢一帧：
               Y = (c0*4899 + c1*9617 + c2*1868 + 8192) >> 14
    【坑2】必须用 int32：255*9617 超出 uint16，溢出会让**所有**帧检测失败。
    """
    c0 = bgr[:, :, 0].astype(np.int32)
    c1 = bgr[:, :, 1].astype(np.int32)
    c2 = bgr[:, :, 2].astype(np.int32)
    y = (c0 * 4899 + c1 * 9617 + c2 * 1868 + 8192) >> 14
    return np.clip(y, 0, 255).astype(np.uint8)


def detect_corners(backend, gray, bw, bh):
    """自适应阈值块尺寸搜索：bs 从 301 递减到 31，步长 -10，命中即用。
    """
    for bs in range(301, 31, -10):
        binimg = backend.adaptive_threshold(gray, bs)
        ok, c = backend.find_corners(binimg, bw, bh)
        if ok and c is not None and len(c) == bw * bh:
            return bs, c
    return None, None


def check_grid_consistency(img_corners, bw, bh):
    """角点网格规则性校验：验证检测到的角点是否构成**无折叠、无错乱的规则网格**。

    这是模型无关的「错误检出」拦截：无论相机是针孔还是鱼眼，真实棋盘格的角点在
    行列方向上都保持**单调有序**（不会折叠、不会交叉）。而「假棋盘格」（重复纹理）
    或「角点顺序错乱」会破坏这种单调性。

    校验项：
      1. 行内 x 坐标单调（允许微小噪声），列内 y 坐标单调
      2. 相邻角点间距的变异系数不过大（拦截大范围乱序）

    返回 (ok, reason)：ok=False 时 reason 说明原因。
    """
    g = np.asarray(img_corners, np.float64).reshape(bh, bw, 2)
    # 1) 单调性：行内 x、列内 y 的反向差分次数
    dx = np.diff(g[:, :, 0], axis=1)          # (bh, bw-1)
    dy = np.diff(g[:, :, 1], axis=0)          # (bh-1, bw)
    # 允许极小噪声（< 1px 的反向视为抖动），只统计显著反向
    rev_x = int((dx < -1.0).sum())
    rev_y = int((dy < -1.0).sum())
    tol = int(max(bw, bh) * 0.3)              # 反向差分超过 30% 视为错乱
    if rev_x > tol or rev_y > tol:
        return False, "行列单调性破坏（反向 %d/%d）" % (rev_x, rev_y)
    # 2) 相邻间距变异系数：鱼眼畸变下边缘格子间距会变，但整体 CV 有限
    drow = np.linalg.norm(g[:, 1:] - g[:, :-1], axis=2).ravel()
    dcol = np.linalg.norm(g[1:] - g[:-1], axis=2).ravel()
    d_all = np.concatenate([drow, dcol])
    d_all = d_all[d_all > 1e-3]
    if len(d_all) == 0:
        return False, "角点退化（间距全为 0）"
    cv = float(d_all.std() / d_all.mean())
    if cv > 0.8:   # 鱼眼下边缘格距变化本就明显，0.8 只拦截点序错乱这类极端散布
        return False, "间距变异系数过大(%.2f)" % cv
    return True, ""


def draw_detection_vis(bgr, corners, bw, bh, found, note=""):
    """在原始 BGR 图上绘制角点检测的中间可视化。

    - 成功帧（found=True）：drawChessboardCorners 画出角点与棋盘格连线；
    - 失败帧（found=False）：保持原图，左上角用红字标注失败原因；
    - note 非空时（无论成功/失败）都会在左上角叠加状态文字，成功帧用橙色、失败帧用红色。
    """
    vis = bgr.copy()
    if found and corners is not None and len(corners) == bw * bh:
        cv2.drawChessboardCorners(
            vis, (bw, bh), np.asarray(corners, np.float32).reshape(-1, 1, 2), True)
        color = (0, 200, 255)       # 橙色：成功但带有提示信息
    else:
        color = (0, 0, 255)         # 红色：检测失败
    if note:
        cv2.putText(vis, note, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA)
    return vis


def save_scaled_jpeg(img, path, max_side=1280, quality=90):
    """按最长边缩放到 max_side（保持纵横比）后，以 JPEG 压缩保存。

    quality ∈ [0,100] 为 OpenCV 的 JPEG 质量参数，默认 90。
    缩放使用 INTER_AREA，适合降采样；不超过 max_side 时保持原尺寸。
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = float(max_side) / longest
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise DataError("写可视化图失败: %s" % path)


def check_chessboard(backend, cfg, img_paths, verbose=True, homography_check=True,
                     save_vis=False, vis_dir=None):
    """「Start check chessboard.」阶段：逐帧检测 + r_board 筛选 + 网格一致性校验

    当 save_vis=True 时，逐帧把检测中间结果（原图 + 角点连线，失败帧标注原因）
    最长边缩放到 1280 后以 JPEG(q=90) 压缩保存到 vis_dir/。
    """
    if cv2 is None:
        raise DataError("未安装 OpenCV（cv2），无法读取图像")
    if save_vis:
        if vis_dir is None:
            vis_dir = "detect_vis"
        try:
            os.makedirs(vis_dir, exist_ok=True)
        except OSError as e:
            raise CalibError("无法创建可视化输出目录 %s: %s" % (vis_dir, e))

    def _save_vis(bgr, corners, found, note=""):
        if not save_vis:
            return
        try:
            vis = draw_detection_vis(bgr, corners, cfg.board_size_width,
                                     cfg.board_size_height, found, note)
            save_scaled_jpeg(vis, os.path.join(vis_dir, name + ".jpg"))
        except Exception as e:
            if verbose:
                print("  [vis] frame %s 可视化保存失败(%s)" % (name, e))

    kept, dropped_detect, dropped_radius = [], [], []
    dropped_homography = []                               # 网格一致性校验不通过的帧
    corners = []
    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        bgr = cv2.imread(p)
        if bgr is None:
            dropped_detect.append((name, "imread 失败"))
            if verbose:
                print("  Image: %s 无法读取（imread 失败），已跳过" % name)
            continue
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            dropped_detect.append((name, "非 3 通道彩色图"))
            if verbose:
                print("  Image: %s 非彩色图，已跳过" % name)
            continue
        gray = to_gray_rgb2gray(bgr)
        try:
            bs, c = detect_corners(backend, gray, cfg.board_size_width, cfg.board_size_height)
        except Exception as e:
            dropped_detect.append((name, "检测异常: %s" % e))
            if verbose:
                print("  Image: %s 角点检测异常(%s)，已跳过" % (name, e))
            _save_vis(bgr, None, False, "检测异常")
            continue
        if c is None:
            dropped_detect.append((name, "cannot be found any corners"))
            if verbose:
                print("  Image: %s cannot be found any corners." % name)
            _save_vis(bgr, None, False, "未检出角点")
            continue
        # 亚像素细化：在**灰度图**上做；细化失败则跳过该帧，不影响整体
        try:
            c = backend.corner_subpix(gray, c)
        except Exception as e:
            dropped_detect.append((name, "亚像素细化异常: %s" % e))
            if verbose:
                print("  Image: %s 亚像素细化异常(%s)，已跳过" % (name, e))
            _save_vis(bgr, None, False, "亚像素细化异常")
            continue
        # 网格一致性校验：拦截「假棋盘格」/角点顺序错乱
        if homography_check:
            gok, greason = check_grid_consistency(
                c, cfg.board_size_width, cfg.board_size_height)
            if not gok:
                dropped_homography.append((name, greason))
                if verbose:
                    print("  Image: %s 网格不一致(%s)，判为错误检出，丢弃" % (name, greason))
                _save_vis(bgr, c, False, "网格不一致: %s" % greason)
                continue
        d = np.hypot(c[:, 0] - cfg.CX, c[:, 1] - cfg.CY)
        if float(d.max()) > cfg.r_board:
            dropped_radius.append((name, float(d.max())))
            if verbose:
                print("  Image: %s too close. (rmax=%.1f > r_board=%.1f)"
                      % (name, d.max(), cfg.r_board))
            _save_vis(bgr, c, True, "超出半径: rmax=%.1f > %d" % (d.max(), cfg.r_board))
            continue
        kept.append(name)
        corners.append(c)
        if verbose:
            print("  frame %-6s bs=%-4d rmax=%7.2f  keep" % (name, bs, d.max()))
        _save_vis(bgr, c, True, "")
    return kept, corners, dropped_detect, dropped_radius, dropped_homography


def load_txt_corners(paths, per):
    """txt 角点模式：直接读取已有角点"""
    out = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                vals = [float(t) for t in f.read().split()]
        except (OSError, ValueError) as e:
            raise DataError("读取角点文件 %s 失败: %s" % (p, e))
        if len(vals) < 3:
            raise DataError("角点文件 %s 内容不足（至少需要 n/bw/bh 三个整数）" % p)
        try:
            n, bw, bh = int(vals[0]), int(vals[1]), int(vals[2])
        except (ValueError, TypeError):
            raise DataError("角点文件 %s 头部格式错误（n/bw/bh 应为整数）" % p)
        if bw * bh != per:
            raise DataError("角点文件 %s 的棋盘格 %dx%d 与 config 的 %d 点不一致" %
                            (p, bw, bh, per))
        need = n * bw * bh * 2
        if len(vals) - 3 != need:
            raise DataError("角点文件 %s 数据量不符：期望 %d 个浮点，实际 %d" %
                            (p, need, len(vals) - 3))
        a = np.array(vals[3:], np.float64).reshape(n, bw * bh, 2)
        out.extend(list(a))
    if not out:
        raise DataError("未从任何 txt 读取到角点")
    return out
