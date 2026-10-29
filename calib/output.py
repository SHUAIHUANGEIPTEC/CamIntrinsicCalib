# -*- coding: utf-8 -*-
"""结果输出：intrinsic.xml / 角点 txt / 棋盘格覆盖图。"""
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .exceptions import CalibrationError
from .models import KBModel


def fmt_fs_values(vals, indent="  ", limit=72):
    """cv::FileStorage 写 Mat 的换行规则。

    规则（对照 data/reference/intrinsic.xml）：
      写下一个值之前，若  当前行长 + 1 + len(下一个值字符串) > 72  则先换行。
    用这条规则能与参考文件逐字节一致（K 为 4+5，D 为 3+2）。
    """
    lines, cur = [], indent
    for v in vals:
        # OpenCV writeRealNumber(): 整数值写成 "%d."（所以 0 → "0."、1 → "1."），
        # 非整数才用 %.16e。只特判 0 是不够的，K[2][2]=1 会写成长串并触发换行。
        if v == round(v) and abs(v) < 1e10:
            s = "%d." % int(round(v))
        else:
            s = "%.16e" % v
        if cur.strip() and len(cur) + 1 + len(s) > limit:
            lines.append(cur)
            cur = indent
        cur = (cur + " " + s) if cur.strip() else (indent + s)
    if cur.strip():
        lines.append(cur)
    return "\n".join(lines)


def write_intrinsic_xml(path, K, D, model=None):
    """写出 intrinsic.xml。

    KB（缺省）：D 写成 5 个元素——fisheye 只有 4 个系数，第 5 个为占位
    （参考文件中该位是未初始化残留值 5.5640221667399963e-315，本实现写 0.0）；
    输出与 cv::FileStorage 格式逐字节兼容。
    Brown：D 写 5/8 个真实系数，追加 <camera_model>brown</camera_model>
    标签供下游识别模型。
    """
    name = getattr(model, "name", model)
    if name in (None, "kb"):
        vals = list(np.asarray(D, np.float64).ravel()[:4]) + [0.0]
        model_tag = ""
    else:
        vals = list(np.asarray(D, np.float64).ravel())
        model_tag = "<camera_model>%s</camera_model>\n" % name
    body = ('<?xml version="1.0"?>\n<opencv_storage>\n'
            '<intrinsic_matrix>\n%s</intrinsic_matrix>\n'
            '<distortion_coeffs>\n%s</distortion_coeffs>\n%s</opencv_storage>\n'
            % (fmt_fs_values(K.ravel()), fmt_fs_values(vals), model_tag))
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def write_point_txt(path, arr, bw, bh):
    """与参考一致：首行 n、bw、bh，随后 n*per 行 '%f '"""
    n, per = arr.shape[0], arr.shape[1]
    with open(path, "w", encoding="utf-8") as f:
        f.write("%d \n%d\n%d\n" % (n, bw, bh))
        for k in range(n):
            for j in range(per):
                f.write("%f %f \n" % (arr[k, j, 0], arr[k, j, 1]))


def make_cover_image(cfg, corners_list, obj_pts, rvecs, tvecs, K, D, out_path,
                     model=None):
    """chessCoverImg.png：把每帧棋盘格按标定出的外参投影回图像平面并叠加，
    用来检查棋盘格轨迹是否覆盖完整。

    投影按模型分发（KB 缺省：fisheye::projectPoints(..., alpha=0.1)；
    Brown：cv2.projectPoints）。
    """
    if model is None:
        model = KBModel()
    W, H = cfg.W, cfg.H
    bw, bh = cfg.board_size_width, cfg.board_size_height
    canvas = np.zeros((H, W, 3), np.uint8)
    K64 = np.asarray(K, np.float64)
    projected = 0
    for k, rv in enumerate(rvecs):
        r = np.asarray(rv, np.float64).reshape(3)
        t = np.asarray(tvecs[k], np.float64).reshape(3)
        o = obj_pts.astype(np.float64).reshape(-1, 3)
        try:
            # alpha=0.1：覆盖图投影参数（仅 KB 生效，Brown 忽略）
            prj = model.project_points(o, r, t, K64, D, alpha=0.1)
        except Exception as e:
            print("  frame %d 投影失败，跳过: %s" % (k, e))
            continue
        p = np.asarray(prj).reshape(bh, bw, 2)
        if not np.isfinite(p).all():   # 非有限投影（越界/退化）丢弃该帧
            continue
        p = np.clip(p, [0, 0], [W - 1, H - 1])
        # 绿色填充整个棋盘格区域
        hull = np.concatenate([p[0], p[:, -1], p[-1][::-1], p[:, 0][::-1]])
        hull_i = np.round(hull).astype(np.int32)
        cv2.fillConvexPoly(canvas, hull_i, (0, 255, 0))
        # 黑色格线
        for i in range(bh):
            cv2.line(canvas,
                     tuple(np.round(p[i, 0]).astype(int)),
                     tuple(np.round(p[i, -1]).astype(int)), (0, 0, 0), 1, cv2.LINE_AA)
        for j in range(bw):
            cv2.line(canvas,
                     tuple(np.round(p[0, j]).astype(int)),
                     tuple(np.round(p[-1, j]).astype(int)), (0, 0, 0), 1, cv2.LINE_AA)
        projected += 1
    if projected == 0:
        raise CalibrationError("覆盖图生成失败：没有任何一帧能成功投影")
    cv2.imwrite(out_path, canvas)
