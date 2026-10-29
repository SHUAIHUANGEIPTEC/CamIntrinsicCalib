# -*- coding: utf-8 -*-
"""主流程：标定求解 run()，及重投影误差/离群剔除等辅助。"""
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .exceptions import CalibrationError, CalibError, ConfigError, DataError
from .config import Config
from .backends import get_backend
from .imageio import prepare_images
from .detect import check_chessboard, load_txt_corners
from .geometry import build_object_points
from .output import write_intrinsic_xml, write_point_txt, make_cover_image
from .models import (KBModel, VALID_BROWN_COEFFS, solve_calibration,
                     check_pose_diversity)


def _check_calib_result(K, D, rms, cfg):
    """校验标定结果是否合法，不合法抛 CalibrationError。

    常见失效模式：角点退化（同一直线/共面性差）→ fx/fy 发散为 NaN/inf；
    角点噪声过大 → 主点跑出图像或焦距异常。
    """
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise CalibrationError("标定结果含 NaN/Inf（角点可能退化或数量不足）")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if fx <= 0 or fy <= 0:
        raise CalibrationError("焦距非法：fx=%.4f fy=%.4f（应为正数）" % (fx, fy))
    # 主点应大致落在图像内（鱼眼模型允许略偏，给 2 倍图像宽度的宽容度）
    if not (-cfg.W <= cx <= 2 * cfg.W) or not (-cfg.H <= cy <= 2 * cfg.H):
        raise CalibrationError("主点异常：cx=%.2f cy=%.2f 偏离图像(%dx%d)过远"
                               % (cx, cy, cfg.W, cfg.H))
    # 重投影误差过大通常说明角点/模型匹配有问题（正常 < 1 px 量级）
    if rms > 10.0:
        raise CalibrationError("重投影误差 %.3f px 过大，标定结果不可信" % rms)
    return True


def _per_frame_reproj_error(obj_all, img_pts, K, D, model=None):
    """用给定 K/D 求每帧外参并计算每帧的重投影 RMS（像素）。

    model 为 CameraModel（KBModel / BrownModel），缺省 KB 鱼眼（向后兼容）。
    返回 (per_frame_rms, rvecs, tvecs)：per_frame_rms 形状 (N,)。
    对单帧失败（退化）的帧，其误差置为 inf。
    """
    if model is None:
        model = KBModel()
    N = len(obj_all)
    K64 = np.asarray(K, np.float64)
    Dv = np.asarray(D, np.float64)
    errs = np.full(N, np.inf)
    rvecs, tvecs = [None] * N, [None] * N
    for i in range(N):
        o = obj_all[i].astype(np.float64).reshape(-1, 3)
        ip = img_pts[i].astype(np.float64).reshape(-1, 2)
        try:
            rv, tv = model.solve_pnp(o, ip, K64, Dv)
            prj = model.project_points(o, rv, tv, K64, Dv)
        except Exception:
            continue
        p = np.asarray(prj).reshape(-1, 2)
        e = np.hypot(p[:, 0] - img_pts[i][:, 0], p[:, 1] - img_pts[i][:, 1])
        errs[i] = float(np.sqrt((e ** 2).mean()))
        rvecs[i] = np.asarray(rv).reshape(3)
        tvecs[i] = np.asarray(tv).reshape(3)
    return errs, rvecs, tvecs


def reprojection_outlier_filter(obj_all, img_pts, K, D,
                                mad_k=8.0, max_ratio=0.3, min_keep=3,
                                model=None):
    """逐帧重投影离群值剔除。

    先标定得到 K/D，再用它求每帧外参、算每帧重投影 RMS，用 MAD（中位数绝对偏差）
    识别离群帧（被「错误检出」污染、但整体 rms 未被拉爆的帧），返回剔除后的索引。

    model 必须传当前标定所选的 CameraModel——KB 与 Brown 的投影/求外参几何不同，
    用错模型会得到失真的逐帧误差（Brown 标定结果尤其不能按 KB 鱼眼算）。

    注意：正常标定数据本身就有误差偏高的帧（运动模糊、倾斜大等），阈值过严会误杀。
    因此 mad_k 默认放宽到 8（只抓极端离群），且默认在 run() 中**关闭**（reproj_filter=False），
    保持保守的默认行为。

    返回 (inlier_idx, per_frame_errs, out_idx)：
      inlier_idx   —— 保留帧的下标
      per_frame_errs —— 每帧重投影 RMS（剔除前）
      out_idx      —— 被判定为离群帧的下标
    """
    N = len(obj_all)
    if N <= min_keep:
        return list(range(N)), None, []
    errs, _, _ = _per_frame_reproj_error(obj_all, img_pts, K, D, model=model)
    finite = errs[np.isfinite(errs)]
    if len(finite) == 0:
        return list(range(N)), errs, []
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) if len(finite) > 1 else 0.0
    # MAD 为 0 时退化为固定阈值（数据噪声极小）
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = 0.5
    thr = med + mad_k * sigma
    inlier = [i for i in range(N) if np.isfinite(errs[i]) and errs[i] <= thr]
    outlier = [i for i in range(N) if not (np.isfinite(errs[i]) and errs[i] <= thr)]
    # 限制剔除比例，避免把整体噪声略高的正常帧全删光
    max_drop = int(N * max_ratio)
    if len(outlier) > max_drop:
        # 只保留误差最大的 max_drop 个离群
        out_sorted = sorted(outlier, key=lambda i: -errs[i])
        outlier = out_sorted[:max_drop]
        inlier = [i for i in range(N) if i not in set(outlier)]
    if len(inlier) < min_keep:
        inlier = list(range(N))
        outlier = []
    return inlier, errs, outlier


def run(case_name, root, no_undistort=False, prefer320=True, verbose=True,
        reproj_filter=False, homography_check=True, detect_vis=False,
        model=None, brown_coeffs=None):
    """主流程。

    model: 相机模型覆盖（kb / brown / auto / None=取 config.xml）；
    brown_coeffs: Brown 畸变系数个数覆盖（5 / 8 / None=取 config.xml）。
    """
    case_dir = os.path.join(root, case_name)
    cfg = Config(os.path.join(root, "config.xml"))
    if brown_coeffs is not None:
        brown_coeffs = int(brown_coeffs)
        if brown_coeffs not in VALID_BROWN_COEFFS:
            raise ConfigError("brown_coeffs 只能是 %s，当前 %d"
                              % (VALID_BROWN_COEFFS, brown_coeffs))
        cfg.distortion_coeffs = brown_coeffs
    backend = get_backend(prefer320)
    print("=" * 78)
    print("标定目录 : %s" % case_dir)
    print("配置     : %s" % cfg)
    print("后端     : %s" % backend.name)
    print("=" * 78)

    # ---------- 1. 准备图像 ----------
    print("\n[1] 准备图像")
    paths, is_txt = prepare_images(cfg, case_dir)
    print("  待处理文件 %d 个%s" % (len(paths), "（txt 角点模式）" if is_txt else ""))

    # ---------- 2. 检测角点 + 筛选 ----------
    if is_txt:
        print("\n[2] 读取 txt 角点")
        corners = load_txt_corners(paths, cfg.PER)
        # kept 必须与角点帧一一对应：一个 txt 可含多帧（头部 n），按 <stem>_<i> 命名
        kept = []
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            with open(p, encoding="utf-8", errors="replace") as f:
                n_i = int(float(f.read().split()[0]))
            if n_i <= 1:
                kept.append(stem)
            else:
                kept.extend("%s_%d" % (stem, i) for i in range(n_i))
        if len(kept) != len(corners):
            raise DataError("txt 角点帧数不一致：%d 个名称 vs %d 帧"
                            % (len(kept), len(corners)))
        dropped_detect, dropped_radius = [], []
        print("  读入 %d 帧" % len(corners))
    else:
        print("\n[2] Start check chessboard.")
        kept, corners, dropped_detect, dropped_radius, dropped_homography = \
            check_chessboard(backend, cfg, paths, verbose=verbose,
                             homography_check=homography_check,
                             save_vis=detect_vis,
                             vis_dir=os.path.join(case_dir, "Result", "detect_vis"))
        print("  检测失败 %d 帧: %s" % (len(dropped_detect), [n for n, _ in dropped_detect]))
        print("  超出 r_board %d 帧: %s" % (len(dropped_radius), [n for n, _ in dropped_radius]))
        if homography_check:
            print("  网格不一致 %d 帧: %s" % (len(dropped_homography),
                                                [n for n, _ in dropped_homography]))
    print("  ==> 保留 %d 帧" % len(kept))
    if len(kept) < 3:
        raise DataError("有效帧太少（%d），无法标定（检测失败 %d 帧、超出半径 %d 帧）"
                        % (len(kept), len(dropped_detect), len(dropped_radius)))

    img_pts = np.array(corners, np.float64)                       # (N,P,2)
    obj_pts = build_object_points(cfg)                            # (P,3)
    obj_all = np.repeat(obj_pts[None, :, :], len(kept), axis=0)   # (N,P,3)

    # ---------- 3. 标定（按相机模型分发：kb / brown / auto） ----------
    print("\n[3] Start doing calibration")
    try:
        solved = solve_calibration(backend, obj_all, img_pts, (cfg.W, cfg.H),
                                   cfg, model_override=model, verbose=verbose)
    except CalibrationError:
        raise
    except Exception as e:
        raise CalibrationError("标定求解失败: %s" % e)
    model_obj = solved["model"]
    rms, K, D = solved["rms"], solved["K"], solved["D"]
    rvecs, tvecs = solved["rvecs"], solved["tvecs"]
    print("  模型     : %s" % model_obj.name)
    _check_calib_result(K, D, rms, cfg)
    # 长焦专项：标定板姿态多样性不足只警告（焦距/畸变耦合导致解不稳）
    for w in check_pose_diversity(rvecs, tvecs):
        print("  [姿态警告] %s" % w)

    # 逐帧重投影离群值剔除：剔除被「错误检出」污染、单帧误差异常大的帧后重标定
    reproj_outliers = []
    if reproj_filter and len(kept) > 3:
        inlier_idx, per_frame_errs, out_idx = reprojection_outlier_filter(
            obj_all, img_pts, K, D, model=model_obj)
        if out_idx:
            reproj_outliers = [kept[i] for i in out_idx]
            print("  重投影离群 %d 帧（剔除并重标定）: %s" %
                  (len(out_idx), reproj_outliers))
            kept = [kept[i] for i in inlier_idx]
            corners = [corners[i] for i in inlier_idx]
            img_pts = img_pts[inlier_idx]
            obj_all = obj_all[inlier_idx]
            # 沿用已选定的模型重标定（不再重新选型），保持决策一致
            try:
                solved = solve_calibration(backend, obj_all, img_pts,
                                           (cfg.W, cfg.H), cfg,
                                           model_override=model_obj.name,
                                           verbose=verbose)
            except CalibrationError:
                raise
            except Exception as e:
                raise CalibrationError("剔除离群帧后重标定失败: %s" % e)
            model_obj = solved["model"]
            rms, K, D = solved["rms"], solved["K"], solved["D"]
            rvecs, tvecs = solved["rvecs"], solved["tvecs"]
            _check_calib_result(K, D, rms, cfg)
        else:
            print("  无重投影离群帧")

    print("  rms = %.6f px" % rms)
    print("  K =\n%s" % np.array2string(K, precision=8))
    print("  D = %s" % np.array2string(D, precision=8))
    print("  fx=%.4f fy=%.4f cx=%.4f cy=%.4f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    if model_obj.name == "brown":
        print("  （Brown 模型，畸变系数 %d 个）" % model_obj.n_coeffs)

    # ---------- 4. 输出 ----------
    res = os.path.join(case_dir, "Result")
    try:
        os.makedirs(res, exist_ok=True)
    except OSError as e:
        raise CalibError("无法创建输出目录 %s: %s" % (res, e))
    print("\n[4] 写出结果 -> %s" % res)

    try:
        write_intrinsic_xml(os.path.join(res, "intrinsic.xml"), K, D,
                            model=model_obj)
        write_point_txt(os.path.join(res, "imgPoint_intrinsic.txt"),
                        img_pts, cfg.board_size_width, cfg.board_size_height)
        write_point_txt(os.path.join(res, "wordPoint_intrinsic.txt"),
                        obj_all, cfg.board_size_width, cfg.board_size_height)
    except OSError as e:
        raise CalibError("写出结果文件失败: %s" % e)
    print("  intrinsic.xml / imgPoint_intrinsic.txt / wordPoint_intrinsic.txt")

    if not no_undistort:
        print("\n[5] Start doing undistortion")
        written = 0
        if is_txt:
            # txt 角点模式没有对应源图，逐帧 imread 必然全部失败，直接跳过写出
            print("  txt 角点模式无源图，跳过逐帧写出（覆盖图仍生成）")
        else:
            stem_to_path = {os.path.splitext(os.path.basename(p))[0]: p
                            for p in paths}
            for k, name in enumerate(kept):
                p = stem_to_path.get(name)
                src = cv2.imread(p) if p else None
                if src is None:
                    print("  frame %s 源图读取失败，跳过去畸变" % name)
                    continue
                try:
                    cv2.imwrite(os.path.join(res, "%s.png" % k), src)
                    vis = src.copy()
                    cv2.drawChessboardCorners(
                        vis, (cfg.board_size_width, cfg.board_size_height),
                        img_pts[k].astype(np.float32).reshape(-1, 1, 2), True)
                    cv2.imwrite(os.path.join(res, "%s_corner.png" % k), vis)
                    # 去畸变：Knew = K，尺寸不变（按模型分发 KB/Brown）
                    und = model_obj.undistort_image(
                        src, K, D, np.asarray(K, np.float64), (cfg.W, cfg.H))
                    cv2.imwrite(os.path.join(res, "%s_d.png" % k), und)
                    written += 1
                except Exception as e:
                    print("  frame %s 去畸变失败(%s: %s)，跳过"
                          % (name, type(e).__name__, e))
                    continue
        print("  写出 %d 组 (png / _corner.png / _d.png)" % written)

        cov = os.path.join(res, "chessCoverImg.png")
        try:
            make_cover_image(cfg, corners, obj_pts, rvecs, tvecs, K, D, cov,
                             model=model_obj)
        except CalibrationError:
            raise
        except Exception as e:
            raise CalibrationError("生成覆盖图失败: %s" % e)
        print("  chessCoverImg.png")

    print("\n完成。")
    return dict(cfg=cfg, K=K, D=D, rms=rms, kept=kept,
                img_pts=img_pts, obj_all=obj_all, reproj_outliers=reproj_outliers,
                model=model_obj.name, model_obj=model_obj,
                decisions=solved.get("decisions", []))
