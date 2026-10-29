# -*- coding: utf-8 -*-
"""命令行入口：argparse、intrinsic.xml 解析、与参考结果对比。"""
import argparse
import os
import re
import sys

import numpy as np

from .exceptions import CalibError
from .calibration import run

# 工程根（本文件在 calib/ 下，上溯一级）—— 用于默认 --ref-dir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_intrinsic_model(path):
    """读取 intrinsic.xml 的相机模型标签；无标签视为 kb（历史兼容格式）。"""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            txt = f.read()
    except OSError as e:
        raise CalibError("无法读取内参文件 %s: %s" % (path, e))
    m = re.search(r'<camera_model[^>]*>\s*([A-Za-z0-9_]+)\s*</camera_model>', txt)
    return m.group(1).lower() if m else "kb"


def parse_intrinsic_xml(path):
    """解析 cv::FileStorage 写出的 intrinsic.xml。

    【坑】不能对整个文件做 findall —— <?xml version="1.0"?> 里的 1.0 也会被抓进去。
    必须只取 <intrinsic_matrix> / <distortion_coeffs> 标签内部。

    畸变系数按模型返回：kb（含无标签的历史文件）取前 4 个；
    brown（带 <camera_model>brown</camera_model> 标签）返回完整 5/8 个。
    """
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            txt = f.read()
    except OSError as e:
        raise CalibError("无法读取内参文件 %s: %s" % (path, e))
    num = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    m = re.search(r'<intrinsic_matrix[^>]*>(.*?)</intrinsic_matrix>', txt, re.S)
    km = [float(x) for x in re.findall(num, m.group(1))] if m else []
    m2 = re.search(r'<distortion_coeffs[^>]*>(.*?)</distortion_coeffs>', txt, re.S)
    dm = [float(x) for x in re.findall(num, m2.group(1))] if m2 else []
    if len(km) < 9:
        raise CalibError("内参文件 %s 缺少 <intrinsic_matrix>（或元素不足 9 个）" % path)
    if len(dm) < 4:
        raise CalibError("内参文件 %s 缺少 <distortion_coeffs>（或元素不足 4 个）" % path)
    mm = re.search(r'<camera_model[^>]*>\s*([A-Za-z0-9_]+)\s*</camera_model>', txt)
    if mm and mm.group(1).lower() == "brown":
        return np.array(km[:9]).reshape(3, 3), np.array(dm)
    return np.array(km[:9]).reshape(3, 3), np.array(dm[:4])


def compare(res_dir, K, D, kept, img_pts, model=None):
    """与已有的 Result/ 对比（内参仅在同相机模型下对比；角点始终可对比）"""
    print("\n" + "=" * 78)
    print("与参考结果对比")
    print("=" * 78)
    cur_model = getattr(model, "name", model) or "kb"
    ref_xml = os.path.join(res_dir, "intrinsic.xml")
    if os.path.exists(ref_xml):
        try:
            ref_model = parse_intrinsic_model(ref_xml)
            if ref_model != cur_model:
                print("  模型不同（参考 %s vs 本次 %s），跳过内参对比"
                      % (ref_model, cur_model))
                Kr = None
            else:
                Kr, Dr = parse_intrinsic_xml(ref_xml)
        except CalibError as e:
            print("  解析参考内参失败，跳过内参对比: %s" % e)
            Kr = None
        if Kr is not None:
            print("参考(%s):" % ref_xml)
            print("参考 K:\n%s" % np.array2string(Kr, precision=8))
            print("本次 K:\n%s" % np.array2string(K, precision=8))
            print("dK(max) = %.6e px   dD(max) = %.6e"
                  % (np.abs(K - Kr).max(), np.abs(D - Dr).max()))
            print("  d_fx=%+.6f d_fy=%+.6f d_cx=%+.6f d_cy=%+.6f"
                  % (K[0, 0] - Kr[0, 0], K[1, 1] - Kr[1, 1],
                     K[0, 2] - Kr[0, 2], K[1, 2] - Kr[1, 2]))
    else:
        print("  未找到参考内参: %s" % ref_xml)
    ref_txt = os.path.join(res_dir, "imgPoint_intrinsic.txt")
    if os.path.exists(ref_txt):
        try:
            with open(ref_txt, encoding='utf-8') as f:
                v = [float(t) for t in f.read().split()]
            n, bw, bh = int(v[0]), int(v[1]), int(v[2])
            ref = np.array(v[3:], np.float64).reshape(n, bw * bh, 2)
        except (OSError, ValueError, IndexError) as e:
            print("  解析参考角点失败，跳过角点对比: %s" % e)
        else:
            if ref.shape == img_pts.shape:
                d = np.abs(ref - img_pts)
                print("角点 vs imgPoint_intrinsic.txt: 平均=%.6f px 最大=%.6f px"
                      % (d.mean(), d.max()))
            else:
                print("角点形状不一致: 参考 %s vs 本次 %s" % (ref.shape, img_pts.shape))
    else:
        print("  未找到参考角点: %s" % ref_txt)


def main():
    ap = argparse.ArgumentParser(description="相机内参标定（KB 鱼眼 / Brown 模型）")
    ap.add_argument("case", nargs="?", default="aaa", help="Data 下的子目录名，如 aaa")
    ap.add_argument("--root", default=os.path.join(_ROOT, "data"),
                    help="Data 目录（默认项目内 data/）")
    ap.add_argument("--no-undistort", action="store_true", help="跳过去畸变与覆盖图")
    ap.add_argument("--cv2", action="store_true", help="强制使用 pip 的 cv2 后端")
    ap.add_argument("--compare", action="store_true", help="与参考结果对比")
    ap.add_argument("--reproj-filter", action="store_true",
                    help="开启逐帧重投影离群值剔除（默认关闭）")
    ap.add_argument("--model", choices=["kb", "brown", "auto"], default=None,
                    help="相机模型：kb=Kannala-Brandt 鱼眼(默认)、brown=径向切向、"
                         "auto=按 FOV 自动选择（缺省取 config.xml 的 camera_model）")
    ap.add_argument("--brown-coeffs", type=int, choices=[5, 8], default=None,
                    help="Brown 畸变系数个数（缺省取 config.xml 的 distortion_coeffs，"
                         "默认 5）")
    ap.add_argument("--no-homography-check", action="store_true",
                    help="关闭角点网格一致性校验")
    ap.add_argument("--detect-vis", action="store_true",
                    help="逐帧保存检测中间结果可视化（原图+角点连线，失败帧标注原因），"
                         "最长边缩放到 1280 后以 JPEG q=90 保存到 Result/detect_vis/")
    ap.add_argument("--ref-dir", default=os.path.join(_ROOT, "data", "reference"),
                    help="参考结果目录（默认 data/reference/，避免读到本次刚写出的文件）")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    try:
        out = run(a.case, a.root, no_undistort=a.no_undistort,
                  prefer320=not a.cv2, verbose=not a.quiet,
                  reproj_filter=a.reproj_filter,
                  homography_check=not a.no_homography_check,
                  detect_vis=a.detect_vis,
                  model=a.model, brown_coeffs=a.brown_coeffs)
    except CalibError as e:
        print("\n[标定失败] %s" % e, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[已取消] 用户中断", file=sys.stderr)
        return 130
    except Exception as e:
        print("\n[未预期错误] %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1

    if a.compare:
        try:
            ref_dir = a.ref_dir
            compare(ref_dir, out["K"], out["D"], out["kept"], out["img_pts"],
                    model=out.get("model", "kb"))
        except CalibError as e:
            print("\n[对比失败] %s" % e, file=sys.stderr)
            return 2
    return 0
