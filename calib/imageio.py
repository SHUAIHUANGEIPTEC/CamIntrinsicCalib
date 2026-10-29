# -*- coding: utf-8 -*-
"""图像准备：文件遍历（字典序）、视频抽帧、按 calib_mode 分流。"""
import os

try:
    import cv2
except ImportError:
    cv2 = None

from .exceptions import DataError

VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
# 固定抽帧间隔：对 10fps 视频每 20 帧取 1（1226 帧视频恰好得到 62 张标定图）。
# 非可调参数——改动会改变帧序列，影响与参考数据集的对照
FRAME_STEP = 20


def glob_sorted(patterns):
    """按 basename 的**字典序**排序（注意不是数值序）。

    字符串比较使 "10.png" 排在 "2.png" 前面，帧编号与帧顺序完全建立在这个排序上。
    """
    out = []
    for pat in patterns:
        import glob as _g
        out.extend(_g.glob(pat))
    out.sort(key=lambda p: os.path.basename(p))
    return out


def extract_frames(video_path, out_dir, step=20):
    """从视频抽帧。

    默认 step=20：对 10fps 视频每 20 帧取 1，对应视频帧 0, 20, 40, ...
    （如 1226 帧的视频抽出 62 张标定图）。
    """
    if cv2 is None:
        raise DataError("未安装 OpenCV（cv2），无法读取视频")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise DataError("无法打开视频: %s（文件损坏或缺少解码器 opencv_ffmpeg320_64.dll）"
                            % video_path)
        idx, saved = 0, 0
        names = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                p = os.path.join(out_dir, "%d.png" % saved)
                if not cv2.imwrite(p, frame):
                    raise DataError("写帧失败: %s（磁盘满或目录无写权限）" % p)
                names.append(p)
                saved += 1
            idx += 1
    finally:
        cap.release()
    if saved == 0:
        raise DataError("视频 %s 未抽出任何帧（step=%d 可能过大）" % (video_path, step))
    return saved, names


def prepare_images(cfg, case_dir):
    """准备待标定图像，返回 (图像文件列表, 是否为 txt 角点模式)。

    分流只按 Original/ 内容的扩展名，与 config.xml 的 calib_mode / id_cam 无关
    （这两个字段仅为兼容既有 config.xml 格式而解析，当前流程不参与选相机）：
      txt 角点  → *.txt；
      视频      → 抽帧到 Img/（已抽过则复用）；
      图片      → 直接使用（多视频时抽到 Img/<video_stem>/ 子目录后汇总）。
    README 中的各种机位/摆位的数据在此都收敛为上述三类输入。
    """
    original = os.path.join(case_dir, "Original")
    img_dir = os.path.join(case_dir, "Img")

    txts = glob_sorted([os.path.join(original, "*.txt")])
    if txts:
        return txts, True

    vids = glob_sorted([os.path.join(original, "*" + e) for e in VIDEO_EXT])
    imgs = glob_sorted([os.path.join(original, "*" + e) for e in IMAGE_EXT])

    # 已经抽过帧就复用（抽帧统一写入 Img/ 后再处理）
    existed = glob_sorted([os.path.join(img_dir, "*" + e) for e in IMAGE_EXT])
    if not vids:
        picked = existed or imgs
        if not picked:
            raise DataError(
                "未找到待标定图像：Original/ 目录(%s)既无视频也无图片，且 Img/ 无已抽取帧" % original)
        return picked, False

    if existed:
        return existed, False

    paths = []
    for v in vids:
        base = os.path.splitext(os.path.basename(v))[0]
        sub = img_dir if len(vids) == 1 else os.path.join(img_dir, base)
        n, ps = extract_frames(v, sub, step=FRAME_STEP)
        print("  抽帧 %s -> %d 张 (step=%d)" % (os.path.basename(v), n, FRAME_STEP))
        paths.extend(ps)
    return glob_sorted(paths), False
