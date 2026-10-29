# -*- coding: utf-8 -*-
"""
dllpath.py —— 统一解析 OpenCV 3.2.0 DLL 的位置，供 ocv320.py / ocv320_calib.py 使用。

本模块位于 calib/ 包内，lib/ 在**工程根**目录，故需上溯一级（_ROOT）寻址。

优先级（本地优先，便于整个工程打包转移）：
    1. 工程根 lib/ 下的 opencv_world320.dll（自包含）
    2. 环境变量 OPENCV320_DIR 指定的 bin 目录
"""
import os

# 本文件在 <工程根>/calib/ 下，lib/ 在 <工程根>/lib/，故上溯一级
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidates():
    local = os.path.join(_ROOT, "lib")
    yield local
    env = os.environ.get("OPENCV320_DIR")
    if env:
        yield env


def find_dll_dir():
    """返回第一个包含 opencv_world320.dll 的目录，找不到返回 None"""
    for d in _candidates():
        if os.path.isfile(os.path.join(d, "opencv_world320.dll")):
            return os.path.abspath(d)
    return None


def resolve():
    """返回 (dll_dir, dll_path)；本地 lib 不存在时抛 FileNotFoundError"""
    d = find_dll_dir()
    if d is None:
        raise FileNotFoundError(
            "找不到 opencv_world320.dll。请将 OpenCV 3.2.0 的 DLL 放入本目录 lib/，"
            "或设置环境变量 OPENCV320_DIR 指向其 bin 目录。")
    return d, os.path.join(d, "opencv_world320.dll")


if __name__ == "__main__":
    print("DLL dir :", find_dll_dir())
    print("DLL path:", resolve()[1])
