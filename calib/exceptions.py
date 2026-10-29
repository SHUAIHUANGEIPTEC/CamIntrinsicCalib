# -*- coding: utf-8 -*-
"""异常体系：标定流程的预期内错误层级。"""


class CalibError(Exception):
    """标定流程的预期内错误（配置、数据、环境等），带用户可读信息。"""


class ConfigError(CalibError):
    """config.xml 缺失、字段缺失或取值非法。"""


class DataError(CalibError):
    """输入数据不足或不可用（无图像、视频打不开、角点退化等）。"""


class CalibrationError(CalibError):
    """标定求解失败或结果不合法（NaN/发散/重投影过大）。"""
