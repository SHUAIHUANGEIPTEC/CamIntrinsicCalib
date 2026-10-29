# -*- coding: utf-8 -*-
"""配置解析：Config 对应 Data/config.xml（cv::FileStorage 风格的 XML）。"""
import re
import warnings

from .exceptions import ConfigError
from .models import VALID_MODELS, VALID_BROWN_COEFFS  # 单一定义处：models.py


class Config(object):
    """对应 Data/config.xml（cv::FileStorage 风格的 XML）"""

    FIELDS = {
        'r_board': float, 'image_width': int, 'image_height': int,
        'board_size_width': int, 'board_size_height': int,
        'square_size_width': float, 'square_size_height': float,
        # num_frame_save / calib_mode / id_cam：仅为兼容既有 config.xml 格式而
        # 解析校验；当前流程不使用（图像按扩展名分流，不按 id_cam 选相机）
        'num_frame_save': int, 'calib_mode': int, 'id_cam': int,
    }

    # 可选字段（旧 config.xml 没有时用默认值，保持向后兼容）
    OPTIONAL_FIELDS = {
        'camera_model': str,          # kb | brown | auto
        'distortion_coeffs': int,     # Brown 畸变系数个数：5 | 8（仅 brown 生效）
    }
    DEFAULTS = {
        'camera_model': 'kb',         # 默认 KB 鱼眼
        'distortion_coeffs': 5,
    }

    # 各字段的合法性约束（取值不合法时抛 ConfigError）
    POSITIVE_INT = {'image_width', 'image_height', 'board_size_width',
                    'board_size_height', 'num_frame_save'}
    POSITIVE_FLOAT = {'r_board', 'square_size_width', 'square_size_height'}
    NONNEG_INT = {'calib_mode', 'id_cam'}

    def __init__(self, path):
        self.path = path
        raw = self._parse(path)
        for k, t in self.FIELDS.items():
            if k not in raw:
                raise ConfigError("config.xml 缺少字段: %s" % k)
            val = raw[k]
            if val == '':
                raise ConfigError("config.xml 字段 <%s> 为空" % k)
            try:
                setattr(self, k, t(val))
            except (ValueError, TypeError) as e:
                raise ConfigError("config.xml 字段 <%s> 取值非法: %r (%s)" % (k, val, e))
        # 可选字段：缺省用 DEFAULTS（旧 config.xml 无这些字段时完全兼容）
        for k, t in self.OPTIONAL_FIELDS.items():
            val = raw.get(k, '')
            if val == '':
                setattr(self, k, self.DEFAULTS[k])
                continue
            try:
                setattr(self, k, t(val))
            except (ValueError, TypeError) as e:
                raise ConfigError("config.xml 字段 <%s> 取值非法: %r (%s)" % (k, val, e))
        self.camera_model = str(self.camera_model).lower()
        self._validate()
        # 标定中实际使用的量
        self.W = self.image_width
        self.H = self.image_height
        self.CX = self.W / 2.0
        self.CY = self.H / 2.0
        self.PER = self.board_size_width * self.board_size_height

    def _validate(self):
        for k in self.POSITIVE_INT:
            if getattr(self, k) <= 0:
                raise ConfigError("config.xml 字段 <%s> 必须为正整数，当前 %d" %
                                  (k, getattr(self, k)))
        for k in self.POSITIVE_FLOAT:
            if getattr(self, k) <= 0:
                raise ConfigError("config.xml 字段 <%s> 必须为正数，当前 %r" %
                                  (k, getattr(self, k)))
        for k in self.NONNEG_INT:
            if getattr(self, k) < 0:
                raise ConfigError("config.xml 字段 <%s> 必须为非负整数，当前 %d" %
                                  (k, getattr(self, k)))
        if self.r_board > max(self.image_width, self.image_height):
            # r_board 过大等价于不过滤（所有角点半径都小于它），不阻塞标定
            warnings.warn(
                "r_board(%.1f) 大于图像尺寸(%dx%d)，将保留所有棋盘格，疑似配置错误"
                % (self.r_board, self.image_width, self.image_height),
                stacklevel=2)
        if self.camera_model not in VALID_MODELS:
            raise ConfigError("config.xml 字段 <camera_model> 只能是 %s，当前 %r" %
                              (" / ".join(VALID_MODELS), self.camera_model))
        if self.distortion_coeffs not in VALID_BROWN_COEFFS:
            raise ConfigError("config.xml 字段 <distortion_coeffs> 只能是 %s，当前 %d" %
                              (" / ".join(str(v) for v in VALID_BROWN_COEFFS),
                               self.distortion_coeffs))

    @staticmethod
    def _parse(path):
        """config.xml 不是标准 XML（无根元素），这里用正则按 <key>value</key> 抽取"""
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                txt = f.read()
        except OSError as e:
            raise ConfigError("无法读取配置文件 %s: %s" % (path, e))
        # 去掉注释
        txt = re.sub(r'<!--.*?-->', '', txt, flags=re.S)
        out = {}
        for m in re.finditer(r'<\s*([A-Za-z_][\w]*)\s*>\s*([^<>]*?)\s*<\s*/\s*\1\s*>', txt):
            out[m.group(1)] = m.group(2).strip()
        return out

    def __repr__(self):
        return ("Config(W=%d H=%d board=%dx%d square=%.1fx%.1f r_board=%.1f "
                "num_frame_save=%d calib_mode=%d id_cam=%d model=%s coeffs=%d)" % (
                    self.W, self.H, self.board_size_width, self.board_size_height,
                    self.square_size_width, self.square_size_height,
                    self.r_board, self.num_frame_save, self.calib_mode, self.id_cam,
                    self.camera_model, self.distortion_coeffs))
