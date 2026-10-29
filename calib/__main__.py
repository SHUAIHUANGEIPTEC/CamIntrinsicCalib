# -*- coding: utf-8 -*-
"""支持 `python -m calib` 直接运行命令行入口。"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
