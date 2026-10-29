# camcalib —— 相机内参标定工具（KB 鱼眼 / Brown 模型）

基于棋盘格的相机内参标定工具，完整覆盖：**输入归一（视频抽帧/图片/角点 txt）→
角点检测与筛选 → 标定求解（KB 鱼眼 / Brown / auto 自动选型）→ 去畸变 → 覆盖图**。

- **KB 鱼眼**（Kannala–Brandt 4 参数）：可选 OpenCV 3.2.0 C API 直连后端，
  结果与 3.2.0 参考实现**位级一致**；无 DLL 时自动降级为 pip 的 cv2（近似）。
- **Brown 径向切向**（5/8 参数）与**长焦相机专项**：稳健初值、不收敛自动降级、
  小畸变防过拟合、姿态多样性警告。
- **稳健性**：自定义异常体系 + 网格一致性校验 + 可选逐帧重投影离群剔除，
  避免错误检出（假棋盘/点序错乱）污染内参。

---

## 目录结构

```
camcalib/
├── README.md                      # 本文件
├── pyproject.toml                 # 打包与依赖声明（pip install -e .）
│
├── calib/                         # ★ 核心包
│   ├── __init__.py / __main__.py  #   公共 API 出口 / python -m calib 入口
│   ├── cli.py / config.py         #   命令行、config.xml 解析
│   ├── imageio.py / detect.py     #   抽帧与输入归一 / 灰度化+角点检测+网格校验
│   ├── geometry.py / models.py    #   标定板物点 / KB·Brown 模型与求解分发
│   ├── calibration.py / output.py #   标定主流程 / 内参写出·覆盖图
│   ├── backends.py                #   后端抽象（3.2.0 直连优先，cv2 降级）
│   ├── ocv320.py                  #   ctypes 调用 3.2.0 角点检测/二值化
│   ├── ocv320_calib.py            #   ctypes 调用 3.2.0 fisheye::calibrate
│   ├── dllpath.py                 #   DLL 路径解析（lib/ → OPENCV320_DIR）
│   └── exceptions.py              #   异常体系 CalibError 等
│
├── tests/
│   └── test_calib.py              # ★ 测试用例（见「四、测试用例」）
│
├── data/                          # 自包含测试数据（便于打包转移）
│   ├── config/config.xml          #   标定配置
│   ├── reference/                 #   参考真值（内参/角点/去畸变图）
│   └── img/0.png                  #   示例帧（完整 62 帧图集不入库，见「环境依赖」）
└── lib/                           # OpenCV 3.2.0 运行库（不入库，见「环境依赖」）
```

---

## 快速开始

```bash
pip install -e .            # 或 pip install -e .[cv2] 同时装 pip 后端
camcalib aaa --compare      # 标定 data/<case> 并与参考结果对比
```

---

## 一、相机与标定参数

### 配置文件 `config.xml` 字段

| 字段                                         | 含义                    | 示例值         |
| ------------------------------------------ | --------------------- | ----------- |
| `r_board`                                  | 以图像中心为圆心、保留棋盘格的半径（px） | 960         |
| `image_width` / `image_height`             | 图像宽 / 高（px）           | 1920 / 1536 |
| `board_size_width` / `board_size_height`   | 棋盘格**内角点**数（宽×高）      | 11 × 8      |
| `square_size_width` / `square_size_height` | 棋盘格块尺寸（mm）            | 40 × 40     |
| `num_frame_save` / `calib_mode` / `id_cam` | 历史格式字段；仅解析校验，当前流程不使用 | 1 / 4 / 0    |
| `camera_model`（可选）                       | 相机模型：`kb` / `brown` / `auto` | `kb`（缺省）   |
| `distortion_coeffs`（可选）                  | Brown 畸变系数个数：5 / 8（仅 brown 生效） | 5           |

> `camera_model` / `distortion_coeffs` 为可选字段：缺省时回落到 `kb` + 5。
> CLI 的 `--model` / `--brown-coeffs` 可覆盖之。详见「五、多相机模型支持」。

### 标定板（棋盘格）

| 参数    | 值                   |
| ----- | ------------------- |
| 内角点数  | **11 × 8 = 88 点/帧** |
| 棋盘格尺寸 | 40 mm × 40 mm（正方形）  |
| 有效半径  | `r_board = 960` px  |

**角点分布**：88 个内角点按行列 `(i, j)` 展开，`i ∈ [0,8)` 为行（纵向）、`j ∈ [0,11)` 为列（横向），
一维索引 `idx = i * 11 + j`。世界坐标（Z=0 平面）：

```
x = j * 40.0    （mm，横向）
y = i * 40.0    （mm，纵向）
z = 0.0
```

### 标定模型与求解（KB 缺省）

| 项      | 值                                                                 |
| ------ | ----------------------------------------------------------------- |
| 相机模型   | **KB 鱼眼**（Kannala–Brandt）                                          |
| 内参矩阵 K | 3×3：`fx, 0, cx; 0, fy, cy; 0, 0, 1`（`FIX_SKEW` 强制 skew=0）         |
| 畸变系数 D | **4 维** `[k1, k2, k3, k4]`                                        |
| 求解函数   | `fisheye::calibrate`                                              |
| flags  | **14** = `RECOMPUTE_EXTRINSIC(2) \| CHECK_COND(4) \| FIX_SKEW(8)` |
| 终止条件   | `TermCriteria(COUNT+EPS, 100, 1e-6)`                              |

### 参考内参（data/img 52 帧实测）

| 内参    | 值           |
| ----- | ----------- |
| fx    | 552.1998 px |
| fy    | 552.2913 px |
| cx    | 952.0405 px |
| cy    | 763.6649 px |
| 重投影误差 | 0.3612 px   |

---

## 二、算法流程

```
config.xml
   ↓
Original/{视频|图片}  ── 抽帧(每20帧取1) ──▶  Img/{0..N}.png
   ↓  按【字典序】排序
Start check chessboard.  （逐帧）
   ├─ 定点灰度化（COLOR_RGB2GRAY 权重）
   ├─ for bs = 301; bs > 31; bs -= 10:
   │      adaptiveThreshold(gray→bin, 255, MEAN_C, BINARY, bs, C=20.0)
   │      findChessboardCorners(bin, (11,8), corners, flags=0xB)  → 命中即停
   ├─ cornerSubPix(gray, corners, (9,9), (-1,-1), (COUNT|EPS,30,0.1))
   ├─ 网格一致性校验（默认开启，拦截假棋盘/点序错乱）
   └─ 半径筛选: max hypot(角点 − 图像中心) > r_board  →  丢弃该帧
   ↓
Start doing calibration
   └─ fisheye::calibrate(obj[CV_32FC3], img[CV_32FC2], (W,H), K, D, rvecs, tvecs,
                         flags=14, (COUNT|EPS,100,1e-6))
   ↓
Start doing undistortion / 覆盖图
   ├─ fisheye::undistortImage(src, dst, K, D, Knew=K, Size())   ← 同尺寸
   └─ fisheye::projectPoints(obj, rvec, tvec, K, D, alpha=0.1)
   ↓
Result/intrinsic.xml, imgPoint_intrinsic.txt, wordPoint_intrinsic.txt,
       {k}.png, {k}_corner.png, {k}_d.png, chessCoverImg.png
```

几个**关键坑**（实现与测试均有覆盖）：

1. 文件排序是**字典序**（`0,1,10,11,…,2,20,…`），不是数值序 —— 帧编号依赖此顺序。
2. `COLOR_RGB2GRAY` 按**通道下标**解释，对 BGR 图权重实际被调反：
   `Y = (c0*4899 + c1*9617 + c2*1868 + 8192) >> 14`（必须 int32，uint16 会溢出）。
3. 角点**检测在二值图、细化在灰度图**上，不是同一张图。
4. 半径筛选判据是角点**超出** r_board 圆（丢弃远离中心的帧），不是"太靠近中心"。
5. fisheye 的 D 是 **4 维**；`intrinsic.xml` 中第 5 个数是占位 0（历史格式约定）。

---

## 三、使用说明

### 环境依赖

- Python ≥ 3.8，`numpy`
- 后端二选一（自动探测）：
  - **首选**：OpenCV 3.2.0 的 `opencv_world320.dll`，经 `ctypes` 直连 —— **位级一致**。
    DLL 查找顺序：**工程根 `lib/` → 环境变量 `OPENCV320_DIR`**，找不到则降级 cv2。
  - **降级**：`pip install opencv-python`（实测 4.10），结果近似。

> **不在版本库内的大文件**（见 `.gitignore`）：
> - `lib/opencv_world320.dll` + `lib/opencv_ffmpeg320_64.dll`（约 57 MB）——
>   取自 OpenCV 3.2.0 官方 x86 包 `build/x86/vc14/bin/`，放到 `lib/` 即可启用位级一致后端。
> - `data/img/` 下除 `0.png`（示例帧，测试引用）外的 61 帧（约 126 MB）——
>   完整图集放置于该目录；缺失时依赖全量数据的测试自动 SKIP，纯逻辑与参考真值用例不受影响。

### 命令行

```bash
# 安装后用 camcalib，或免安装直接 python -m calib
# 标定 data/aaa（完整：抽帧→检测→筛选→标定→去畸变→覆盖图）
camcalib aaa
# 等价写法
python -m calib aaa

# 只标定、跳过去畸变与覆盖图（快很多）
camcalib aaa --no-undistort

# 标定并与参考结果对比
camcalib aaa --compare

# 指定 Data 根目录 / 强制使用 pip cv2 / 静默
camcalib aaa --root /path/to/Data --cv2 --quiet

# 稳健性增强（见「六、稳健性增强」）
camcalib aaa --reproj-filter          # 开启逐帧重投影离群剔除
camcalib aaa --no-homography-check    # 关闭网格一致性校验（默认开启）

# 可视化：逐帧保存检测中间结果到 Result/detect_vis/（见「输入 / 输出」）
camcalib aaa --detect-vis

# 相机模型选择（见「五、多相机模型支持」，缺省取 config.xml 的 camera_model）
camcalib aaa --model kb        # KB 鱼眼（Kannala-Brandt 4 参数）
camcalib aaa --model brown     # Brown 径向切向（默认 5 参数）
camcalib aaa --model auto      # 按 FOV 自动选择（中间带双标定比较）
camcalib aaa --model brown --brown-coeffs 8   # rational 8 参数
```

### 输入 / 输出

- **输入**：`<Data>/config.xml` + `<Data>/<case>/Original/`（视频或图片或角点 txt）
- **输出**（写到 `<Data>/<case>/Result/`）：
  - `intrinsic.xml` —— 内参 + 畸变（cv::FileStorage 格式）
  - `imgPoint_intrinsic.txt` / `wordPoint_intrinsic.txt` —— 图像/世界角点
  - `{k}.png` / `{k}_corner.png` / `{k}_d.png` —— 标定帧 / 角点可视化 / 去畸变图
  - `chessCoverImg.png` —— 棋盘格覆盖图
  - `detect_vis/{帧名}.jpg` ——（`--detect-vis` 时）逐帧检测中间结果：
    成功帧在原图上绘制角点与连线；失败帧（检测失败/亚像素异常/网格不一致/超出半径）
    在左上角用红字标注原因。先按最长边缩放到 1280（保持纵横比），再以 JPEG 质量 90
    压缩保存（1920×1536 的 PNG 约 2 MB → 1280×1024 的 JPG 约 150 KB）。

> ⚠️ 标定会**覆盖** `Result/` 下的同名文件。

---

## 四、测试用例

```bash
pytest tests/test_calib.py -v   # pytest
python tests/test_calib.py      # 或独立运行（无 pytest 也可）
```

依赖 `data/` 的用例（无需外部环境）：

| 用例                           | 断言                                       |
| ---------------------------- | ---------------------------------------- |
| `test_config_parse`          | config.xml 字段解析正确                        |
| `test_object_points`         | 世界坐标 `x=j*40,y=i*40,z=0`，首点(0,0,0)、88 点 |
| `test_rgb2gray_fixedpoint`   | 定点灰度化与 `cv2.COLOR_RGB2GRAY` 逐像素一致       |
| `test_frame_filter_exact`    | 62 帧全量数据下保留集与参考完全一致（52 帧）                |
| `test_corners_vs_reference`  | 角点与参考 txt 平均差 < 0.01 px                  |
| `test_calibration_reference` | 用参考角点标定，dK=dD≈0（需 3.2.0 DLL）           |
| `test_undistort_reference`   | 用参考 K/D 复算，去畸变图逐像素一致                    |
| `test_xml_format`            | 用参考 K/D 回写，与 intrinsic.xml 逐字节一致       |
| `test_grid_consistency_*`   | 网格校验：正常帧不误杀、乱序/行交换被拦截                     |
| `test_reprojection_filter`  | 逐帧重投影剔除：乱序坏帧误差显著、被 MAD 判为离群                |

> 数据缺失时相关用例自动 **SKIP**，其余纯逻辑用例仍会执行。
> 完整 62 帧图集（`data/img/`）不在版本库内，放置后 `test_frame_filter_exact` 才会运行。

多相机模型相关用例（均为合成数据，无外部依赖）：
`test_config_optional_model_fields`、`test_brown_synthetic_roundtrip`、
`test_brown_reproj_error_consistency`、`test_outlier_filter_wrong_model_inflates_error`、
`test_telephoto_small_distortion`、`test_auto_model_selection`、
`test_brown_xml_roundtrip_and_model_tag`。

---

## 五、多相机模型支持（KB / Brown / 长焦）

同一套标定流程支持三种场景：KB 鱼眼、Brown 径向切向、长焦相机（小 FOV / 小畸变）。
核心是模型抽象层 **`calib/models.py`**，把「标定 / 投影 / 求外参 / 去畸变 / 写出」
五个模型相关操作统一到 `CameraModel` 接口：

```
CameraModel                 ← 统一接口
├── KBModel                 ← fisheye 4 参数 k1..k4；标定走 3.2 DLL 位级一致后端；
│                              XML 写历史 5 元素格式
└── BrownModel(n=5|8)       ← cv2.calibrateCamera；5 参数 k1 k2 p1 p2 k3，
                               或 8 参数 rational k1..k6 p1 p2（--brown-coeffs 8）
```

### 模型选择（config.xml `camera_model` 或 CLI `--model`）

| 取值      | 行为                                                    |
| ------- | ----------------------------------------------------- |
| `kb`    | 显式 KB；若 fisheye 求解失败（ill-conditioned / 不收敛）自动降级 Brown 并记录原因 |
| `brown` | 显式 Brown                                               |
| `auto`  | `initCameraMatrix2D` 粗估焦距 → FOV≥70° 选 KB、≤45° 选 Brown；中间带双标定按「RMS+参数惩罚」选优 |

auto 决策全程打印日志，例如（鱼眼参考数据）：

```
camera_model = auto：粗估 f0 = 688.0 px，水平 FOV ≈ 70.0°（45°~70° 中间带）→ KB/Brown 双标定比较
候选比较：kb rms=0.313560（score=0.313560），brown rms=2.918855（score=2.928855） → 选 kb
```

### 长焦相机专项

1. **稳健初值**：`cv2.initCameraMatrix2D`（Zhang 单应分解）→ 不带初值 →
   f0 ∈ [0.8W, 40W] 对数扫描，全部试算取最优（长焦 fx 可达数十 W，普通初值远不收敛）；
2. **KB 不收敛自动降级**：长焦小 FOV 下 `fisheye::calibrate` 常报 ill-conditioned，
   kb / auto 模式下自动转 Brown；
3. **小畸变防过拟合**：解出后若 k3 / 切向项在**图像对角归一化半径**处的贡献可忽略
   （<1% 径向项），自动以 `FIX_K3 + ZERO_TANGENT_DIST` 裁剪重标（裁剪后重投影
   不变差才接受）。长焦数据上 k3 常被解出 -0.05 量级的过拟合噪声，正是由此裁掉；
4. **姿态多样性警告**：各帧倾角标准差 <5° 或景深相对极差 <10% 时提示补充姿态/位移
   （长焦下焦距与畸变强耦合，姿态单一会导致解不稳）。

### 输出格式

- KB：`intrinsic.xml` 保持 5 元素 D 布局（4 系数 + 占位 0），无新增标签；
- Brown：D 写真实 5/8 个系数，追加 `<camera_model>brown</camera_model>` 标签；
  `parse_intrinsic_xml` 按标签返回对应长度系数，`--compare` 仅在同模型下对比内参。

---

## 六、稳健性增强（异常处理 + 错误检出拦截）

### 6.1 异常处理体系

自定义异常层级（`calib/exceptions.py`）：

```
CalibError                # 基类
├── ConfigError           # 配置缺失/字段非法/数值越界
├── DataError             # 数据缺失/格式错误/路径不存在/图像无法读取
└── CalibrationError      # 标定过程失败（有效帧不足 / fisheye::calibrate 失败 / NaN）
```

覆盖的内外部异常场景（逐环节 try/except + 明确报错，而非静默崩溃）：

| 环节     | 异常场景                                                         | 处理                                   |
| ------ | ------------------------------------------------------------ | ------------------------------------ |
| 配置     | config.xml 不存在、字段缺失、数值越界                                     | 抛 `ConfigError`，指明字段                       |
| 数据     | 视频/图片目录为空、图像无法 imread、角点 txt 头部/数据量不符                     | 抛 `DataError`，指明文件名与期望/实际值                 |
| 检测     | 单帧角点检测或亚像素细化抛异常                                             | 记录原因并**跳过该帧**，不中断整体                     |
| 标定     | 有效帧数不足、`fisheye::calibrate` 失败、结果含 NaN/inf                 | 抛 `CalibrationError`，含诊断信息                 |
| 后端     | `opencv_world320.dll` 缺失/加载失败                                   | 自动降级 cv2 后端；均不可用则明确报错                    |

### 6.2 「有效帧」错误检出拦截

即便一帧通过了「能检出 88 个角点 + 半径筛选」，仍可能是**错误检出**（假棋盘格、
点序错乱、局部遮挡导致的角点错位）。两类拦截：

**(1) 网格一致性校验** —— `check_grid_consistency(img_corners, bw, bh)`，**默认开启**。
模型无关（针孔/鱼眼通用），校验两件事：

- **行列单调性**：真实棋盘格角点在行内 x 单调、列内 y 单调（不折叠、不交叉）。
  统计显著反向差分次数，超过 `0.3 × max(bw,bh)` 判错乱；
- **间距变异系数**：相邻角点间距的 CV 过大（> 0.8）说明局部错位/大范围乱序。

对「打乱角点顺序」「两行交换」等错误分布能稳定拦截（见测试），对正常 52 帧零误杀。

**(2) 逐帧重投影离群剔除** —— `reprojection_outlier_filter(...)`，**默认关闭**
（`--reproj-filter` 开启）。先用标定结果求每帧外参、算每帧重投影 RMS，再用
MAD（中位数绝对偏差）识别离群帧：

```
thr = median + mad_k × 1.4826 × MAD(errs)      # mad_k=8（放宽，只抓极端离群）
```

剔除比例受 `max_ratio=0.3` 上限约束、保留帧数不得低于 `min_keep=3`，避免把整体噪声
略高的正常帧误删。它针对的是「点序错乱 / 局部角点错位」这类**错误检出**帧——这类帧
误差极高，能被稳定识别；而**整体平移**的棋盘仍是合法标定视角，会被外参吸收、误差
不放大，故**不**应（也**不会**）被误判（这正是测试用例明确验证的边界）。

> 默认 `reproj_filter=False` 保持保守的默认行为；开启后内参可能有轻微漂移
> （误杀边缘正常帧），请按需使用。

---

## 七、参考数据集验证结果（data/img，52 帧）

| 项目              | 结果                                                       |
| --------------- | -------------------------------------------------------- |
| 帧筛选             | **52/52** 与参考保留集一致（检测失败 10 帧，超出半径 0 帧）                      |
| 角点              | 与 `imgPoint_intrinsic.txt` 平均差 **0.0027 px**、最大 0.098 px |
| 内参              | 自检测角点标定 `dK ≤ 0.009 px`；用参考角点标定 **dK=dD=0**               |
| 去畸变图            | 用参考 K/D 复算，与 `*_d.png` **逐像素零差异**                         |
| `intrinsic.xml` | 格式逐字节一致（整数值写 `%d.`，换行阈值 72）                                |
