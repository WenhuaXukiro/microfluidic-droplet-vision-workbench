# 微流控液滴图像识别训练工作台 · DropletLab

[简体中文](README.md) | [English](README.en.md)

用于微流控显微图像的本地工作台，将图片导入、液滴标注、批量前处理、训练集划分、模型训练、验证和预测串成一个流程。界面在浏览器中运行，图片与标注保存在本机。

支持 **ConvNeXt-Tiny / ResNet18 单液滴分类**，也支持 **YOLO 多液滴目标检测**。适合细胞与磁珠包封实验中的液滴状态识别和训练数据整理。

## 主要功能

- **图片与项目管理**：按文件夹导入 PNG/JPG；独立管理实验项目；查看审核进度；回收区预览、恢复和永久删除。
- **中英文界面**：右上角选择“中文 / English”，即时切换并在当前浏览器记住选择；图片名、项目名和模型类别编号保持原样。
- **五类液滴标注**：手动画框；单液滴 ROI 可整图快速分类；数字键 1–5 分类，方向键切图，修改自动保存。
- **批量前处理**：按相对位置或固定像素裁剪，调整灰度、亮度、对比度和输出尺寸；保存规则后复用于下一批图片。
- **识别前增强**：原样、温和局部对比度、光照校正、保边去噪与轻锐化；增强方案写入模型，训练、验证、预测使用相同处理。
- **数据集划分与导出**：按实验批次划分训练/验证集，保留测试集；也支持独立图片按原图随机划分；导出 YOLO 数据集或五类文件夹。
- **模型训练**：自动判断单液滴分类或目标检测；支持 CPU、Apple 芯片 GPU 和 NVIDIA GPU；显示训练日志。
- **模型验证与使用**：分类训练输出准确率、宏平均 F1、类别平衡准确率、混淆矩阵、逐图结果和错误清单；预测可接受为待审核草稿。

## 五类标签

类别编号固定，所有项目、导出文件和模型使用相同顺序。

| 编号 | 类别 | 定义 | 快捷键 |
|---|---|---|---|
| 0 | `1cell 1bead` | 恰好 1 个细胞和 1 个磁珠 | 1 |
| 1 | `only cell` | 有细胞、无磁珠；允许多个细胞 | 2 |
| 2 | `only beads` | 有磁珠、无细胞；允许多个磁珠 | 3 |
| 3 | `multiplet` | 细胞和磁珠都有，且至少一种数量超过 1 | 4 |
| 4 | `empty` | 无细胞、无磁珠的空液滴 | 5 |

## 快速开始

建议使用 Python 3.12 或 3.13。浏览器可使用 Chrome、Edge 或 Safari。基础图片标注与数据导出使用 Python 标准库即可启动；模型训练、识别和增强预览需要额外安装依赖。

### Apple 芯片 Mac

1. 克隆本仓库，或下载源码 ZIP 并完整解压。
2. 仅查看和标注：双击 `启动软件.command`。
3. 训练与模型识别：先双击 `安装训练环境.command`，安装完成后重新启动工作台。
4. 可双击 `检查训练环境.command` 检查依赖与 Apple GPU。

也可在终端运行：

```sh
git clone https://github.com/WenhuaXukiro/microfluidic-droplet-vision-workbench.git
cd microfluidic-droplet-vision-workbench
zsh 安装训练环境.command
zsh 启动软件.command
```

Mac 安装脚本要求原生 Apple 芯片 Python 和 macOS 14 或以上。其他环境可按下面的手动方式安装，使用 CPU 运行。

### Windows

1. 安装 64 位 Python 3.12 或 3.13，启用 Python Launcher 和 PATH。
2. 完整解压源码到可写目录。
3. 双击 `02_Install_Training_Windows.bat` 安装训练环境。
4. 双击 `01_Start_Windows.bat` 启动，`03_Check_Environment_Windows.bat` 可检查环境。

CPU 可训练与识别。使用 NVIDIA GPU 时，按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 为当前显卡和驱动安装 CUDA 版 PyTorch，并安装到本项目 `.venv` 中。详细操作见 [Windows_README.md](Windows_README.md)。

### 手动启动

只使用基础标注功能：

```sh
python3 server.py
```

需要训练与识别时，以 macOS 为例：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_pretrained_weights.py
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python server.py
```

Windows 将上面的 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`；启用 MPS 的环境变量仅用于 Mac。安装脚本会准备分类所需的官方预训练权重。YOLO 目标检测首次训练会按 Ultralytics 流程获取 `yolo11n.pt`。依赖安装和首次权重下载需要联网。

工作台地址为 **http://127.0.0.1:8765/**。请保留启动窗口；端口被占用时先关闭之前的工作台。服务仅监听本机回环地址。

## 推荐工作流程

1. **建立项目并导入图片**：填写实验批次，保留不同实验或视频的来源信息。
2. **生成单液滴处理图**：框选包含完整液滴的区域，预览后批量生成处理副本；原图含多个液滴时也可直接做检测标注。
3. **标注并审核**：每张单液滴图设置一个类别；多液滴图逐个框选。看不清的图片先不审核。
4. **划分数据**：优先将独立实验或视频整体分到训练、验证和测试集合。当前自动划分提供“按实验批次”和“按原图随机”两种方式；同一视频连续帧不应随机拆分。
5. **选择模型并训练**：单液滴选择 ConvNeXt-Tiny 或 ResNet18；多液滴原图选择目标检测。只有当前项目已审核、未移除的图片参与训练与验证。
6. **查看验证结果**：结合宏平均 F1、各类召回率和错误图片判断效果，再用独立视频或新实验测试泛化能力。
7. **使用模型识别**：选择匹配项目的一张处理图，点击“模型识别”；接受的预测仍需人工复核。

详细标注操作、快捷键和前处理规则见 [使用说明.md](使用说明.md)。

## 训练与验证方式

单液滴分类保留完整 ROI 和纵横比，等比例缩放并补边；默认输入长边为 512 像素，不做随机裁剪。训练使用轻微亮度/对比度变化、水平/垂直翻转和按类别频数设置的损失权重。

前两轮适配分类头；ConvNeXt-Tiny 随后固定前两个特征阶段，微调后两个阶段及分类头。每次训练保存具体参数、随机种子、数据审计与逐轮结果。以验证集宏平均 F1 选择最佳轮次，同分时比较准确率，并支持提前停止。

分类候选与现用模型的比较要求验证图片及标签完全一致；准确率和宏平均 F1 都提高，且五类都有正确预测时，才自动启用。没有匹配旧模型验证记录时使用多数类基线。旧权重保留。分类结果在界面显示为覆盖整张 ROI 的框，这个框不表示模型学习了液滴边界。

### 已完成的一次重训记录

2026-10-22，ConvNeXt-Tiny 在同一视频按时间留出末尾图片验证的实验中：训练 2,150 张，验证 537 张，正确 522 张。

| 指标 | 旧模型在相同验证图上重测 | 重训模型 |
|---|---:|---:|
| 准确率 | 93.48% | 97.21% |
| 宏平均 F1 | 86.72% | 95.22% |
| 类别平衡准确率 | 83.25% | 93.96% |

这些分数来自参与最佳轮次选择的同视频验证集，不代表新视频或新实验的识别率，也不是独立测试结果。源码仓库不附带该实验的图片、标注或权重。

## 数据与文件结构

```text
server.py                         本地服务与项目、导出、训练接口
static/index.html                 浏览器工作台
worker.py                         训练、预测与增强任务
roi_classifier.py                 分类网络、统一图像输入和预测
train_roi_classifier.py           分类训练与验证
image_preprocessing.py            统一增强与预览
evaluate_model.py                 数据审计与检测模型评估
evaluate_preprocessing.py         增强方案对照评估
download_pretrained_weights.py    获取并校验分类预训练权重
requirements*.txt                 依赖与安装入口
test_*.py / test_*.cjs            已有回归检查
data/                             运行后生成：图片、标注、导出和训练记录
models/                           运行后生成：按项目保存启用模型信息
weights/                          下载的官方分类预训练权重
```

`data/project.json` 保存标注和项目配置，`data/images/` 保存工作台图片，`data/exports/` 保存训练快照。分类模型位于 `data/classifier_runs/`，检测模型位于 `data/runs/`。`models/<项目编号>/active_model.json` 通过项目内相对路径引用启用模型。

**GitHub 仓库提供源码、说明和检查脚本。** 实验图片、人工标注、模型权重、训练记录、完整压缩包和虚拟环境均由 `.gitignore` 排除。首次克隆从空工作台开始；已有完整项目包的用户可先备份，再将对应的 `data/`、`models/` 和 `weights/` 一起迁移到项目目录。请保持目录结构，并使用匹配的图片与标注。

从 GitHub 克隆的源码与“含实验数据的完整备份包”内容不同；包内迁移说明中的历史图片数量与模型记录不表示新克隆仓库自带这些数据。

## 检查与开发

不安装训练依赖也可运行基础回归检查：

```sh
python3 -m unittest test_auto_split test_classified test_recycle test_workflow
```

安装训练依赖后可运行全部 Python 检查：

```sh
.venv/bin/python -m unittest discover -p 'test_*.py'
```

前端检查需要 Node.js，可分别运行 `node test_preprocess.cjs`、`node test_quick_label.cjs` 和 `node test_trash.cjs`。模型训练不要求 Node.js。

运行 `node test_i18n.cjs` 可检查中英文词条、动态占位符、语言偏好保存和错误提示翻译。

## 当前范围

工作台输出液滴类别或目标检测框，不输出细胞/磁珠的精确计数或液滴轮廓。界面没有独立测试集评估入口。增强方案是否有效需要通过配套训练和验证判断，不能仅凭图像看起来更清楚来认定准确率提高。

主要训练组件来自 PyTorch、TorchVision 和 Ultralytics，使用和分发这些组件与权重时适用其各自许可。
