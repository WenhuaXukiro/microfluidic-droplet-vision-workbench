# Windows 迁移说明

> GitHub源码仓库不包含实验图片、标注和已训练权重。本文中的“含现有项目”指完整数据备份包；源码安装与使用入口见 `README.md`，安装脚本会另行下载官方分类预训练权重。

此包是 Windows 源码运行版，不是免安装 EXE。含现有项目、原图、处理图、标注、回收区及预训练权重；Mac 的虚拟环境不包含在内。

1. 从 https://www.python.org/downloads/windows/ 安装 64 位 Python 3.12 或以上，启用 Python Launcher 和 Add Python to PATH。本项目在Mac的Python 3.13上完成验证。
2. 将 ZIP 完整解压到可写目录，如 D:\DropletLab。不要在压缩包内直接运行。
3. 双击 01_Start_Windows.bat，浏览器打开后刷新一次。保留命令窗口；关闭窗口会停止服务。标注和前处理无需训练库。
4. 要训练或识别，联网运行 02_Install_Training_Windows.bat，安装完成后关闭并重新启动工作台。
5. 训练设备选“自动选择”。CPU 可以运行；NVIDIA GPU 需要兼容驱动及 CUDA 版 PyTorch。按 https://pytorch.org/get-started/locally/ 的 Windows / Pip / CUDA 选择器安装到本目录 .venv（将官方命令中的 pip 替换成 .venv\Scripts\python.exe -m pip）。03_Check_Environment_Windows.bat 显示 NVIDIA CUDA available: True 后可选 NVIDIA GPU。

## 数据

项目数据在 data/project.json，图片在 data/images，分类模型在 data/classifier_runs，检测模型在 data/runs，启用记录在 models/项目编号/active_model.json。迁移后项目ID保持不变。旧训练快照已保留，配置和报告可能包含原电脑绝对路径；在 Windows 从界面重新导出训练集或重新启动训练即可生成新路径。整个文件夹须一起移动和备份。

本包包含所有当前保存图片（包括回收区），请妥善保管。源图片文件夹不会被修改。程序监听 127.0.0.1，不对局域网开放。端口8765被占用时，先关闭此前启动的工作台。网页打不开时先看启动窗口是否报错，再刷新 http://127.0.0.1:8765/。

## 验证范围

包内保留此前 Windows 训练运行与依赖记录；最新一轮标注2重训在 Apple 芯片 Mac 上完成，已启用模型的验证准确率为97.21%（522/537）。本次打包在Mac检查文件完整性与解压后的工作台和模型识别，未在新的Windows电脑验证安装。安装依赖需联网；CUDA 的具体版本取决于 Windows 机器的显卡和驱动。
