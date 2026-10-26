# DropletLab · Apple 芯片 Mac 使用说明

> GitHub源码仓库不包含实验图片、标注和模型。下面“迁移内容”记录的是2026-10-24完整数据备份包；首次克隆请按 `README.md` 安装并导入自己的图片。安装脚本会另行下载官方分类预训练权重。

## 首次使用

1. 使用 Apple 芯片 Mac（M1/M2/M3/M4 等），macOS 14 或以上。
2. 从 https://www.python.org/downloads/macos/ 安装 Python 3.12 或 3.13 的 macOS universal2 版本，使用原生 ARM 模式。
3. 完整解压 ZIP，将整个 DropletLab 文件夹放到可读写位置。
4. 联网双击 `安装训练环境.command`，等待完成。
5. 双击 `启动软件.command`，浏览器打开 http://127.0.0.1:8765/ 。保留启动终端窗口。
6. 可用 `检查训练环境.command` 检查 Apple GPU 是否可用；工作台训练设备选“自动选择”或“Apple 芯片 GPU”。

如果双击脚本无法打开，在终端输入 `cd `（末尾留空格），把解压后的 DropletLab 文件夹拖到终端并回车，然后运行：

```sh
zsh 安装训练环境.command
zsh 检查训练环境.command
zsh 启动软件.command
```

环境安装需要网络和足够磁盘空间。虚拟环境与依赖下载缓存未打包；Apple 芯片使用 PyTorch MPS。当前项目已在 Apple 芯片 Mac、Python 3.13 环境完成安装、模型训练和验证；其他电脑仍需按上述步骤安装环境。

## 迁移内容

- 最新浅色工作台，包含回收区跨页全选、批量恢复、永久删除及二级设置。
- 两个项目的当前图片、标注、划分、前处理规则，包括回收区。
- 标注1启用的 ConvNeXt-Tiny，保存的验证准确率78.44%。
- 标注2最新重训的 ConvNeXt-Tiny：训练2,150张、验证537张，验证正确522张，准确率97.21%、宏平均F1为95.22%；最佳第5轮，共训练11轮，已启用，旧模型保留。
- 当前共4,657张图片：标注1有1,942张，标注2有2,715张；未审核图片保留，但未参与训练和验证。
- 历史训练模型、预训练权重、导出快照、备份、逐图验证记录与报告。

标注2按同一视频的帧顺序划分，末尾约20%作为验证；该验证集参与最佳轮次选择，不是独立视频测试集。未审核图片未参与训练。

## 路径与完整性

当前模型引用使用项目内相对路径，可随整个文件夹移动。历史报告、训练配置和旧导出快照可能保留原电脑绝对路径，仅作为历史记录；新训练应从工作台重新导出数据。

`package-manifest.json` 列出包内文件的大小与SHA-256；ZIP旁的 `.sha256` 文件用于核验压缩包。`verify_package.py` 可校验解压后的全部文件：

```sh
python3 verify_package.py
```

完整文件夹应一起移动和备份。源码包不是独立Mac应用，首次需安装Python环境。不要在原Windows电脑直接用Mac安装脚本覆盖Windows环境。

当前打包时的Mac依赖版本见 `training-environment.macos.current.lock.txt`；`training-environment.lock.txt` 和 `training-environment.windows.lock.txt` 是保留的历史环境记录。安装入口仍为 `requirements-macos.txt`。
