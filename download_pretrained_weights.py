"""Prepare official classification weights omitted from the source repository."""
import argparse
import hashlib
import os
from pathlib import Path
import ssl
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
WEIGHTS = {
    "convnext_tiny": ("convnext_tiny-983f1562.pth", "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"),
    "resnet18": ("resnet18-f37072fd.pth", "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(model, directory):
    filename, expected = WEIGHTS[model]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.is_file() and digest(destination) == expected:
        print(f"已准备并校验：{filename}", flush=True)
        return destination
    url = f"https://download.pytorch.org/models/{filename}"
    descriptor, temporary_name = tempfile.mkstemp(prefix=filename+".", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    try:
        print(f"从 PyTorch 官方下载：{filename}", flush=True)
        context = ssl.create_default_context()
        try:
            import certifi
        except ImportError:
            pass
        else:
            context.load_verify_locations(cafile=certifi.where())
        with os.fdopen(descriptor, "wb") as target:
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(block)
        if digest(temporary) != expected:
            raise ValueError(f"预训练权重校验失败，未替换原文件：{filename}")
        temporary.replace(destination)
        print(f"下载及 SHA-256 校验完成：{filename}", flush=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="下载并校验 ConvNeXt-Tiny / ResNet18 官方预训练权重")
    parser.add_argument("--model", choices=["all", *WEIGHTS], default="all")
    parser.add_argument("--directory", type=Path, default=ROOT/"weights")
    args = parser.parse_args()
    for model in WEIGHTS if args.model == "all" else [args.model]:
        prepare(model, args.directory)


if __name__ == "__main__":
    main()
