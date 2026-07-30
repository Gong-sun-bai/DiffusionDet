#!/usr/bin/env python3
"""Build the vendored Detectron2 C++ extension without CUDA sources.

DiffusionDet's SAR baseline uses torchvision's CUDA implementations for regular
ROIAlign and NMS.  The vendored Detectron2 extension is still required for
module imports and the optimized COCO evaluator, but it does not need to be
compiled against the workstation's CUDA toolkit.  Keeping this build CPU-only
avoids mixing the system CUDA 11.4 compiler with the cu128 PyTorch wheel.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CSRC_ROOT = REPOSITORY_ROOT / "detectron2" / "layers" / "csrc"
MAIN_SOURCE = CSRC_ROOT / "vision.cpp"


def extension_sources() -> list[str]:
    sources = [MAIN_SOURCE]
    sources.extend(
        path
        for path in sorted(CSRC_ROOT.rglob("*.cpp"))
        if path != MAIN_SOURCE
    )
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Detectron2 C++ 源码缺失：{missing}")
    return [str(path) for path in sources]


setup(
    name="diffusiondet-local-detectron2-extension",
    version="0.6",
    ext_modules=[
        CppExtension(
            name="detectron2._C",
            sources=extension_sources(),
            include_dirs=[str(CSRC_ROOT)],
            extra_compile_args={"cxx": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
