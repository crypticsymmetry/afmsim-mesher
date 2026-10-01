"""
Build ``afmsim-mesher.exe`` (Windows x64) as the Axial Flux Simulator ships it.

    pip install gmsh numpy nuitka zstandard ordered-set
    python packaging/build_exe.py

Run it from an "x64 Native Tools Command Prompt for VS" (MSVC 2022). The
result is ``build/mesher_entry.dist``, a folder with ``afmsim-mesher.exe``,
the Python runtime, NumPy and the Gmsh library (under ``lib/``). The shipped
build used Python 3.12, Nuitka 4.2.2, gmsh 4.15.2 and NumPy 2.5.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

subprocess.run([
    sys.executable, "-m", "nuitka", "--standalone", "--msvc=latest",
    "--assume-yes-for-downloads", "--python-flag=no_docstrings",
    "--product-name=Axial Flux Simulator", "--file-version=2.0.0.0",
    "--product-version=2.0.0.0", "--company-name=crypticsymmetry",
    f"--output-dir={ROOT / 'build'}", "--output-filename=afmsim-mesher.exe",
    "--windows-console-mode=disable", "--include-package=afmsim_mesher",
    "--file-description=afmsim-mesher (GPL-2.0-or-later)",
    str(HERE / "mesher_entry.py"),
], cwd=ROOT, check=True)
