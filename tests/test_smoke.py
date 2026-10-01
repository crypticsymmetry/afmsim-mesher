"""The example request meshes, and a bad one fails with a result.json that says why."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _run(command, request_path):
    return subprocess.run([sys.executable, "-m", "afmsim_mesher", command, str(request_path)],
                          cwd=ROOT, capture_output=True, text=True)


def test_the_example_meshes(tmp_path):
    req = tmp_path / "request.json"
    shutil.copyfile(ROOT / "examples" / "ring" / "request.json", req)
    assert _run("mesh3d", req).returncode == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] and result["protocol"] == 2 and (tmp_path / "ring.msh").is_file()
    d = result["diagnostics"]
    assert {"Body", "Layer", "Air", "SectorFace_0", "SectorFace_1"} <= set(d["physical_groups"])
    assert d["structured"] == {"columns": 1, "refused": 0}
    with np.load(result["quality"]) as q:
        assert set(q.files) == {"Body", "Layer", "Air"}
        assert len(q["Layer"]) == d["region_element_counts"]["Layer"]


def test_a_bad_request_says_why(tmp_path):
    req = tmp_path / "request.json"
    req.write_text("{}")
    assert _run("mesh3d", req).returncode == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert not result["ok"] and "protocol 2" in result["error"]
