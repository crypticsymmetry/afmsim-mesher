"""
The command line: ``python -m afmsim_mesher <command> <request.json>``.

``result.json`` is written beside the request whatever happens -- ``ok`` and
what was made, or ``ok: false`` and the error -- so the requester never has
to parse this process's output.

The request format is versioned (``PROTOCOL``); a request for another
version is refused rather than misread.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

#: The request and result format. 2 (2026-10-01): the requester sends every
#: volume to mesh, the air round the machine and any airgap columns
#: included, names the structured regions, and gets measurements back
#: (counts, and each region's element quality in a .npz) to judge itself.
PROTOCOL = 2


class _Collect(logging.Handler):
    """The warnings logged while meshing, to hand back to the requester."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _mesh3d(req: dict, base: Path) -> dict:
    import numpy as np

    from .builder import GmshModelBuilder, Structured
    from .ir import ir_from_json, sizes_from_json

    if int(req.get("protocol", 1)) != PROTOCOL:
        raise ValueError(f"this mesher reads protocol {PROTOCOL} requests, "
                         f"not {req.get('protocol', 1)}")
    s = req.get("structured")
    builder = GmshModelBuilder(
        ir_from_json(req["ir"]), sizes=sizes_from_json(req.get("sizes", {})),
        structured=Structured(
            regions=list(s["regions"]), arc_radius_mm=float(s["arc_radius_mm"]),
            arrangement=str(s.get("arrangement", "Alternate")),
            prisms=bool(s.get("prisms", False))) if s else None,
        options=req.get("options") or {}, verbosity=int(req.get("verbosity", 0)))
    try:
        builder.build()
        out = builder.export(base / req.get("output", "sector.msh"))
        diag = builder.diagnostics()
        quality = base / req.get("quality_output", "quality.npz")
        np.savez_compressed(quality, **builder.element_gammas())
    finally:
        builder.finalize()
    return {"output": str(out), "quality": str(quality), "diagnostics": diag}


def _step(req: dict, base: Path) -> dict:
    from .ir import sector_from_json
    from .step import write_step

    sectors = [sector_from_json(s) for part in req["parts"] for s in part["sectors"]]
    out = base / req.get("output", "machine.step")
    return {"output": str(out), "count": write_step(sectors, out)}


def _gui(path: Path) -> int:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.open(str(path))
        gmsh.fltk.run()
    finally:
        gmsh.finalize()
    return 0


COMMANDS = {"mesh3d": _mesh3d, "step": _step}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in (*COMMANDS, "gui"):
        print(f"usage: python -m afmsim_mesher {{{'|'.join((*COMMANDS, 'gui'))}}} FILE",
              file=sys.stderr)
        return 2
    if argv[1] == "gui":
        return _gui(Path(argv[2]))
    req_path = Path(argv[2]).resolve()
    result_path = req_path.with_name("result.json")
    collect = _Collect()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    logging.getLogger().addHandler(collect)
    try:
        req = json.loads(req_path.read_text(encoding="utf-8"))
        result = {"ok": True, "protocol": PROTOCOL,
                  **COMMANDS[argv[1]](req, req_path.parent)}
    except Exception as exc:                                  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(limit=8)}
    result["warnings"] = collect.messages
    tmp = result_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(result_path)
    return 0 if result["ok"] else 1


def cli() -> None:
    """The ``afmsim-mesher`` console script."""
    raise SystemExit(main(sys.argv))


if __name__ == "__main__":
    cli()
