"""
Annular solids to a STEP file, in millimetres -- the unit STEP carries, and
one in which OpenCASCADE's tolerances close a full revolution cleanly (the
solver's metres, scaled after, left full rings 0.9 % large). One solid per
sector, in the order given; naming them is the requester's business.
"""
from __future__ import annotations

import math
from pathlib import Path

import gmsh

from .ir import AnnularSector


def _sector_mm(sec: AnnularSector) -> int:
    occ = gmsh.model.occ
    ri, ro, z0, z1 = sec.r_inner_mm, sec.r_outer_mm, sec.z_start_mm, sec.z_end_mm
    pts = [occ.addPoint(ri, 0, z0), occ.addPoint(ro, 0, z0),
           occ.addPoint(ro, 0, z1), occ.addPoint(ri, 0, z1)]
    lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    vol = next(t for d, t in occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, sec.arc_rad)
               if d == 3)
    if sec.theta_start_deg:
        occ.rotate([(3, vol)], 0, 0, 0, 0, 0, 1, math.radians(sec.theta_start_deg))
    return vol


def write_step(sectors: list[AnnularSector], path: str | Path) -> int:
    """Write one solid per sector to ``path``; return how many were made."""
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("afpm_step")
    try:
        for sec in sectors:
            _sector_mm(sec)
        gmsh.model.occ.synchronize()
        n = len(gmsh.model.getEntities(3))
        if n != len(sectors):
            raise RuntimeError(f"{n} solids built for {len(sectors)} sectors")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(p))
    finally:
        gmsh.finalize()
    return n
