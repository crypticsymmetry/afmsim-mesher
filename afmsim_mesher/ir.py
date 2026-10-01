"""
The plain data the mesher is given: annular sectors by region name, and an
element size per region.

This is all afmsim_mesher knows of a machine. The requester (afmsim) turns
its own geometry into this JSON and reads back the files the mesher writes;
the two share no code and run in separate processes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnnularSector:
    """A sector of a hollow cylinder, in mm and degrees: ``theta_start_deg``
    to ``theta_end_deg`` counter-clockwise from +x seen from +z, and
    ``z_start_mm`` to ``z_end_mm`` axially."""
    r_inner_mm: float
    r_outer_mm: float
    z_start_mm: float
    z_end_mm: float
    theta_start_deg: float
    theta_end_deg: float
    tag: str = ""

    def __post_init__(self) -> None:
        if (self.theta_end_deg <= self.theta_start_deg or self.z_end_mm <= self.z_start_mm
                or self.r_outer_mm <= self.r_inner_mm):
            raise ValueError(f"AnnularSector {self.tag!r} is empty or inverted")

    @property
    def arc_rad(self) -> float:
        return math.radians(self.theta_end_deg - self.theta_start_deg)


@dataclass
class SectorIR:
    """The named volumes of one periodic sector (or a whole machine)."""
    volumes: dict[str, list[AnnularSector]] = field(default_factory=dict)
    sector_angle_deg: float = 360.0
    r_inner_mm: float = 0.0
    r_outer_mm: float = 0.0
    total_axial_mm: float = 0.0


@dataclass
class Sizes:
    """The element size each region is meshed at, in mm, already chosen by
    the requester; a region not listed gets ``global_max_mm``."""
    table: dict[str, float] = field(default_factory=dict)
    global_min_mm: float = 0.1
    global_max_mm: float = 10.0

    def size_for(self, region_name: str) -> float:
        return float(self.table.get(region_name, self.global_max_mm))


def sector_from_json(d: dict[str, Any]) -> AnnularSector:
    return AnnularSector(float(d["r_inner_mm"]), float(d["r_outer_mm"]),
                         float(d["z_start_mm"]), float(d["z_end_mm"]),
                         float(d["theta_start_deg"]), float(d["theta_end_deg"]),
                         str(d.get("tag", "")))


def ir_from_json(d: dict[str, Any]) -> SectorIR:
    return SectorIR(
        volumes={name: [sector_from_json(s) for s in secs]
                 for name, secs in d["volumes"].items()},
        sector_angle_deg=float(d.get("sector_angle_deg", 360.0)),
        r_inner_mm=float(d.get("r_inner_mm", 0.0)),
        r_outer_mm=float(d.get("r_outer_mm", 0.0)),
        total_axial_mm=float(d.get("total_axial_mm", 0.0)))


def sizes_from_json(d: dict[str, Any]) -> Sizes:
    return Sizes(table={k: float(v) for k, v in d.get("table", {}).items()},
                 global_min_mm=float(d.get("global_min_mm", 0.1)),
                 global_max_mm=float(d.get("global_max_mm", 10.0)))
