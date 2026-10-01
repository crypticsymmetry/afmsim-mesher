# afmsim_mesher

The Gmsh-based mesher of the AFPM simulator, as a program of its own.

Copyright (C) 2026 ccbb1.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See `COPYING` for the full licence.

It uses [Gmsh](https://gmsh.info) (GPL-2.0-or-later) and NumPy (BSD-3-Clause).

Published at https://github.com/crypticsymmetry/afmsim-mesher.

## What it does

It reads a JSON request and writes files. It imports nothing of the simulator,
and the simulator imports nothing of it; they talk only through those files.

```
python -m afmsim_mesher mesh3d request.json   # -> the .msh named in the request, and result.json
python -m afmsim_mesher step   request.json   # -> a STEP file of named annular solids, and result.json
python -m afmsim_mesher gui    mesh.msh       # -> opens a mesh in the Gmsh window
```

A `mesh3d` request (protocol 2; a request in another format is refused):

```json
{
  "protocol": 2,
  "ir": {"volumes": {"Body": [{"r_inner_mm": 20, "r_outer_mm": 30,
          "z_start_mm": 0, "z_end_mm": 3, "theta_start_deg": 0, "theta_end_deg": 20,
          "tag": "Body"}],
          "Layer": [{"r_inner_mm": 20, "r_outer_mm": 30,
          "z_start_mm": 3, "z_end_mm": 4, "theta_start_deg": 0, "theta_end_deg": 20,
          "tag": "Layer"}]},
         "sector_angle_deg": 20, "r_inner_mm": 20, "r_outer_mm": 30,
         "total_axial_mm": 4},
  "sizes": {"table": {"Body": 1.0, "Layer": 0.5}, "global_min_mm": 0.1, "global_max_mm": 10},
  "structured": {"regions": ["Layer"], "arc_radius_mm": 30,
                 "arrangement": "Alternate", "prisms": false},
  "options": {"algorithm3d": 1, "optimize_netgen": false, "threads": 0},
  "output": "sector.msh",
  "quality_output": "quality.npz"
}
```

Every volume in `ir` is built, in the order given, and the whole set is
fragmented into a conforming mesh: where volumes overlap, the smallest wins.
Each region is meshed at its size in `sizes`. The regions named in
`structured` (optional) are meshed transfinite, each of their volumes a
topological hexahedron. A sector (`sector_angle_deg` under 360) gets periodic
faces at its two cut planes, as physical groups `SectorFace_0` and
`SectorFace_1`, and the rest of the outside is `OuterBoundary`.

`result.json`, beside the request, carries `ok`, `protocol`, and the mesh's
measurements -- `nodes`, `elements_3d`, `physical_groups`,
`region_volume_counts`, `region_element_counts`, `region_target_size_mm`,
`sector_faces`, and `structured` (volumes made transfinite, and refused) -- with
the warnings logged; or `ok: false` and the `error`. `quality.npz` holds each
region's element shape quality (Gmsh's gamma), one array per region.

A `step` request is `{"parts": [{"name": ..., "sectors": [...]}], "output": "machine.step"}`;
each sector becomes one solid, in millimetres, in the order given, and `result.json`
lists their count.

## The Windows executable

The Axial Flux Simulator ships `afmsim-mesher.exe`, built from this source by
`packaging/build_exe.py` (Nuitka, MSVC 2022, Python 3.12) from
`packaging/mesher_entry.py`.
