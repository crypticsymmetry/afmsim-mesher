"""
The 3-D sector mesher: annular sectors -> Gmsh OCC solids -> boolean
fragment -> physical groups -> sized, periodic tetrahedral mesh -> .msh.

Part of afmsim_mesher (GPL-2.0-or-later, see COPYING). It runs as its own
process (``python -m afmsim_mesher mesh3d request.json``); the requester
sends plain JSON (:mod:`afmsim_mesher.ir`) and reads back the files.

It makes no choices of its own about the machine. The volumes it is given
are the volumes it meshes, each at the size it is given; the regions to mesh
structured, and how, are named in the request; and what it reports back is
measurement -- counts, and each element's shape quality -- for the requester
to judge.

Usage, in this process::

    from afmsim_mesher.builder import GmshModelBuilder
    builder = GmshModelBuilder(ir, sizes=sizes)
    builder.build()
    builder.export("sector.msh")
    builder.finalize()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from .ir import AnnularSector, Sizes
from .ir import SectorIR as GeometryIR

logger = logging.getLogger(__name__)


#: Gmsh's options, as every caller gets them unless it passes its own: the
#: 3-D algorithm (1 Delaunay, 10 HXT), whether the Netgen optimiser runs
#: after it, and the thread count (0: Gmsh's default).
MESH_OPTIONS: dict = {"algorithm3d": 1, "optimize_netgen": False, "threads": 0}


@dataclass
class Structured:
    """Regions to mesh transfinite (structured), each volume of which must be
    a topological hexahedron: ``regions`` by name; arcs counted at
    ``arc_radius_mm``; the across-the-layer faces triangulated in
    ``arrangement`` (a Gmsh transfinite arrangement) with corners in one
    (angle, radius) order, or recombined into prisms with ``prisms``."""
    regions: list[str] = field(default_factory=list)
    arc_radius_mm: float = 0.0
    arrangement: str = "Alternate"
    prisms: bool = False


class GmshModelBuilder:
    """
    End-to-end: IR -> Gmsh OCC geometry -> meshed sector -> .msh export.

    Lifecycle::

        builder = GmshModelBuilder(ir)
        builder.build()               # geometry + fragment + groups + mesh
        builder.export("out.msh")
        builder.finalize()             # release Gmsh
    """

    def __init__(
        self,
        ir: GeometryIR,
        *,
        sizes: Sizes | None = None,
        structured: Structured | None = None,
        options: dict | None = None,
        verbosity: int = 0,
    ) -> None:
        self.ir = ir
        self.policy = sizes or Sizes()
        self.structured = structured if structured and structured.regions else None
        self.options = {**MESH_OPTIONS, **(options or {})}
        self.verbosity = verbosity
        #: Structured volumes the transfinite mesh was applied to, and refused.
        self._structured_gap = {"columns": 0, "refused": 0}

        # Maps region name → list of OCC volume dim-tags after fragmentation
        self._region_volumes: dict[str, list[tuple[int, int]]] = {}
        # Maps region name → original annular sectors for robust post-fragment remap
        self._region_sectors: dict[str, list[AnnularSector]] = {}
        # Maps region name → physical group tag
        self._physical_groups: dict[str, int] = {}
        # Sector boundary face tags (populated before meshing)
        self._face_at_0: list[int] = []
        self._face_at_sector: list[int] = []
        # Target element size actually applied per region (mm)
        self._region_target_size_mm: dict[str, float] = {}
        # Track whether gmsh is initialised by us
        self._initialised = False

    # ── public API ───────────────────────────────────────────

    def build(self) -> None:
        """Full pipeline: init → geometry → fragment → groups → mesh."""
        self._init_gmsh()
        self._build_geometry()
        self._boolean_fragment()
        self._assign_physical_groups()
        self._apply_mesh_policy()
        self._set_periodic_constraints()
        self._generate_mesh()

    def export(self, path: str | Path, *, fmt: str = "msh") -> Path:
        """Write the mesh to *path*.  Supports 'msh', 'vtk', 'stl'."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(p))
        logger.info("Mesh exported to %s", p)
        return p

    def finalize(self) -> None:
        """Release Gmsh resources."""
        if self._initialised:
            gmsh.finalize()
            self._initialised = False

    @property
    def physical_groups(self) -> dict[str, int]:
        return dict(self._physical_groups)

    @property
    def region_volume_counts(self) -> dict[str, int]:
        """Return the number of post-fragment OCC volumes per named region."""
        return {name: len(tags) for name, tags in self._region_volumes.items()}

    # ── internal helpers ─────────────────────────────────────

    def _init_gmsh(self) -> None:
        import threading
        # Gmsh registers a SIGINT handler which fails in non-main threads.
        # Disable it when not on the main thread.
        on_main = threading.current_thread() is threading.main_thread()
        gmsh.initialize(interruptible=on_main)
        self._initialised = True
        gmsh.option.setNumber("General.Verbosity", self.verbosity)
        gmsh.model.add("afpm_sector")

    def _build_annular_sector(self, sec: AnnularSector) -> int:
        """
        Create one AnnularSector volume in Gmsh OCC via revolve.

        Returns the volume tag (dim=3).
        """
        occ = gmsh.model.occ
        th_rad = sec.arc_rad

        # Convert mm → metres so the mesh is in SI units
        MM2M = 1e-3
        r_in = sec.r_inner_mm * MM2M
        r_out = sec.r_outer_mm * MM2M
        z0 = sec.z_start_mm * MM2M
        z1 = sec.z_end_mm * MM2M

        # Cross-section rectangle in the x-z plane (y = 0)
        p1 = occ.addPoint(r_in,  0, z0)
        p2 = occ.addPoint(r_out, 0, z0)
        p3 = occ.addPoint(r_out, 0, z1)
        p4 = occ.addPoint(r_in,  0, z1)

        l1 = occ.addLine(p1, p2)
        l2 = occ.addLine(p2, p3)
        l3 = occ.addLine(p3, p4)
        l4 = occ.addLine(p4, p1)

        cl = occ.addCurveLoop([l1, l2, l3, l4])
        surf = occ.addPlaneSurface([cl])

        # Revolve around z-axis by the sector arc
        revolved = occ.revolve(
            [(2, surf)],
            0, 0, 0,    # point on axis
            0, 0, 1,    # axis direction (z)
            th_rad,
        )

        # Extract the volume tag from the revolved entities
        vol_tag = None
        for dim, tag in revolved:
            if dim == 3:
                vol_tag = tag
                break

        if vol_tag is None:
            raise RuntimeError(f"Revolve produced no volume for {sec.tag}")

        # Now rotate the entire volume so it starts at theta_start
        if sec.theta_start_deg != 0.0:
            occ.rotate(
                [(3, vol_tag)],
                0, 0, 0,
                0, 0, 1,
                math.radians(sec.theta_start_deg),
            )

        return vol_tag

    def _build_geometry(self) -> None:
        """Translate every IR volume into Gmsh OCC entities, in the IR's order."""
        for name, sectors in self.ir.volumes.items():
            tags: list[tuple[int, int]] = []
            for sec in sectors:
                vtag = self._build_annular_sector(sec)
                tags.append((3, vtag))
            self._region_volumes[name] = tags
            self._region_sectors[name] = list(sectors)

        gmsh.model.occ.synchronize()
        logger.info("Built %d named regions in Gmsh OCC",
                     len(self._region_volumes))

    def _boolean_fragment(self) -> None:
        """
        Fragment all volumes for conforming mesh interfaces.

        After fragmentation the original tags are invalidated;
        we remap by centroid matching.
        """
        all_vols: list[tuple[int, int]] = []
        input_region_names: list[str] = []
        input_specificities: list[float] = []
        for name, tags in self._region_volumes.items():
            for dim, tag in tags:
                all_vols.append((dim, tag))
                input_region_names.append(name)
                bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                input_specificities.append(
                    max(bbox[3] - bbox[0], 1e-12)
                    * max(bbox[4] - bbox[1], 1e-12)
                    * max(bbox[5] - bbox[2], 1e-12)
                )

        if len(all_vols) < 2:
            gmsh.model.occ.synchronize()
            return

        # Compute bounding boxes and centroids before fragmentation for remapping
        gmsh.model.occ.synchronize()
        pre_centroids: dict[str, list[np.ndarray]] = {}
        pre_bboxes: dict[str, list[tuple[float, float, float, float, float, float]]] = {}
        for name, tags in self._region_volumes.items():
            cents: list[np.ndarray] = []
            bboxes: list[tuple[float, float, float, float, float, float]] = []
            for dim, tag in tags:
                bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                cx = (bbox[0] + bbox[3]) / 2.0
                cy = (bbox[1] + bbox[4]) / 2.0
                cz = (bbox[2] + bbox[5]) / 2.0
                cents.append(np.array([cx, cy, cz]))
                bboxes.append(tuple(float(v) for v in bbox))
            pre_centroids[name] = cents
            pre_bboxes[name] = bboxes

        # Fragment all volumes against each other
        obj = [all_vols[0]]
        tool = all_vols[1:]
        _, out_dimtags_map = gmsh.model.occ.fragment(obj, tool)
        gmsh.model.occ.synchronize()

        parent_candidates: dict[int, list[tuple[float, str]]] = {}
        for idx, mapped in enumerate(out_dimtags_map):
            if idx >= len(input_region_names):
                break
            for dim, tag in mapped:
                if dim == 3:
                    parent_candidates.setdefault(tag, []).append(
                        (input_specificities[idx], input_region_names[idx])
                    )

        # Get all resulting volumes
        result_vols = gmsh.model.getEntities(3)

        # Compute post-fragmentation centroids
        post_centroids: list[tuple[int, int, np.ndarray]] = []
        for dim, tag in result_vols:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            cx = (bbox[0] + bbox[3]) / 2.0
            cy = (bbox[1] + bbox[4]) / 2.0
            cz = (bbox[2] + bbox[5]) / 2.0
            post_centroids.append((dim, tag, np.array([cx, cy, cz])))

        # Remap: assign each post-fragment volume to the pre-fragment region
        # whose bounding box contains the post centroid. For overlapping
        # regions (e.g. bodies carved out of an enclosing volume), choose the
        # smallest containing box so the most specific region wins.
        new_region_volumes: dict[str, list[tuple[int, int]]] = {
            name: [] for name in self._region_volumes
        }
        tol = 1e-9

        def _bbox_volume(
            bbox: tuple[float, float, float, float, float, float],
        ) -> float:
            return max(bbox[3] - bbox[0], tol) * max(bbox[4] - bbox[1], tol) * max(bbox[5] - bbox[2], tol)

        def _contains(
            bbox: tuple[float, float, float, float, float, float],
            point: np.ndarray,
        ) -> bool:
            return (
                bbox[0] - tol <= point[0] <= bbox[3] + tol
                and bbox[1] - tol <= point[1] <= bbox[4] + tol
                and bbox[2] - tol <= point[2] <= bbox[5] + tol
            )

        def _sector_volume(sec: AnnularSector) -> float:
            return (
                max(sec.r_outer_mm * sec.r_outer_mm - sec.r_inner_mm * sec.r_inner_mm, tol)
                * max(abs(sec.theta_end_deg - sec.theta_start_deg), tol)
                * max(abs(sec.z_end_mm - sec.z_start_mm), tol)
            )

        def _sector_contains(sec: AnnularSector, point_m: np.ndarray) -> bool:
            x_mm = point_m[0] * 1e3
            y_mm = point_m[1] * 1e3
            z_mm = point_m[2] * 1e3
            r_mm = math.hypot(x_mm, y_mm)
            theta_deg = math.degrees(math.atan2(y_mm, x_mm))
            if abs(theta_deg) < 1e-7:
                theta_deg = 0.0
            elif theta_deg < 0.0:
                theta_deg += 360.0

            tol_mm = 1e-5
            tol_deg = 1e-5
            return (
                sec.r_inner_mm - tol_mm <= r_mm <= sec.r_outer_mm + tol_mm
                and sec.z_start_mm - tol_mm <= z_mm <= sec.z_end_mm + tol_mm
                and sec.theta_start_deg - tol_deg <= theta_deg <= sec.theta_end_deg + tol_deg
            )

        for dim, tag, post_c in post_centroids:
            best_name = None
            best_specificity = float("inf")

            for specificity, name in parent_candidates.get(tag, []):
                if specificity < best_specificity:
                    best_specificity = specificity
                    best_name = name

            if best_name is None:
                for name, sectors in self._region_sectors.items():
                    for sec in sectors:
                        if _sector_contains(sec, post_c):
                            specificity = _sector_volume(sec)
                            if specificity < best_specificity:
                                best_specificity = specificity
                                best_name = name

            if best_name is None:
                for name, bboxes in pre_bboxes.items():
                    for bbox in bboxes:
                        if _contains(bbox, post_c):
                            specificity = _bbox_volume(bbox)
                            if specificity < best_specificity:
                                best_specificity = specificity
                                best_name = name

            if best_name is None:
                # Fallback for tiny sliver entities whose centroid lands
                # outside all analytical sectors by numerical noise.
                best_dist = float("inf")
                for name in pre_centroids:
                    for pre_c in pre_centroids[name]:
                        d = float(np.linalg.norm(post_c - pre_c))
                        if d < best_dist:
                            best_dist = d
                            best_name = name

            if best_name is not None:
                new_region_volumes[best_name].append((dim, tag))

        for name, tags in new_region_volumes.items():
            if not tags:
                logger.warning("Boolean fragment left region %s with no volumes", name)

        self._region_volumes = new_region_volumes
        logger.info("Boolean fragment produced %d volumes", len(result_vols))

    def _assign_physical_groups(self) -> None:
        """Create Gmsh physical groups for each named region (volumes + boundary surfaces)."""
        for name, tags in self._region_volumes.items():
            if not tags:
                continue
            vol_tags = [t for _, t in tags]
            pg = gmsh.model.addPhysicalGroup(3, vol_tags)
            gmsh.model.setPhysicalName(3, pg, name)
            self._physical_groups[name] = pg

        # Detect sector boundary faces BEFORE creating OuterBoundary
        # so they can be excluded (they get their own physical groups
        # for periodic conditions, not the outer boundary's).
        self._detect_sector_faces()

        sector_face_tags = set(self._face_at_0 + self._face_at_sector)

        # Create named physical groups for sector faces
        if self._face_at_0:
            pg0 = gmsh.model.addPhysicalGroup(2, self._face_at_0)
            gmsh.model.setPhysicalName(2, pg0, "SectorFace_0")
            self._physical_groups["SectorFace_0"] = pg0

        if self._face_at_sector:
            pg1 = gmsh.model.addPhysicalGroup(2, self._face_at_sector)
            gmsh.model.setPhysicalName(2, pg1, "SectorFace_1")
            self._physical_groups["SectorFace_1"] = pg1

        # A 2-D physical group for the outer boundary surfaces, excluding
        # the sector faces, which have their own groups.
        all_vols = gmsh.model.getEntities(3)
        if all_vols:
            boundary_surfs = gmsh.model.getBoundary(
                all_vols, combined=True, oriented=False, recursive=False,
            )
            surf_tags = sorted(
                t for t in set(abs(t) for d, t in boundary_surfs if d == 2)
                if t not in sector_face_tags
            )
            if surf_tags:
                pg = gmsh.model.addPhysicalGroup(2, surf_tags)
                gmsh.model.setPhysicalName(2, pg, "OuterBoundary")
                self._physical_groups["OuterBoundary"] = pg
                logger.info("Created OuterBoundary physical surface with %d faces "
                            "(excluded %d sector faces)",
                            len(surf_tags), len(sector_face_tags))

        logger.info("Assigned %d physical groups", len(self._physical_groups))

    def _detect_sector_faces(self) -> None:
        """
        Identify the boundary faces lying in the theta=0 and theta=sector planes.

        A face belongs to a cut plane when every one of its vertices, and its
        centre of mass, lie in that plane, on the plane's own side of the z
        axis. The side matters because at a 180-degree sector the two cut
        planes are one plane: the radial coordinate along it, ``r_par``, is
        positive on the theta=0 half and negative on the other.

        Until 2026-10-01 this tested the corners of the face's bounding box,
        which lie in the plane only when the plane is x=0 or y=0: sectors of
        90 and 180 degrees found their faces, and every other angle (30, 60,
        72, 120...) found none at theta=sector, and the mesh went out with no
        periodic pairs and no error. Now a sector without faces on both cut
        planes is an error.
        """
        sector_deg = self.ir.sector_angle_deg
        sector_rad = math.radians(sector_deg)
        if sector_rad <= 0 or sector_deg >= 360.0:
            return

        all_surfaces = gmsh.model.getEntities(2)
        if not all_surfaces:
            return

        # Geometry is in metres; tolerance must also be in metres
        mean_r_m = (self.ir.r_inner_mm + self.ir.r_outer_mm) / 2.0 * 1e-3
        tol = max(mean_r_m * 0.001, 1e-7)

        planes = (
            (0.0, self._face_at_0),
            (sector_rad, self._face_at_sector),
        )

        for dim, tag in all_surfaces:
            try:
                pts = {abs(p) for _d, p in gmsh.model.getBoundary(
                    [(dim, tag)], combined=False, oriented=False, recursive=True)}
                xy = [gmsh.model.getValue(0, p, [])[:2] for p in pts]
                xy.append(gmsh.model.occ.getCenterOfMass(dim, tag)[:2])
            except Exception:
                continue

            for angle, bucket in planes:
                nx, ny = -math.sin(angle), math.cos(angle)   # plane normal
                tx, ty = math.cos(angle), math.sin(angle)    # in-plane radial
                if any(abs(x * nx + y * ny) >= tol for x, y in xy):
                    continue
                r_par = [x * tx + y * ty for x, y in xy]
                if min(r_par) > -tol and max(r_par) > tol:
                    bucket.append(tag)
                    break

        logger.info("Detected sector faces: %d at θ=0, %d at θ=%.1f°",
                    len(self._face_at_0), len(self._face_at_sector), sector_deg)
        if not (self._face_at_0 and self._face_at_sector):
            raise RuntimeError(
                f"A {sector_deg:g}-degree sector with {len(self._face_at_0)} faces "
                f"found at theta=0 and {len(self._face_at_sector)} at theta=sector: "
                f"it cannot be made periodic, and would solve silently wrong.")

    def _apply_mesh_policy(self) -> None:
        """
        Drive element size from a per-region background field: a ``Constant``
        field per region, combined with ``Min``, assigns size by *volume
        containment*, so each region gets exactly the size it is given, and
        a vertex two regions share does not take whichever was set last.
        """
        MM2M = 1e-3
        field_ids: list[int] = []

        for name, tags in self._region_volumes.items():
            vol_tags = [t for _, t in tags]
            if not vol_tags:
                continue
            size = self.policy.size_for(name) * MM2M
            fid = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumbers(fid, "VolumesList", vol_tags)
            gmsh.model.mesh.field.setNumber(fid, "VIn", size)
            # Outside this region the field must not compete in the Min.
            gmsh.model.mesh.field.setNumber(fid, "VOut", self.policy.global_max_mm * MM2M)
            field_ids.append(fid)
            self._region_target_size_mm[name] = size / MM2M

        if field_ids:
            bg = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(bg, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(bg)

            # The background field is the single authority on size; the other
            # sources would otherwise override it near boundaries.
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        if self.structured:
            self._structure_regions()

        # Global mesh size limits
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.policy.global_min_mm * MM2M)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.policy.global_max_mm * MM2M)

        # Use Mesh Algorithm 6 (Frontal-Delaunay) for quality
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(self.options["algorithm3d"]))
        if self.options.get("threads"):
            gmsh.option.setNumber("General.NumThreads", int(self.options["threads"]))
        netgen = bool(self.options.get("optimize_netgen"))
        if netgen and self.structured and self.structured.prisms:
            # Netgen on pyramids and tetrahedra together has crashed Gmsh
            # (an access violation in generate).
            logger.warning("Netgen optimiser skipped: not run with prisms")
            netgen = False
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if netgen else 0)

    def _structure_regions(self) -> None:
        """A transfinite mesh in every volume of the structured regions. Curve
        counts come from each curve's angle, radial length or height at the
        region's element size, so the curves two volumes share agree; a
        volume that is not a topological hexahedron after the fragment keeps
        the unstructured mesh."""
        spec = self.structured
        MM = 1e-3
        tol = 1e-9
        r_ref = spec.arc_radius_mm * MM
        done_curves: set[int] = set()
        for name in spec.regions:
            tags = self._region_volumes.get(name, [])
            h = self.policy.size_for(name) * MM
            for _dim, vol in tags:
                faces = gmsh.model.getBoundary([(3, vol)], oriented=False)
                curves = {abs(c) for _d, c in gmsh.model.getBoundary(
                    faces, combined=False, oriented=False)}
                if len(faces) != 6 or len(curves) != 12:
                    self._structured_gap["refused"] += 1
                    continue
                for c in curves:
                    if c in done_curves:
                        continue
                    ends = gmsh.model.getBoundary([(1, c)], oriented=False)
                    (x0, y0, z0), (x1, y1, z1) = (
                        gmsh.model.getValue(0, abs(p), []) for _d, p in ends)
                    r0, r1 = math.hypot(x0, y0), math.hypot(x1, y1)
                    if abs(z1 - z0) > 1e-6:                 # through the layer
                        n = 2
                    elif abs(r1 - r0) > tol:                # radial
                        n = max(2, math.ceil(round(abs(r1 - r0) / h, 6)) + 1)
                    else:                                   # an arc
                        dth = abs(math.atan2(x0 * y1 - y0 * x1, x0 * x1 + y0 * y1))
                        n = max(2, math.ceil(round(dth * r_ref / h, 6)) + 1)
                    gmsh.model.mesh.setTransfiniteCurve(c, n)
                    done_curves.add(c)
                # The faces across the layer are triangulated with their
                # corners in one (angle, radius) order, so the diagonals line
                # up through the volume; or, with prisms, the faces through it
                # are recombined.
                for _d, f in faces:
                    f = abs(f)
                    bb = gmsh.model.getBoundingBox(2, f)
                    # OCC pads a bounding box by its tolerance: judge by a
                    # micron.
                    if bb[5] - bb[2] > 1e-6:
                        gmsh.model.mesh.setTransfiniteSurface(f)
                        if spec.prisms:
                            gmsh.model.mesh.setRecombine(2, f)
                        continue
                    pts = {abs(p) for _dd, p in gmsh.model.getBoundary(
                        [(2, f)], combined=False, oriented=False, recursive=True)}
                    xyz = {p: gmsh.model.getValue(0, p, []) for p in pts}
                    corners = sorted(
                        pts, key=lambda p: (round(math.atan2(xyz[p][1], xyz[p][0]), 9),
                                            round(math.hypot(xyz[p][0], xyz[p][1]), 12)))
                    if len(corners) == 4:
                        # (th0,r0), (th0,r1), (th1,r0), (th1,r1) -> a loop.
                        a, b, c, d = corners
                        gmsh.model.mesh.setTransfiniteSurface(
                            f, spec.arrangement, [a, c, d, b])
                    else:
                        gmsh.model.mesh.setTransfiniteSurface(f)
                gmsh.model.mesh.setTransfiniteVolume(vol)
                self._structured_gap["columns"] += 1
        logger.info("Structured: %d volumes transfinite, %d left unstructured",
                    self._structured_gap["columns"], self._structured_gap["refused"])

    def _set_periodic_constraints(self) -> None:
        """
        Set periodic mesh constraints on the sector boundary faces: each
        slave face at theta=sector is paired with the master at theta=0 that
        it maps onto under the rotation, by rotated bounding-box centroids,
        nearest pairs first and each face used once. An unpaired face is an
        error: a partly periodic mesh would solve silently wrong.
        """
        sector_rad = math.radians(self.ir.sector_angle_deg)
        if sector_rad <= 0 or self.ir.sector_angle_deg >= 360.0:
            return
        if not (self._face_at_0 and self._face_at_sector):
            return

        # gmsh setPeriodic(dim, slave, master, affine) expects an affine that
        # maps MASTER coordinates → SLAVE coordinates.  Master is at theta=0,
        # slave at theta=sector, so it is a rotation by +sector about z.
        cos_a = math.cos(sector_rad)
        sin_a = math.sin(sector_rad)
        affine = [
            cos_a, -sin_a, 0, 0,
            sin_a,  cos_a, 0, 0,
            0,      0,     1, 0,
            0,      0,     0, 1,
        ]

        def centroid(tag: int) -> tuple[float, float, float]:
            b = gmsh.model.getBoundingBox(2, tag)
            return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)

        masters = {tag: centroid(tag) for tag in self._face_at_0}
        mean_r_m = (self.ir.r_inner_mm + self.ir.r_outer_mm) / 2.0 * 1e-3
        tol = max(mean_r_m * 0.01, 1e-6)

        candidates: list[tuple[float, int, int]] = []
        for slave in self._face_at_sector:
            sx, sy, sz = centroid(slave)
            for master, (mx, my, mz) in masters.items():
                # Rotate the master centroid onto the slave plane.
                rx = mx * cos_a - my * sin_a
                ry = mx * sin_a + my * cos_a
                dist = math.dist((rx, ry, mz), (sx, sy, sz))
                if dist <= tol:
                    candidates.append((dist, slave, master))
        candidates.sort()

        n_paired = 0
        used_slaves: set[int] = set()
        used_masters: set[int] = set()
        for dist, slave, master in candidates:
            if slave in used_slaves or master in used_masters:
                continue
            try:
                gmsh.model.mesh.setPeriodic(2, [slave], [master], affine)
            except Exception:
                logger.warning(
                    "setPeriodic rejected slave face %d / master %d "
                    "(centroid distance %.3e m)", slave, master, dist,
                )
                continue
            used_slaves.add(slave)
            used_masters.add(master)
            n_paired += 1

        unmatched = [t for t in self._face_at_sector if t not in used_slaves]

        if n_paired == 0:
            raise RuntimeError(
                f"setPeriodic: NO face pairs matched. Mesh periodicity is NOT "
                f"enforced ({len(self._face_at_0)} faces at θ=0, "
                f"{len(self._face_at_sector)} at θ=sector). Check the sector "
                f"face detection tolerance."
            )
        if unmatched:
            raise RuntimeError(
                f"setPeriodic: {len(unmatched)} of "
                f"{len(self._face_at_sector)} sector faces found no partner "
                f"(tags {unmatched[:8]}). A partially periodic mesh gives a "
                f"silently wrong field, so this is fatal rather than a warning."
            )
        logger.info(
            "Set periodic constraints: %d face pairs (%d at θ=0, %d at θ=sector)",
            n_paired, len(self._face_at_0), len(self._face_at_sector),
        )

    def _generate_mesh(self) -> None:
        """Generate the 3D tetrahedral mesh."""
        gmsh.model.mesh.generate(3)
        node_count = len(gmsh.model.mesh.getNodes()[0])
        _types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        elem_count = sum(len(t) for t in elem_tags)
        logger.info("Mesh: %d nodes, %d 3D elements", node_count, elem_count)

    # ── measurements ─────────────────────────────────────────

    def mesh_stats(self) -> dict[str, int]:
        """Return basic mesh statistics."""
        nodes = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        n_elements = sum(len(et) for et in elem_tags)
        return {
            "nodes": len(nodes[0]),
            "elements_3d": n_elements,
        }

    def region_element_counts(self) -> dict[str, int]:
        """The 3-D elements in each region."""
        out: dict[str, int] = {}
        for name, tags in self._region_volumes.items():
            n = 0
            for dim, tag in tags:
                try:
                    _types, tag_lists, _ = gmsh.model.mesh.getElements(dim, tag)
                except Exception:
                    continue
                n += sum(len(t) for t in tag_lists)
            out[name] = n
        return out

    def element_gammas(self) -> dict[str, np.ndarray]:
        """Each region's elements' shape quality, gamma (inradius over
        circumradius, normalised: 1 is regular). Pyramids are left out:
        Gmsh does not define gamma on them."""
        out: dict[str, np.ndarray] = {}
        for name, tags in self._region_volumes.items():
            values: list[float] = []
            for _dim, tag in tags:
                try:
                    types, elem_tags, _ = gmsh.model.mesh.getElements(3, tag)
                except Exception:
                    continue
                for etype, etags in zip(types, elem_tags):
                    if len(etags) == 0 or etype in (7, 14):       # pyramids
                        continue
                    try:
                        values.extend(gmsh.model.mesh.getElementQualities(etags, "gamma"))
                    except Exception:
                        continue
            out[name] = np.asarray(values, dtype=float)
        return out

    def diagnostics(self) -> dict:
        """What was made: counts, groups, and the sizes applied."""
        return {
            **self.mesh_stats(),
            "physical_groups": self.physical_groups,
            "region_volume_counts": self.region_volume_counts,
            "region_element_counts": self.region_element_counts(),
            "region_target_size_mm": dict(self._region_target_size_mm),
            "sector_faces": {
                "theta0": len(self._face_at_0),
                "theta_sector": len(self._face_at_sector),
            },
            "structured": dict(self._structured_gap),
        }

    def launch_gui(self) -> None:
        """Open the Gmsh GUI for visual inspection (blocks)."""
        gmsh.fltk.run()
