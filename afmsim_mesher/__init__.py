"""
afmsim_mesher: the AFPM simulator's Gmsh mesher, run as a separate program.

GPL-2.0-or-later (see README.md and COPYING). It imports Gmsh; it imports
nothing of afmsim, which talks to it only through files
(``python -m afmsim_mesher <command> request.json``).
"""

__version__ = "2.0.0"
