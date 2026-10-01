"""The entry point the mesher's own executable is compiled from:
``afmsim-mesher.exe`` is ``python -m afmsim_mesher`` (GPL-2.0-or-later; its
source is https://github.com/crypticsymmetry/afmsim-mesher)."""
from afmsim_mesher.__main__ import cli

if __name__ == "__main__":
    cli()
