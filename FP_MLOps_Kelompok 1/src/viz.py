"""
viz.py - Konfigurasi palet visualisasi terpusat.

Satu-satunya tempat palet brand didefinisikan. Modul lain (notebook maupun skrip)
cukup `import src.viz` supaya rcParams matplotlib otomatis konsisten - tidak ada
penyalinan ulang palet di tempat lain.

Palet dan seluruh konstanta warna WAJIB persis seperti yang ditentukan dokumen soal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

# =============================================================================
# PALET BRAND
# =============================================================================

BRAND_INDIGO = "#3D2B9E"
BRAND_VIOLET = "#7B3FE4"
BRAND_MAGENTA = "#CE1C8E"
BRAND_PINK = "#EC6DB4"
BRAND_LIGHT = "#F7B4DA"
BRAND_COLORS = [BRAND_MAGENTA, BRAND_INDIGO, BRAND_VIOLET, BRAND_PINK, BRAND_LIGHT]
ACCENT = BRAND_MAGENTA
ACCENT2 = BRAND_INDIGO

BRAND_CMAP = LinearSegmentedColormap.from_list(
    "brand", ["#241663", BRAND_INDIGO, BRAND_VIOLET, BRAND_MAGENTA, BRAND_PINK, BRAND_LIGHT]
)
try:
    mpl.colormaps.register(BRAND_CMAP)
    mpl.colormaps.register(BRAND_CMAP.reversed())
except (ValueError, AttributeError):
    pass

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=BRAND_COLORS)
plt.rcParams["image.cmap"] = "brand"
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def savefig(fig, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Simpan figure ke berkas PNG di `directory` (default outputs/figures).

    Dipakai supaya seluruh visualisasi tersimpan sebagai berkas, bukan hanya
    tampil di notebook.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


__all__ = [
    "BRAND_INDIGO", "BRAND_VIOLET", "BRAND_MAGENTA", "BRAND_PINK", "BRAND_LIGHT",
    "BRAND_COLORS", "ACCENT", "ACCENT2", "BRAND_CMAP", "savefig",
]
