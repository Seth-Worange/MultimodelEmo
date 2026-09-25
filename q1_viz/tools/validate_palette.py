"""Faithful Python port of the dataviz skill's validate_palette.js (categorical checks).

The machine has no Node.js; this port keeps the JS math identical (OKLab Delta E
x100, Machado-Oliveira-Fernandes 2009 severity-1.0 CVD simulation, WCAG contrast)
so the six computable checks run here rather than being eyeballed.

Usage:
    python q1_viz/tools/validate_palette.py "#2a78d6,#eb6834" --mode light
"""

from __future__ import annotations

import argparse
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(value: str) -> list[float]:
    text = value.strip().lstrip("#")
    return [int(text[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(value: str) -> list[float]:
    return [s2lin(c) for c in hex2srgb(value)]


def rel_lum(value: str) -> float:
    r, g, b = lin(value)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb: list[float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(value: str) -> tuple[float, float]:
    lab, a, b = oklab_from_lin(lin(value))
    return lab, math.hypot(a, b)


def simulate(value: str, kind: str) -> list[float]:
    matrix = MACHADO[kind]
    r, g, b = lin(value)
    return [max(0.0, min(1.0, matrix[i][0] * r + matrix[i][1] * g + matrix[i][2] * b))
            for i in range(3)]


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    first = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    second = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.sqrt(sum((x - y) ** 2 for x, y in zip(first, second)))


def validate(palette: list[str], mode: str, surface: str, pairs: str = "all") -> tuple[list, bool]:
    lo, hi = BAND[mode]
    report, ok = [], True

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not offband
    report.append(("Lightness band", not offband,
                   f"outside band: {offband}" if offband else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("Chroma floor", not lowc,
                   f"below floor: {lowc}" if lowc else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    pairlist = ([(i, j) for i in range(n) for j in range(i + 1, n)] if pairs == "all"
                else [(i, i + 1) for i in range(n - 1)])
    label = "all-pairs" if pairs == "all" else "adjacent"
    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            distance = delta_e(palette[i], palette[j], kind)
            if worst is None or distance < worst[0]:
                worst = (distance, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99)
    worst_delta = worst[0] if worst else 99
    cvd_state = "PASS" if worst_delta >= CVD_TARGET else ("WARN" if worst_delta >= CVD_FLOOR else "FAIL")
    ok &= cvd_state != "FAIL"
    report.append(("CVD separation", cvd_state,
                   f"worst {label} {worst[3]}<->{worst[2]} dE {worst_delta:.1f} ({worst[1]}) tritan {tri:.1f}"))

    normal_worst = min((delta_e(palette[i], palette[j]) for i, j in pairlist), default=99)
    normal_state = "PASS" if normal_worst >= NORMAL_FLOOR else "FAIL"
    ok &= normal_state == "PASS"
    report.append(("Normal-vision floor", normal_state,
                   f"worst {label} dE {normal_worst:.1f} (floor {NORMAL_FLOOR})"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "WARN" if low else "PASS",
                   f"below {CONTRAST_MIN}:1 relief required: {low}" if low
                   else f"all {len(palette)} >= {CONTRAST_MIN}:1"))
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("palette")
    parser.add_argument("--mode", choices=("light", "dark"), default="light")
    parser.add_argument("--surface", default=None)
    parser.add_argument("--pairs", choices=("adjacent", "all"), default="all")
    args = parser.parse_args()
    palette = [item.strip() for item in args.palette.split(",") if item.strip()]
    surface = args.surface or DEFAULT_SURFACE[args.mode]
    report, ok = validate(palette, args.mode, surface, args.pairs)
    print(f"Palette ({args.mode}, surface {surface}, {args.pairs}): {len(palette)} slots")
    for name, state, detail in report:
        print(f"  [{state:<4}] {name:<22} {detail}")
    print("  -> " + ("ALL CHECKS PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
