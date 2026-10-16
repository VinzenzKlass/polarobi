# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# matplotlib's annotations are only partially typed.
"""
Figures for the documentation from a benchmark result file.

Writes a light and a dark variant of each figure to `docs/img/`, and prints the results as a
Markdown table for the page.

    uv run --group bench python benchmarks/plot.py benchmarks/results/windows.json
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_IMG = Path(__file__).parents[1] / "docs" / "img"
# Series in legend order, with their labels. polarobi's label goes first where labels compete.
_BUILDERS = {
    "gurobipy_loop": "gurobipy, addLConstr loop",
    "gurobipy_addconstrs": "gurobipy, addConstrs",
    "gurobipy_pandas": "gurobipy-pandas",
    "polarobi": "polarobi",
}


@dataclass(frozen=True, slots=True)
class Theme:
    """Colors from the validated reference palette (categorical slots 1-4), stepped per mode."""

    name: str
    series: dict[str, str]
    ink: str
    muted: str
    grid: str
    axis: str
    surface: str
    """The documentation's page background, for the rings around markers."""


_THEMES = (
    Theme(
        "light",
        {
            "polarobi": "#2a78d6",
            "gurobipy_loop": "#eb6834",
            "gurobipy_addconstrs": "#1baf7a",
            "gurobipy_pandas": "#eda100",
        },
        ink="#0b0b0b",
        muted="#52514e",
        grid="#e1e0d9",
        axis="#c3c2b7",
        surface="#ffffff",
    ),
    Theme(
        "dark",
        {
            "polarobi": "#3987e5",
            "gurobipy_loop": "#d95926",
            "gurobipy_addconstrs": "#199e70",
            "gurobipy_pandas": "#c98500",
        },
        ink="#f0efec",
        muted="#c3c2b7",
        grid="#2c2c2a",
        axis="#383835",
        surface="#0b0c0f",
    ),
)


def _style(ax: Axes, theme: Theme, grid: Literal["both", "x", "y"] = "both") -> None:
    ax.set_facecolor("none")
    ax.grid(True, which="major", axis=grid, color=theme.grid, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.axis)
    ax.tick_params(which="both", colors=theme.muted, labelsize=10, length=0, pad=6)


def _seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.1f} s"
    return f"{value * 1000:.0f} ms"


def _plain_seconds(value: float, _: object = None) -> str:
    return f"{value:g} s"


def _rows(value: float, _: object = None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _legend(
    ax: Axes, theme: Theme, loc: Literal["upper left", "upper right"] = "upper left"
) -> None:
    ax.legend(loc=loc, frameon=False, fontsize=10, handlelength=1.2, labelcolor=theme.ink)


def build_time(results: Sequence[dict[str, Any]], theme: Theme, out: Path) -> None:
    """Total build time against model size, both axes linear, with the spread across runs."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100)
    fig.patch.set_alpha(0)
    _style(ax, theme)
    ends: dict[str, tuple[float, float]] = {}
    for builder, label in _BUILDERS.items():
        rows = [r for r in results if r["builder"] == builder]
        sizes = [r["rows"] for r in rows]
        totals = [r["total_s"] for r in rows]
        # The band runs from the fastest to the slowest run; the line is the median.
        ax.fill_between(
            sizes,
            [min(r["runs_total_s"]) for r in rows],
            [max(r["runs_total_s"]) for r in rows],
            color=theme.series[builder],
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            sizes,
            totals,
            color=theme.series[builder],
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            marker="o",
            markersize=10,
            # A 2px ring in the page color keeps overlapping markers apart.
            markeredgecolor=theme.surface,
            markeredgewidth=2,
            label=label,
            zorder=3,
        )
        ends[builder] = (sizes[-1], totals[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_rows))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain_seconds))
    ax.set_xlabel("Rows (variables in the largest family)", color=theme.muted, fontsize=10)
    ax.margins(x=0.12)
    _legend(ax, theme)
    fig.tight_layout()
    # Direct labels at the largest size, polarobi's first. A label that would sit within 15px of
    # one already placed is left to the legend and the table, rather than nudged off its line.
    placed: list[float] = []
    for builder in sorted(ends, key=lambda b: b != "polarobi"):
        size, total = ends[builder]
        y = float(ax.transData.transform((size, total))[1])
        if any(abs(y - other) < 15 for other in placed):
            continue
        placed.append(y)
        ax.annotate(
            _seconds(total),
            (size, total),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=theme.ink,
        )
    fig.tight_layout()
    fig.savefig(out, transparent=True)
    plt.close(fig)


def phases(results: Sequence[dict[str, Any]], theme: Theme, out: Path) -> None:
    """Time per phase at the largest size, as horizontal bars grouped by phase."""
    largest = max(r["rows"] for r in results)
    rows = {r["builder"]: r for r in results if r["rows"] == largest}
    groups = (("Variables", "variables_s"), ("Constraints", "constraints_s"))
    # Tall enough for one 24px bar per build in each group, plus the title and the legend.
    fig, ax = plt.subplots(figsize=(7.2, 1.6 + 0.6 * len(_BUILDERS)), dpi=100)
    fig.patch.set_alpha(0)
    _style(ax, theme, grid="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks(range(len(groups)), [label for label, _ in groups])
    ax.set_ylim(len(groups) - 0.5, -0.5)  # First group on top.
    longest = max(r[key] for r in rows.values() for _, key in groups)
    ax.set_xlim(0, longest * 1.15)
    fig.tight_layout()
    # Bars at most 24px thick with a 2px gap between neighbours, in data units.
    per_px = len(groups) / ax.get_window_extent().height
    height, gap = 24 * per_px, 2 * per_px
    count = len(_BUILDERS)
    for i, (builder, label) in enumerate(_BUILDERS.items()):
        offsets = [g + (i - (count - 1) / 2) * (height + gap) for g in range(len(groups))]
        values = [rows[builder][key] for _, key in groups]
        ax.barh(offsets, values, height=height, color=theme.series[builder], label=label, zorder=3)
        for y, value in zip(offsets, values, strict=True):
            ax.annotate(
                f"{value:.1f} s",  # One unit within the chart keeps the labels comparable.
                (value, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=theme.ink,
            )
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_seconds))
    ax.set_title(f"{_rows(largest)} rows", loc="left", fontsize=10, color=theme.muted)
    _legend(ax, theme, loc="upper right")
    fig.tight_layout()  # Again, now that the title takes space; bars stay about 24px.
    fig.savefig(out, transparent=True)
    plt.close(fig)


def _spread(result: dict[str, Any]) -> str:
    """The median with the fastest and slowest run, in the median's unit."""
    median, runs = result["total_s"], result["runs_total_s"]
    scale, digits = (1, 1) if median >= 1 else (1000, 0)
    low, high = (f"{value * scale:.{digits}f}" for value in (min(runs), max(runs)))
    return f"{_seconds(median)} ({low}–{high})"


def table(results: Sequence[dict[str, Any]]) -> str:
    """
    The results as an HTML table for the page: per size a row with each build's median and, in
    parentheses, its fastest and slowest run, which expands to one row per run. The speedup
    divides the median of the faster plain gurobipy build by polarobi's.

    Markdown tables cannot group rows, hence HTML. Each size is a `tbody` whose first cell holds
    a checkbox; `docs/css/styles.css` shows the run rows of a checked group, without JavaScript.
    """
    headers = {
        "gurobipy_loop": "gurobipy, <code>addLConstr</code> loop",
        "gurobipy_addconstrs": "gurobipy, <code>addConstrs</code>",
        "gurobipy_pandas": "gurobipy-pandas",
        "polarobi": "polarobi",
    }
    head = "".join(f"<th>{headers[b]}</th>" for b in _BUILDERS)
    lines = [
        '<div class="benchmark-table">',
        "<table>",
        f"<thead><tr><th>Rows</th>{head}<th>Speedup</th></tr></thead>",
    ]
    for size in sorted({r["rows"] for r in results}):
        rows = {r["builder"]: r for r in results if r["rows"] == size}
        # Against the faster of the two plain gurobipy builds by median.
        gurobipy = min(
            rows["gurobipy_loop"], rows["gurobipy_addconstrs"], key=lambda r: r["total_s"]
        )
        ours = rows["polarobi"]
        speedup = gurobipy["total_s"] / ours["total_s"]
        # From its fastest run against polarobi's slowest, to its slowest against the fastest.
        worst = min(gurobipy["runs_total_s"]) / max(ours["runs_total_s"])
        best = max(gurobipy["runs_total_s"]) / min(ours["runs_total_s"])
        cells = "".join(f"<td>{_spread(rows[b])}</td>" for b in _BUILDERS)
        toggle = (
            f'<label title="Show each run"><input type="checkbox"'
            f' aria-label="Show each run at {size:,} rows">{size:,}</label>'
        )
        lines += [
            "<tbody>",
            f'<tr class="summary"><td>{toggle}</td>{cells}'
            f"<td>{speedup:.1f}x ({worst:.1f}–{best:.1f})</td></tr>",
        ]
        for run in range(max(len(r["runs_total_s"]) for r in rows.values())):
            runs = "".join(
                f"<td>{_seconds(rows[b]['runs_total_s'][run])}</td>"
                if run < len(rows[b]["runs_total_s"])
                else "<td></td>"
                for b in _BUILDERS
            )
            lines.append(f'<tr class="run"><td>run {run + 1}</td>{runs}<td></td></tr>')
        lines.append("</tbody>")
    lines += ["</table>", "</div>"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Documentation figures from benchmark results.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text(encoding="utf-8"))
    results = report["results"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",  # Text stays text and uses the page's font.
        }
    )
    _IMG.mkdir(parents=True, exist_ok=True)
    for theme in _THEMES:
        build_time(results, theme, _IMG / f"benchmark-build-time-{theme.name}.svg")
        phases(results, theme, _IMG / f"benchmark-phases-{theme.name}.svg")
    print(table(results))


if __name__ == "__main__":
    main()
