"""Render the three-model four-arm A100 comparison from a qualified matrix CSV."""
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#152F3D", "#455B66", "#DAE2E5"
BASE, OPT, SECOND = "#70838C", "#006F70", "#31689B"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42,
})


MODELS = {"timesfm25": "TimesFM", "toto2": "Toto", "moirai2": "Moirai"}
# The same four modes appear in the same order in all three figures.
MODES = {
    "native": ("Native HTTP", BASE, "-"),
    "single": ("One slot", "#AD603C", "-."),
    "unfused": ("Four slots, unfused", SECOND, ":"),
    "fused": ("Four slots, fused", OPT, "-"),
}
OFFERS = {"batch1": (20, 40, 80), "mixed": (10, 30)}
REQUESTS_PER_ROUND = 100
REQUIRED_COLUMNS = {
    "source", "gpu_uuid", "cases_sha256", "reference_sha256", "seed",
    "adapter", "group", "offered_rate", "mode", "rounds", "round_numbers",
    "completed", "req_s", "p95_ms", "worst_round_arrival_p99_ms",
}


@dataclass(frozen=True)
class Point:
    requests_per_second: float
    p95_ms: float
    arrival_p99_ms: float | None


@dataclass(frozen=True)
class Matrix:
    revision: str
    rounds: tuple[int, ...]
    points: dict[tuple[str, str, float | None, str], Point]

    @property
    def rotation_label(self):
        count = len(self.rounds)
        return f"{count} matched rotation" + ("s" if count != 1 else "")


def positive_number(value, column):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{column} must be finite and positive, got {value!r}")
    return number


def parse_rounds(row):
    rounds = tuple(sorted(int(value) for value in row["round_numbers"].split(",")))
    if not rounds or rounds[0] < 1 or len(set(rounds)) != len(rounds):
        raise ValueError("round_numbers must list distinct positive rounds")
    if int(row["rounds"]) != len(rounds):
        raise ValueError("rounds must equal the number of listed rounds")
    if int(row["completed"]) != REQUESTS_PER_ROUND * len(rounds):
        raise ValueError(f"Expected {REQUESTS_PER_ROUND} successful forecasts per round")
    return rounds


def read_matched(path):
    """Require one complete qualified four-arm matrix before plotting any panel."""
    points = {}
    references = {}
    provenance = None
    rounds = None
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("CSV contains duplicate column names")
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))

        for row in reader:
            try:
                if None in row or any(row[c] is None for c in REQUIRED_COLUMNS):
                    raise ValueError("Malformed CSV row")
                run = tuple(row[c].strip() for c in ("source", "gpu_uuid", "cases_sha256", "seed"))
                if not all(run) or (provenance is not None and run != provenance):
                    raise ValueError("Rows must share a source revision, GPU, case manifest and arrival seed")
                provenance = run
                row_rounds = parse_rounds(row)
                if rounds is not None and row_rounds != rounds:
                    raise ValueError("All modes, offered rates and models must use the same round IDs")
                rounds = row_rounds

                adapter, group, mode = row["adapter"], row["group"], row["mode"]
                if adapter not in MODELS or group not in OFFERS or mode not in MODES:
                    raise ValueError(f"Unexpected model, workload or serving mode: {(adapter, group, mode)}")
                reference = row["reference_sha256"].strip()
                if not reference or references.setdefault(adapter, reference) != reference:
                    raise ValueError(f"{adapter} must use one native reference across all arms")
                offer = positive_number(row["offered_rate"], "offered_rate") if row["offered_rate"] else None
                if offer not in OFFERS[group] and (offer is not None or group != "batch1"):
                    raise ValueError(f"Unexpected nominal offer {offer} for {group}")
                key = (adapter, group, offer, mode)
                if key in points:
                    raise ValueError(f"Duplicate arm for {key}")
                arrival = row["worst_round_arrival_p99_ms"]
                if offer is None and arrival:
                    raise ValueError("Closed-loop points must not have an offered-arrival p99")
                points[key] = Point(
                    positive_number(row["req_s"], "req_s"),
                    positive_number(row["p95_ms"], "p95_ms"),
                    positive_number(arrival, "worst_round_arrival_p99_ms") if offer is not None else None,
                )
            except ValueError as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error

    if not points:
        raise ValueError("CSV has no qualified measurements")
    for adapter in MODELS:
        for group, offers in OFFERS.items():
            required_offers = (None, *offers) if group == "batch1" else offers
            for offer in required_offers:
                for mode in MODES:
                    if (adapter, group, offer, mode) not in points:
                        raise ValueError(f"Missing {adapter}/{group} at nominal offer {offer}, mode {mode}")
    return Matrix(provenance[0], rounds, points)


def style_panel(axis, title):
    axis.set_title(title, fontsize=15, weight="bold", pad=12)
    axis.grid(axis="y", color=GRID)
    axis.set_axisbelow(True)
    axis.tick_params(which="both", length=0, pad=6)
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_figure(figure, directory, stem):
    figure.savefig(directory / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(directory / f"{stem}.png", dpi=165, facecolor="white")
    plt.close(figure)


def single_request_figure(matrix, directory):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 6.7), sharey=True)
    figure.subplots_adjust(left=.08, right=.975, top=.73, bottom=.265, wspace=.15)
    for axis, (adapter, name) in zip(axes, MODELS.items(), strict=True):
        for index, (mode, (_, color, _)) in enumerate(MODES.items()):
            point = matrix.points[adapter, "batch1", None, mode]
            axis.bar(index, point.p95_ms, width=.62, color=color)
            axis.annotate(f"{point.p95_ms:.0f}", (index, point.p95_ms),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11.5)
        axis.set_xticks(range(4), ["Native", "1 slot", "4 slots,\nunfused", "4 slots,\nfused"], fontsize=10)
        style_panel(axis, name)
    axes[0].set_ylabel("End-to-end p95 latency (ms)")
    axes[0].margins(y=.15)
    figure.text(.05, .92, "One-request latency across three forecast models", fontsize=24, weight="bold", va="top")
    figure.text(.05, .835, "A100   ·   MetroPT   ·   One series, 2,048 observations, 96 forecast steps   ·   Closed-loop HTTP",
                fontsize=13.5, color=MUTED, va="top")
    figure.text(.05, .08, "Each client waits for its forecast before sending the next request. Cross-request fusion has no work to combine.",
                fontsize=11, color=MUTED)
    figure.text(.05, .038, f"Benchmark {matrix.revision[:7]} · {matrix.rotation_label} · 100 forecasts per arm per rotation · "
                "Each timed response passed the independent native check.", fontsize=10.3, color=MUTED)
    save_figure(figure, directory, "10-single-request-latency")


def arrival_figure(matrix, group, directory):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 6.7), sharex=True, sharey=True)
    figure.subplots_adjust(left=.08, right=.975, top=.72, bottom=.31, wspace=.15)
    offers = OFFERS[group]
    markers = dict(zip(offers, ("o", "^", "D")[:len(offers)], strict=True))
    for axis, (adapter, name) in zip(axes, MODELS.items(), strict=True):
        for mode, (_, color, line_style) in MODES.items():
            points = [matrix.points[adapter, group, rate, mode] for rate in offers]
            axis.plot([p.requests_per_second for p in points], [p.arrival_p99_ms for p in points],
                      color=color, linestyle=line_style, linewidth=1.7)
            for rate, point in zip(offers, points, strict=True):
                axis.plot(point.requests_per_second, point.arrival_p99_ms, marker=markers[rate],
                          markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8, markersize=9 if mode == "single" else 6.8)
        style_panel(axis, name)
        axis.axhline(500, color=INK, linestyle="--", linewidth=1)
        axis.set_yscale("log")

    axes[0].set_xlim(left=0)
    axes[0].set_ylabel("Worst repeat p99 from arrival (ms, log scale)", fontsize=12, color=MUTED)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    figure.supxlabel("Achieved throughput (completed requests/s)", y=.224, fontsize=12, color=MUTED)
    title = "Single-series arrivals" if group == "batch1" else "Mixed-size arrivals"
    figure.text(.05, .92, f"{title} across three forecast models", fontsize=24, weight="bold", va="top")
    figure.text(.05, .835, "A100   ·   MetroPT   ·   Forecast horizon 96   ·   Open-loop Poisson arrivals",
                fontsize=14, color=MUTED, va="top")

    modes = [Line2D([], [], color=color, linestyle=line_style, label=label, linewidth=2.8)
             for label, color, line_style in MODES.values()]
    modes.append(Line2D([], [], color=INK, linestyle="--", label="500 ms target"))
    figure.legend(handles=modes, loc="center", bbox_to_anchor=(.5, .166), ncol=5, frameon=False, fontsize=10.5)
    symbols = [Line2D([], [], color=INK, marker=marker, markerfacecolor="white", linestyle="none",
                      label=f"{rate} req/s offered") for rate, marker in markers.items()]
    figure.legend(handles=symbols, loc="center", bbox_to_anchor=(.5, .106), ncol=len(offers), frameon=False, fontsize=10.5)
    figure.text(.05, .038, f"Benchmark {matrix.revision[:7]} · {matrix.rotation_label} · 100 forecasts per arm per offer and rotation · "
                "Each timed response passed the independent native check.", fontsize=10.3, color=MUTED)
    stem = "11-homogeneous-arrivals" if group == "batch1" else "12-mixed-arrivals"
    save_figure(figure, directory, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=ROOT / "data" / "serving-matrix-a100.csv",
                        help="Qualified output from benchmarks.http_matrix_report")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        matrix = read_matched(args.csv)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    single_request_figure(matrix, directory)
    for workload in OFFERS:
        arrival_figure(matrix, workload, directory)


if __name__ == "__main__":
    main()
