from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KCAL_PER_KJ = 1.0 / 4.184


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot native EVB-gap metadynamics walkers.")
    parser.add_argument("--base", default="outputs/hg317_evb_gap_metad", help="Base gap metadynamics output directory.")
    parser.add_argument("--output", default=None, help="Summary plot directory.")
    args = parser.parse_args()
    base = Path(args.base)
    output = Path(args.output) if args.output else base / "summary"
    output.mkdir(parents=True, exist_ok=True)

    replicas = []
    for rep_dir in sorted(base.glob("rep*")):
        obs_path = rep_dir / "gap_metad_observables.csv"
        fel_path = rep_dir / "gap_metad_fel.csv"
        conv_path = rep_dir / "gap_metad_convergence.json"
        if obs_path.is_file():
            replicas.append((rep_dir.name, obs_path, fel_path, conv_path))
    if not replicas:
        raise SystemExit(f"No gap metadynamics replicas found in {base}")

    obs = {}
    fel = {}
    convergence = {}
    for name, obs_path, fel_path, conv_path in replicas:
        obs[name] = pd.read_csv(obs_path).replace([np.inf, -np.inf], np.nan).dropna(subset=["gap_shifted_kJmol", "w2"])
        if fel_path.is_file():
            fel[name] = pd.read_csv(fel_path)
        convergence[name] = json.loads(conv_path.read_text(encoding="utf-8")) if conv_path.is_file() else {}

    if fel:
        _plot_fel(fel, obs, output / "gap_metad_fel_kcal.png")
    else:
        _plot_no_fel(output / "gap_metad_fel_kcal.png")
    _plot_histogram_fel(fel, obs, output / "gap_metad_histogram_fel_kcal.png")
    _plot_histogram_fel(fel, obs, output / "gap_metad_histogram_fel_ts_zoom_kcal.png", zoom_kcal=(-1500.0, 1500.0))
    _plot_gap_trace(obs, output / "gap_trace_kcal.png")
    _plot_weights(obs, output / "evb_weights_vs_gap.png")
    _plot_evb_energy_scatter(obs, output / "evb_energy_vs_gap_kcal.png")
    _plot_geometry(obs, output / "proton_transfer_geometry.png")

    summary = _summarize(obs, fel, convergence)
    (output / "gap_metad_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _plot_fel(fel: dict[str, pd.DataFrame], obs: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    interpolated = []
    common_grid = None
    for name, df in fel.items():
        x = df["gap_kcalmol"].to_numpy()
        y = df["free_energy_kcalmol"].to_numpy()
        ax.plot(x, y, lw=1.2, alpha=0.65, label=name)
        common_grid = x if common_grid is None else common_grid
        interpolated.append(np.interp(common_grid, x, y))
    if common_grid is not None and interpolated:
        mean = np.mean(np.vstack(interpolated), axis=0)
        mean = mean - np.nanmin(mean)
        ax.plot(common_grid, mean, color="black", lw=2.6, label="mean FEL")
    for name, df in obs.items():
        sampled = df["gap_shifted_kJmol"].to_numpy() * KCAL_PER_KJ
        ax.axvspan(np.nanmin(sampled), np.nanmax(sampled), alpha=0.08, label=f"{name} sampled range")
    ax.axvline(0.0, color="crimson", ls="--", lw=2, label="EVB TS / mixing gap = 0")
    ax.set_xlabel("EVB shifted gap (kcal/mol)")
    ax.set_ylabel("Free energy estimate (kcal/mol)")
    ax.set_title("Native EVB-Gap Metadynamics FEL")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_no_fel(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.text(
        0.5,
        0.55,
        "No final FEL available",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        "The metadynamics run crashed before writing gap_metad_fel.csv.\n"
        "Use the histogram and trace plots as partial diagnostics only.",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_histogram_fel(
    fel: dict[str, pd.DataFrame],
    obs: dict[str, pd.DataFrame],
    path: Path,
    zoom_kcal: tuple[float, float] | None = None,
) -> None:
    fig, ax_hist = plt.subplots(figsize=(9.5, 5.8))
    ax_fel = ax_hist.twinx()
    if fel:
        first_fel = next(iter(fel.values()))
        grid = first_fel["gap_kcalmol"].to_numpy()
        x_min = float(np.nanmin(grid))
        x_max = float(np.nanmax(grid))
    else:
        all_gap = np.concatenate([df["gap_shifted_kJmol"].to_numpy() * KCAL_PER_KJ for df in obs.values() if len(df)])
        x_min = float(np.nanpercentile(all_gap, 1))
        x_max = float(np.nanpercentile(all_gap, 99))
        grid = np.linspace(x_min, x_max, 200)
    if zoom_kcal is not None:
        x_min, x_max = zoom_kcal
    bins = np.linspace(x_min, x_max, 80)
    all_samples = []
    for name, df in obs.items():
        samples = df["gap_shifted_kJmol"].to_numpy() * KCAL_PER_KJ
        samples = samples[(samples >= x_min) & (samples <= x_max)]
        all_samples.append(samples)
        ax_hist.hist(samples, bins=bins, density=False, alpha=0.28, label=f"{name} samples")
    pooled = np.concatenate([samples for samples in all_samples if len(samples)]) if any(len(s) for s in all_samples) else np.array([])
    if len(pooled):
        ax_hist.hist(pooled, bins=bins, histtype="step", linewidth=2.0, color="black", label="pooled samples")

    interpolated = []
    if fel:
        common_grid = grid[(grid >= x_min) & (grid <= x_max)]
        for name, df in fel.items():
            x = df["gap_kcalmol"].to_numpy()
            y = df["free_energy_kcalmol"].to_numpy()
            mask = (x >= x_min) & (x <= x_max)
            ax_fel.plot(x[mask], y[mask] - np.nanmin(y[mask]), lw=1.2, alpha=0.55, label=f"{name} FEL")
            if len(common_grid):
                interpolated.append(np.interp(common_grid, x, y))
        if len(common_grid) and interpolated:
            mean = np.mean(np.vstack(interpolated), axis=0)
            mean = mean - np.nanmin(mean)
            ax_fel.plot(common_grid, mean, color="crimson", lw=2.6, label="mean FEL")
    else:
        ax_fel.set_yticks([])
    ax_hist.axvline(0.0, color="crimson", ls="--", lw=2.0, label="EVB TS / gap = 0")
    ax_hist.axvspan(-250.0, 250.0, color="crimson", alpha=0.08, label="+/-250 kcal TS band")
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.set_xlabel("EVB shifted gap (kcal/mol)")
    ax_hist.set_ylabel("Sample count")
    ax_fel.set_ylabel("Free energy estimate (kcal/mol)" if fel else "")
    ax_hist.set_title("EVB-Gap Metadynamics: Sampled Histogram" + (" and FEL" if fel else " (partial run)"))
    lines1, labels1 = ax_hist.get_legend_handles_labels()
    lines2, labels2 = ax_fel.get_legend_handles_labels()
    ax_hist.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2, loc="upper right")
    ax_hist.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_gap_trace(obs: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, df in obs.items():
        ax.plot(df["time_ps"], df["gap_shifted_kJmol"] * KCAL_PER_KJ, lw=1.1, label=name)
    ax.axhline(0.0, color="crimson", ls="--", lw=2, label="EVB TS / gap = 0")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("EVB shifted gap (kcal/mol)")
    ax.set_title("Gap Sampling Trace")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_weights(obs: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, df in obs.items():
        ax.scatter(df["gap_shifted_kJmol"] * KCAL_PER_KJ, df["w2"], s=14, alpha=0.55, label=f"{name} w2")
    ax.axvline(0.0, color="crimson", ls="--", lw=2, label="EVB TS / gap = 0")
    ax.axhline(0.5, color="black", ls=":", lw=1.5, label="50/50 mixing")
    ax.set_xlabel("EVB shifted gap (kcal/mol)")
    ax.set_ylabel("State-2 EVB weight")
    ax.set_title("EVB Mixing Along the Biased Gap")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_evb_energy_scatter(obs: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_eevb = np.concatenate([df["Eevb_kJmol"].to_numpy() for df in obs.values()])
    e0 = np.nanmin(all_eevb)
    for name, df in obs.items():
        x = df["gap_shifted_kJmol"] * KCAL_PER_KJ
        y = (df["Eevb_kJmol"] - e0) * KCAL_PER_KJ
        ax.scatter(x, y, s=16, alpha=0.55, label=name)
    ax.axvline(0.0, color="crimson", ls="--", lw=2, label="EVB TS / gap = 0")
    ax.set_xlabel("EVB shifted gap (kcal/mol)")
    ax.set_ylabel("EVB potential energy relative to sampled minimum (kcal/mol)")
    ax.set_title("Sampled EVB Energies vs Gap")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_geometry(obs: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, df in obs.items():
        axes[0].plot(df["time_ps"], df["rc_proton_transfer_rc_nm"] * 10.0, lw=1.1, label=name)
        axes[1].scatter(df["distance_donor_h_nm"] * 10.0, df["distance_h_acceptor_nm"] * 10.0, s=14, alpha=0.55, label=name)
    axes[0].axhline(0.0, color="crimson", ls="--", lw=1.8, label="geometry crossing")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("d(donor-H) - d(H-acceptor) (A)")
    axes[0].set_title("Proton-Transfer Coordinate")
    axes[1].plot([0.8, 3.5], [0.8, 3.5], color="crimson", ls="--", lw=1.8, label="equal distances")
    axes[1].set_xlabel("donor-H distance (A)")
    axes[1].set_ylabel("H-acceptor distance (A)")
    axes[1].set_title("Bond Geometry")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _summarize(obs: dict[str, pd.DataFrame], fel: dict[str, pd.DataFrame], convergence: dict[str, dict]) -> dict[str, object]:
    reps = {}
    for name, df in obs.items():
        gap = df["gap_shifted_kJmol"]
        w2 = df["w2"]
        rc = df["rc_proton_transfer_rc_nm"]
        reps[name] = {
            "n_samples": int(len(df)),
            "last_step": int(df["step"].max()),
            "last_time_ps": float(df["time_ps"].max()),
            "gap_min_kcal_mol": float(gap.min() * KCAL_PER_KJ),
            "gap_max_kcal_mol": float(gap.max() * KCAL_PER_KJ),
            "gap_mean_kcal_mol": float(gap.mean() * KCAL_PER_KJ),
            "samples_within_250_kcal_of_ts": int((gap.abs() <= 250.0 / KCAL_PER_KJ).sum()),
            "samples_within_1000_kcal_of_ts": int((gap.abs() <= 1000.0 / KCAL_PER_KJ).sum()),
            "mixed_weight_samples_w2_0p2_to_0p8": int(((w2 >= 0.2) & (w2 <= 0.8)).sum()),
            "ts_mixed_samples_250kcal_w2_0p2_to_0p8": int(
                ((gap.abs() <= 250.0 / KCAL_PER_KJ) & (w2 >= 0.2) & (w2 <= 0.8)).sum()
            ),
            "max_w2": float(w2.max()),
            "mean_w2": float(w2.mean()),
            "pt_rc_min_angstrom": float(rc.min() * 10.0),
            "pt_rc_max_angstrom": float(rc.max() * 10.0),
            "converged_by_internal_criterion": bool(convergence.get(name, {}).get("stopped_by_convergence", False)),
            "convergence_step": convergence.get(name, {}).get("current_step"),
        }
    all_converged = all(item["converged_by_internal_criterion"] for item in reps.values())
    enough_ts = all(item["samples_within_250_kcal_of_ts"] >= 20 for item in reps.values())
    if all_converged and enough_ts:
        assessment = "The calculation reached the configured convergence criterion and sampled the EVB mixing region."
    elif all_converged:
        assessment = (
            "The FEL-change criterion was reached, but TS-region sampling remains thin. Treat the FEL as provisional."
        )
    else:
        assessment = (
            "The calculation finished at the maximum step cap without satisfying the configured convergence criterion. "
            "It sampled both EVB basins and some TS/mixing points, but the FEL should not yet be reported as converged."
        )
    return {
        "units": {
            "gap": "kcal/mol",
            "free_energy": "kcal/mol",
            "geometry": "angstrom",
        },
        "ts_definition": "EVB mixing region is shifted gap = 0 kcal/mol; strong mixing would have w2 near 0.5.",
        "replicas": reps,
        "assessment": assessment,
    }


if __name__ == "__main__":
    main()
