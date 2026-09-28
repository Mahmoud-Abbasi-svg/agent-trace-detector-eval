"""Generate the v0.4 figures directly from the verified revision result files (no hand-typed values).
Each figure is written as PDF (for LaTeX) and PNG (for the Markdown manuscript).
  fig_v04_prevalence   R1: wAUPRC and precision at theta_r80 against attack prevalence    (PO M8_R1, R1_operating_points)
  fig_v04_r3           R3 AUROC against the domain-matched R1 value, per held-out domain   (DS M3)
  fig_v04_coverage     graded tool coverage inside R1: AUROC and FPR at theta_r80          (OOV s2_graded)
  fig_v04_r4           AgentDojo: transfer, within UT and within IT AUROC per detector     (R4D M6 configs, cluster intervals)
  fig_v04_s_thresholds supplement: recall and FPR at theta_r80, R1dm against R3            (DS M3)
Colour-blind-safe Okabe-Ito palette, distinct markers, 8 pt serif text, IEEE widths (3.5 in single, 7.16 in double column).
Run from agent-ids/paper:  python make_figures_v04.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
J = lambda *p: json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))
PO = J("revision", "prevalence_ops_results.json")
DS = J("revision", "domain_shift_results.json")
OOV = J("revision", "oov_mechanism_results.json")
R4D = J("revision", "r4_deep_results.json")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
SINGLE, DOUBLE = 3.5, 7.16
# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
COL = {"LR-A": OI["black"], "LR-B": OI["blue"], "LR-C": OI["orange"], "GRU": OI["purple"], "TFIDF": OI["green"],
       "Markov": OI["sky"], "IForest": OI["vermillion"], "LR-C-drop": OI["orange"], "GRU-drop0.2": OI["purple"]}
MK = {"LR-A": "D", "LR-B": "o", "LR-C": "s", "GRU": "^", "TFIDF": "P", "Markov": "v", "IForest": "X",
      "LR-C-drop": "s", "GRU-drop0.2": "^"}
NAME = {"LR-A": "LR-A (tool identity)", "LR-B": "LR-B (behavioral)", "LR-C": "LR-C (combined)", "GRU": "GRU",
        "TFIDF": "TF-IDF (content)", "Markov": "Markov", "IForest": "IForest", "LR-C-drop": "LR-C-drop (tool-block dropout)",
        "GRU-drop0.2": "GRU-drop0.2 (token dropout)"}
DOMS = ["medical", "banking", "web", "email", "coding"]
TH = r"$\theta_{r80}$"


def tri(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    p = x.get("point", x.get("mean"))
    if "lo" in x:
        return (p, x["lo"], x["hi"])
    return (p, x["ci"][0], x["ci"][1])


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=300)
    plt.close(fig)


def eb(ax, x, vals, det, hollow=False, ls="-", ms=3.6, label=None, lw=0.9, zorder=3):
    v = np.array([tri(a) for a in vals], dtype=float)
    ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=MK[det], ls=ls, color=COL[det], ms=ms,
                mfc="white" if hollow else COL[det], mec=COL[det], mew=0.8, capsize=1.4, lw=lw, elinewidth=0.6,
                label=label, zorder=zorder)


# ---------------------------------------------------------------- fig_v04_prevalence
def fig_prevalence():
    PK = ["pi01", "pi02", "pi05", "pi10"]; PV = np.array([1, 2, 5, 10], dtype=float)
    M8 = PO["M8_R1"]; OP = PO["R1_operating_points"]["detectors"]
    dets = ["LR-B", "LR-C", "GRU", "TFIDF", "Markov", "IForest"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.55))
    for k, d in enumerate(dets):
        x = PV * (1 + (k - 2.5) * 0.028)
        eb(axs[0], x, [M8["detectors"][d]["wAUPRC"][p] for p in PK], d, label=NAME[d])
        pr = []
        for p in PK:
            q = OP[d]["t_r80"]["precision_" + p]
            if q["point"] == 1.0 and "lower_bound_if_FPR_at_CP_upper" in q:      # zero false positives: CP lower bound
                pr.append((1.0, q["lower_bound_if_FPR_at_CP_upper"], 1.0))
            else:
                pr.append(tri(q))
        eb(axs[1], x, pr, d)
    ch = M8["chance_wAUPRC_random_scorer"]
    axs[0].plot(PV, [ch[p]["mean"] for p in PK], color="0.35", ls=":", lw=1.0, zorder=1)
    axs[1].plot(PV, PV / 100.0, color="0.35", ls=":", lw=1.0, zorder=1)
    for ax, lab, t in zip(axs, ["wAUPRC", "Precision at %s" % TH], ["(a) Prevalence-weighted AUPRC", "(b) Precision at the training threshold %s" % TH]):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(PV); ax.set_xticklabels(["1%", "2%", "5%", "10%"]); ax.minorticks_off()
        ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
        ax.set_ylim(0.008, 1.25); ax.set_xlim(0.85, 11.8)
        ax.set_xlabel("Attack prevalence"); ax.set_ylabel(lab); ax.set_title(t, loc="left")
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    h, l = axs[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="0.35", ls=":", lw=1.0)); l.append("Random scorer (chance)")
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save(fig, "fig_v04_prevalence")


# ---------------------------------------------------------------- dot plot helper (R1dm hollow against R3 filled)
def dm_dot(ax, dets, metric, width=0.72):
    M3 = DS["M3"]; n = len(dets)
    for k, d in enumerate(dets):
        off = (k - (n - 1) / 2) * width / n
        for i, dom in enumerate(DOMS):
            a = tri(M3["R1_domain_matched"][dom][d][metric]); b = tri(M3["R3"][dom][d][metric])
            xa, xb = i + off - 0.045, i + off + 0.045
            ax.plot([xa, xb], [a[0], b[0]], color=COL[d], lw=0.7, zorder=2)
            ax.errorbar([xa], [a[0]], yerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt=MK[d], color=COL[d], mfc="white", mec=COL[d],
                        ms=3.4, mew=0.8, capsize=1.2, elinewidth=0.6, zorder=3)
            ax.errorbar([xb], [b[0]], yerr=[[b[0] - b[1]], [b[2] - b[0]]], fmt=MK[d], color=COL[d], ms=3.4, mew=0.8,
                        capsize=1.2, elinewidth=0.6, zorder=4)
    for i in range(1, len(DOMS)):
        ax.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)
    ax.set_xticks(range(len(DOMS))); ax.set_xticklabels(DOMS); ax.set_xlim(-0.5, len(DOMS) - 0.5)
    ax.tick_params(axis="x", length=0)


def dm_legend(fig, dets, ncol, y=1.0, extra=None):
    h = [Line2D([], [], color=COL[d], marker=MK[d], ls="", ms=4) for d in dets]
    l = [NAME[d] for d in dets]
    h += [Line2D([], [], color="0.3", marker="o", mfc="white", ls="", ms=4), Line2D([], [], color="0.3", marker="o", ls="", ms=4)]
    l += ["R1dm, same domain in distribution (hollow)", "R3, domain held out (filled)"]
    if extra:
        h.append(extra[0]); l.append(extra[1])
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), handletextpad=0.2, columnspacing=1.0)


# ---------------------------------------------------------------- fig_v04_r3
def fig_r3():
    dets = ["LR-B", "LR-C", "GRU", "TFIDF"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.45))
    dm_dot(ax, dets, "AUROC")
    ax.axhline(0.5, color="0.35", ls=":", lw=1.0, zorder=1)
    ax.text(len(DOMS) - 0.52, 0.505, "chance", ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.set_ylim(0.47, 1.02); ax.set_ylabel("AUROC (pool E)")
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    dm_legend(fig, dets, ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig_v04_r3")


# ---------------------------------------------------------------- fig_v04_coverage
def fig_coverage():
    L = OOV["s2_graded"]["levels"]; lv = ["1.0", "0.75", "0.5", "0.25", "0.0"]
    cov = np.array([L[x]["coverage_mean"] for x in lv])
    dets = ["LR-B", "LR-C", "LR-C-drop", "GRU", "GRU-drop0.2"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for k, d in enumerate(dets):
        dash = d.endswith("drop") or d.endswith("0.2")
        x = cov + (k - 2) * 0.008
        for ax, m in zip(axs, ["AUROC", "FPR@t_r80"]):
            eb(ax, x, [L[v]["detectors"][d][m] for v in lv], d, hollow=dash, ls="--" if dash else "-",
               label=NAME[d] if ax is axs[0] else None)
    axs[0].axhline(0.5, color="0.35", ls=":", lw=1.0, zorder=1)
    axs[0].text(0.99, 0.505, "chance", ha="right", va="bottom", fontsize=6.5, color="0.35", transform=axs[0].get_yaxis_transform())
    axs[0].set_ylim(0.47, 1.0); axs[1].set_ylim(0, 0.8)
    for ax, lab, t in zip(axs, ["AUROC (R1 pool E)", "FPR at %s" % TH], ["(a) Ranking", "(b) False-positive rate at the training threshold"]):
        ax.set_xlim(1.04, -0.04)                          # coverage falls from left to right
        ax.set_xticks([1.0, 0.775, 0.483, 0.225, 0.0]); ax.set_xticklabels(["1.00", "0.78", "0.48", "0.23", "0"])
        ax.set_xlabel("Test-time tool coverage inside R1 (achieved)"); ax.set_ylabel(lab); ax.set_title(t, loc="left")
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save(fig, "fig_v04_coverage")


# ---------------------------------------------------------------- fig_v04_r4
def fig_r4():
    cf = R4D["M6"]["configs"]
    dets = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
    series = [("P0_UT", "transfer", "Transfer (AgentDrift to AgentDojo)", OI["vermillion"], "s"),
              ("P0_UT", "within", "Within AgentDojo, unseen user tasks (UT)", OI["blue"], "o"),
              ("P0_IT", "within", "Within AgentDojo, unseen attack goals (IT)", OI["green"], "^")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.75))
    y0 = np.arange(len(dets))[::-1]
    for j, (cfg, side, lab, col, mk) in enumerate(series):
        c = cf[cfg]
        for i, d in enumerate(dets):
            key = "%s:%s" % (side, d)
            p = c["point"][key]["AUROC"]; _, lo, hi = c["cluster"]["ci"][key]["AUROC"]
            y = y0[i] + (1 - j) * 0.22
            ax.errorbar([p], [y], xerr=[[p - lo], [hi - p]], fmt=mk, color=col, ms=4, mew=0.8, capsize=1.5, elinewidth=0.8,
                        label=lab if i == 0 else None, zorder=3)
    ax.axvline(0.5, color="0.35", ls=":", lw=1.0, zorder=1)
    ax.text(0.505, len(dets) - 0.45, "chance", ha="left", va="bottom", fontsize=6.5, color="0.35")
    ax.set_yticks(y0); ax.set_yticklabels([NAME[d] for d in dets])
    ax.set_xlim(0.3, 1.02); ax.set_ylim(-0.5, len(dets) - 0.2)
    ax.set_xlabel("AUROC on AgentDojo P0 (cluster 95% interval)")
    ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    for i in range(1, len(dets)):
        ax.axhline(i - 0.5, color="0.9", lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.55, 1.0), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    save(fig, "fig_v04_r4")


# ---------------------------------------------------------------- fig_v04_s_thresholds (supplement)
def fig_s_thresholds():
    dets = ["LR-B", "LR-C", "GRU", "TFIDF"]
    fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 4.3), sharex=True)
    for ax, m, lab, t in zip(axs, ["recall@t_r80", "FPR@t_r80"], ["Recall at %s" % TH, "FPR at %s" % TH], ["(a) Recall", "(b) False-positive rate"]):
        dm_dot(ax, dets, m)
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel(lab); ax.set_title(t, loc="left")
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    axs[0].axhline(0.8, color="0.35", ls="--", lw=0.8, zorder=1)
    dm_legend(fig, dets, ncol=4, extra=(Line2D([], [], color="0.35", ls="--", lw=0.8), "training recall target 0.80 (panel a)"))
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.0)
    save(fig, "fig_v04_s_thresholds")


if __name__ == "__main__":
    fig_prevalence(); fig_r3(); fig_coverage(); fig_r4(); fig_s_thresholds()
    print("wrote fig_v04_prevalence, fig_v04_r3, fig_v04_coverage, fig_v04_r4, fig_v04_s_thresholds (.pdf + .png)")
