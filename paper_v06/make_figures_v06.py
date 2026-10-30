"""Figures of the short paper (v0.6), generated from the verified result files. No value is typed by hand.

  fig_v06_overview    Fig. 1  detection setting and evaluation design (coverage values from revision/cache/linear/_validation)
  fig_v06_prevalence  Fig. 2  R1: wAUPRC and precision at theta_r80 against attack prevalence
                              trained detectors from revision/prevalence_ops_results.json (as fig_v04_prevalence),
                              MON-L from paper/tables_monitor/v04_prevalence.md (the rows of Table III)
  fig_v06_coverage    Fig. 3  copy of paper/fig_v04_coverage.pdf (unchanged)
  fig_v06_transfer    Fig. 4  AgentDojo P0: transfer and within-AgentDojo (UT) AUROC, from paper/tables_monitor/v04_r4.md
                              (the rows of Table V, cluster intervals)

Style follows paper/make_figures_v04.py: serif 8/7/6 pt, Okabe-Ito colours, IEEE widths.
Run from agent-ids/paper_v06:  python make_figures_v06.py
"""
import json, os, re, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
J = lambda *p: json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))
PO = J("revision", "prevalence_ops_results.json")
TABLES = os.path.join(ROOT, "paper", "tables_monitor")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
SINGLE, DOUBLE = 3.5, 7.16
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
COL = {"LR-B": OI["blue"], "LR-C": OI["orange"], "GRU": OI["purple"], "TFIDF": OI["green"], "MON-L": OI["black"]}
MK = {"LR-B": "o", "LR-C": "s", "GRU": "^", "TFIDF": "P", "MON-L": "D"}
NAME = {"LR-B": "LR-B (behavioral)", "LR-C": "LR-C (combined)", "GRU": "GRU (sequence)", "TFIDF": "TF-IDF (content)",
        "MON-L": "MON-L (released monitor)"}
TH = r"$\theta_{r80}$"
NUM = r"[+\-\u2212]?\d+\.\d+"


def ci(cell):
    """'0.048 [0.040, 0.067]' -> (0.048, 0.040, 0.067); '0.930' -> (0.930, None, None)."""
    v = [float(x.replace("\u2212", "-")) for x in re.findall(NUM, cell)]
    assert v, cell
    return (v[0], v[1], v[2]) if len(v) >= 3 else (v[0], None, None)


def md_rows(name):
    lines = [l for l in open(os.path.join(TABLES, name), encoding="utf-8").read().splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {}
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows[cells[0]] = dict(zip(head, cells))
    return head, rows


def tri(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    p = x.get("point", x.get("mean"))
    if "lo" in x:
        return (p, x["lo"], x["hi"])
    return (p, x["ci"][0], x["ci"][1])


def eb(ax, x, vals, det, hollow=False, ls="-", ms=3.6, label=None, lw=0.9, zorder=3):
    v = np.array([tri(a) for a in vals], dtype=float)
    ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=MK[det], ls=ls, color=COL[det], ms=ms,
                mfc="white" if hollow else COL[det], mec=COL[det], mew=0.8, capsize=1.4, lw=lw, elinewidth=0.6,
                label=label, zorder=zorder)


def check_text(fig):
    r_ = fig.canvas.get_renderer()
    t = [(a, a.get_window_extent(r_)) for a in fig.findobj(matplotlib.text.Text) if a.get_text().strip() and a.get_visible()]
    ov = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(t) for b, bb in t[i + 1:] if ba.overlaps(bb)]
    out = [a.get_text() for a, b in t if not fig.bbox.contains(b.x0, b.y0) or not fig.bbox.contains(b.x1, b.y1)]
    assert not ov and not out, (ov, out)


def save(fig, name, check=True):
    if check:
        fig.canvas.draw(); check_text(fig)
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2, prevalence
def fig_prevalence():
    PK = ["pi01", "pi02", "pi05", "pi10"]; PV = np.array([1, 2, 5, 10], dtype=float)
    M8 = PO["M8_R1"]; OP = PO["R1_operating_points"]["detectors"]
    dets = ["LR-B", "LR-C", "GRU", "TFIDF"]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.55))
    for k, d in enumerate(dets):
        x = PV * (1 + (k - 2.0) * 0.028)
        eb(axs[0], x, [M8["detectors"][d]["wAUPRC"][p] for p in PK], d, label=NAME[d])
        pr = []
        for p in PK:
            q = OP[d]["t_r80"]["precision_" + p]
            if q["point"] == 1.0 and "lower_bound_if_FPR_at_CP_upper" in q:      # zero false positives: CP lower bound
                pr.append((1.0, q["lower_bound_if_FPR_at_CP_upper"], 1.0))
            else:
                pr.append(tri(q))
        eb(axs[1], x, pr, d)
    # MON-L, the released monitor, from the rows of Table III
    _, rows = md_rows("v04_prevalence.md")
    mon = rows["MON-L at θr80"]
    xm = PV * (1 + 2.0 * 0.028)
    eb(axs[0], xm, [ci(mon["wAUPRC %d%%" % p]) for p in (1, 2, 5, 10)], "MON-L", label=NAME["MON-L"])
    xm2 = np.array([2.0, 10.0]) * (1 + 2.0 * 0.028)
    eb(axs[1], xm2, [ci(mon["Precision at θr80, 2%"]), ci(mon["Precision at θr80, 10%"])], "MON-L", ls="none")
    ch = M8["chance_wAUPRC_random_scorer"]
    axs[0].plot(PV, [ch[p]["mean"] for p in PK], color="0.35", ls=":", lw=1.0, zorder=1)
    axs[1].plot(PV, PV / 100.0, color="0.35", ls=":", lw=1.0, zorder=1)
    for ax, lab, t in zip(axs, ["wAUPRC", "Precision at %s" % TH],
                          ["(a) Ranking quality falls with prevalence", "(b) At the training threshold, most alerts are false"]):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(PV); ax.set_xticklabels(["1%", "2%", "5%", "10%"]); ax.minorticks_off()
        ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
        ax.set_ylim(0.008, 1.25); ax.set_xlim(0.85, 11.8)
        ax.set_xlabel("Attack prevalence"); ax.set_ylabel(lab); ax.set_title(t, loc="left")
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    h, l = axs[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="0.35", ls=":", lw=1.0)); l.append("Chance")
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save(fig, "fig_v06_prevalence")


# ------------------------------------------------------------------ Fig. 4, transfer
def fig_transfer():
    _, rows = md_rows("v04_r4.md")
    dets = [("LR-B (behavioral)", "LR-B (behavioral)"), ("LR-C (combined)", "LR-C (combined)"), ("GRU", "GRU (sequence)"),
            ("SE (reimplementation)", "SE (step embeddings)"), ("TF-IDF (content)", "TF-IDF (content)"),
            ("MON-L (Llama 3.3 70B w4a16)", "MON-L (monitor, Llama)"), ("MON-A (Claude Sonnet 5.5)", "MON-A (monitor, Claude)")]
    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    n = len(dets); y0 = np.arange(n)[::-1].astype(float)
    tr_lab = "Trained on AgentDrift, tested on AgentDojo"; wi_lab = "Trained within AgentDojo (unseen user tasks)"
    mon_lab = "Not trained (released monitor)"
    used = set()
    for i, (key, lab) in enumerate(dets):
        r = rows[key]
        p, lo, hi = ci(r["Transfer AUROC (P0)"])
        mon = key.startswith("MON")
        col, mk, ll = (OI["black"], "D", mon_lab) if mon else (OI["vermillion"], "s", tr_lab)
        yy = y0[i] + (0 if mon else 0.16)
        ax.errorbar([p], [yy], xerr=[[p - lo], [hi - p]], fmt=mk, color=col, ms=4, mew=0.8, capsize=1.5, elinewidth=0.8,
                    label=None if ll in used else ll, zorder=3); used.add(ll)
        if not mon:
            p, lo, hi = ci(r["Within-AgentDojo AUROC (UT)"])
            ax.errorbar([p], [y0[i] - 0.16], xerr=[[p - lo], [hi - p]], fmt="o", color=OI["blue"], ms=4, mew=0.8, capsize=1.5,
                        elinewidth=0.8, label=None if wi_lab in used else wi_lab, zorder=3); used.add(wi_lab)
    ax.axvline(0.5, color="0.35", ls=":", lw=1.0, zorder=1)
    ax.text(0.505, n - 0.45, "chance", ha="left", va="bottom", fontsize=6, color="0.35")
    ax.set_yticks(y0); ax.set_yticklabels([lab for _, lab in dets])
    ax.set_xlim(0.3, 1.02); ax.set_ylim(-0.5, n - 0.2)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC on AgentDojo P0 (cluster 95% interval)")
    ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    for i in range(1, n):
        ax.axhline(i - 0.5, color="0.9" if i != 2 else "0.55", lw=0.5 if i != 2 else 0.7, zorder=0)
    ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    save(fig, "fig_v06_transfer")


# ------------------------------------------------------------------ Fig. 1, overview (from paper/make_fig_overview.py)
def fig_overview():
    KAP = {s: J("revision", "cache", "linear", "_validation", s + ".json")["sizes"]["kappa"]
           for s in ["R1", "R1wd_s0", "R1wd_s1", "R1wd_s2", "R1wd_s3", "R1wd_s4", "R2", "R3_medical", "R3_banking", "R3_web",
                     "R3_email", "R3_coding", "R4", "WUT"]}
    assert len({KAP[f"R1wd_s{k}"] for k in range(5)}) == 1
    assert len({KAP[f"R3_{d}"] for d in ["medical", "banking", "web", "email", "coding"]}) == 1
    GREY = "#555555"; LGREY = "#E9E9E9"; DRIFT = "#DCEEF9"; DOJO = "#D5EFE6"; INJ = OI["vermillion"]
    B, M, S = 8, 7, 6
    W, H = 7.16, 3.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y, w, h, fc="white", ec=GREY, lw=0.6, r=0.05, ls="-", z=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

    def arrow(x0, y0, x1, y1, color=GREY, lw=0.8, style="-|>", ls="-", z=2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                     zorder=z, shrinkA=0, shrinkB=0))

    def txt(x, y, s, size=B, ha="center", va="center", **k):
        ax.text(x, y, s, fontsize=size, ha=ha, va=va, zorder=5, **k)

    # panel (a)
    ax.text(0.06, H - 0.12, "a", fontsize=10, fontweight="bold", va="top")
    txt(0.26, H - 0.17, "Detection is a post hoc audit of finished sessions", size=B, ha="left")
    y0 = 2.15
    box(0.12, y0, 0.62, 0.36); txt(0.43, y0 + 0.18, "User task", size=M)
    box(0.98, y0, 0.62, 0.36, fc=LGREY); txt(1.29, y0 + 0.18, "LLM agent", size=M)
    box(1.84, y0, 0.52, 0.36); txt(2.10, y0 + 0.18, "Tools", size=M)
    arrow(0.74, y0 + 0.18, 0.98, y0 + 0.18)
    arrow(1.60, y0 + 0.25, 1.84, y0 + 0.25)
    arrow(1.84, y0 + 0.10, 1.60, y0 + 0.10, color=INJ, lw=1.0)
    txt(1.72, y0 + 0.36, "call", size=S, color=GREY)
    txt(2.10, y0 - 0.12, "injected observation", size=S, color=INJ)
    txt(2.10, y0 - 0.23, "(the attacker writes it)", size=S, color=INJ)
    yt = 1.38
    txt(0.12, yt + 0.52, r"Recorded trace $\tau=(s_1,\ldots,s_m)$, each step", size=S, ha="left")
    txt(0.12, yt + 0.40, r"$s_t=$ (tool, arguments, observation)", size=S, ha="left")
    xs = [0.12, 0.62, 1.12, 1.94]; labs = [r"$s_1$", r"$s_2$", r"$s_3$", r"$s_m$"]
    for k, (x, lab) in enumerate(zip(xs, labs)):
        inj = k == 2
        box(x, yt, 0.42, 0.28, fc="#FBE3D3" if inj else "white", ec=INJ if inj else GREY, lw=0.8 if inj else 0.6)
        txt(x + 0.21, yt + 0.14, lab, size=M, color=INJ if inj else "black")
    txt(1.745, yt + 0.14, r"$\cdots$", size=M)
    ax.plot([0.12, 2.36], [yt - 0.08, yt - 0.08], color=GREY, lw=0.6)
    arrow(1.29, y0, 1.29, yt + 0.62, ls=(0, (2, 1.5)))
    yd = 0.42
    box(0.12, yd, 1.05, 0.62, fc=LGREY)
    txt(0.645, yd + 0.47, r"Detector score $f(\tau)$", size=M)
    txt(0.645, yd + 0.31, "Trained: LR-B, LR-C, GRU,", size=S)
    txt(0.645, yd + 0.19, "SE, TF-IDF", size=S)
    txt(0.645, yd + 0.07, "Untrained: MON-L, MON-A", size=S)
    box(1.40, yd, 0.96, 0.62)
    txt(1.88, yd + 0.45, r"Alert if $f(\tau)\geq\theta$", size=M)
    txt(1.88, yd + 0.27, r"$\theta=\theta_{r80}$, fixed on", size=S)
    txt(1.88, yd + 0.14, "training data", size=S)
    arrow(0.645, yt - 0.08, 0.645, yd + 0.62)
    arrow(1.17, yd + 0.31, 1.40, yd + 0.31)
    txt(1.24, 0.24, "an alert sends the session to an analyst", size=S, color=GREY)
    ax.plot([2.53, 2.53], [0.12, H - 0.10], color="#BBBBBB", lw=0.5)

    # panel (b)
    ax.text(2.62, H - 0.12, "b", fontsize=10, fontweight="bold", va="top")
    txt(2.82, H - 0.17, "Graded shifts, with the training threshold carried to every test set", size=B, ha="left")
    cols = [("R1", ("Nothing", "(stratified split)"), "AgentDrift", "R1", "Table III, Fig. 2"),
            ("R1-wd", ("Worlds", "(5 split seeds)"), "AgentDrift", "R1wd_s0", "Section V-A"),
            ("R2", ("Tasks", "(35 unseen tasks)"), "AgentDrift", "R2", "Section V-A"),
            ("R3", ("A domain and", "all its tools"), "AgentDrift", "R3_medical", "Table IV"),
            ("R4", ("The benchmark", "(train AgentDrift)"), "AgentDojo", "R4", "Table V, Fig. 4")]
    cx0, cw, gap, ytop, ch = 2.66, 0.78, 0.115, 1.22, 1.24
    for k, (name, hold, data, key, rep) in enumerate(cols):
        x = cx0 + k * (cw + gap)
        box(x, ytop, cw, ch, fc=DOJO if data == "AgentDojo" else DRIFT)
        txt(x + cw / 2, ytop + ch - 0.14, name, size=B, fontweight="bold")
        txt(x + cw / 2, ytop + ch - 0.33, "Holds out", size=S, color=GREY)
        txt(x + cw / 2, ytop + ch - 0.47, hold[0], size=S)
        txt(x + cw / 2, ytop + ch - 0.58, hold[1], size=S)
        txt(x + cw / 2, ytop + ch - 0.74, "Test on " + data, size=S, color=GREY)
        kv = KAP[key]; bx, by, bw, bh = x + 0.09, ytop + 0.13, cw - 0.18, 0.09
        ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", ec=GREY, lw=0.5, zorder=3))
        if kv > 0:
            ax.add_patch(Rectangle((bx, by), bw * kv, bh, fc=OI["blue"], ec="none", zorder=4))
        txt(x + cw / 2, ytop + 0.32, rf"$\kappa={kv:.3f}$", size=S)
        txt(x + cw / 2, ytop - 0.10, rep, size=S, color=GREY)
        if k < len(cols) - 1:
            arrow(x + cw + 0.01, ytop + ch / 2, x + cw + gap - 0.01, ytop + ch / 2)
    xa0, xa1 = cx0, cx0 + 5 * cw + 4 * gap
    arrow(xa0, ytop + ch + 0.10, xa1, ytop + ch + 0.10, color=GREY, lw=0.6)
    txt(xa1, ytop + ch + 0.19, "more of the deployment changes", size=S, color=GREY, ha="right")
    yc, hc = 0.30, 0.55
    x_r1 = cx0; x_r3 = cx0 + 3 * (cw + gap); x_r4 = cx0 + 4 * (cw + gap)
    box(x_r1, yc, 3 * (cw + gap) - gap, hc, fc="white", ls=(0, (3, 2)))
    txt(x_r1 + 0.08, yc + hc - 0.14, "Graded coverage inside R1 (Fig. 3)", size=M, ha="left")
    txt(x_r1 + 0.08, yc + 0.22, r"Tool names masked at test time, $\kappa$ from 1.000 to 0.000,", size=S, ha="left")
    txt(x_r1 + 0.08, yc + 0.10, "with no change of domain", size=S, ha="left")
    box(x_r3, yc, 2 * cw + gap, hc, fc=DOJO, ls=(0, (3, 2)))
    txt(x_r3 + 0.08, yc + hc - 0.14, "Within-AgentDojo control (Table V)", size=M, ha="left")
    txt(x_r3 + 0.08, yc + 0.22, "Train and test inside AgentDojo,", size=S, ha="left")
    txt(x_r3 + 0.08, yc + 0.10, rf"unseen user tasks ($\kappa={KAP['WUT']:.3f}$)", size=S, ha="left")
    arrow(x_r4 + cw / 2, ytop - 0.20, x_r4 + cw / 2, yc + hc, ls=(0, (2, 1.5)))
    lx = 2.66; ly = 0.10
    ax.add_patch(Rectangle((lx, ly - 0.03), 0.16, 0.06, fc=OI["blue"], ec="none"))
    txt(lx + 0.21, ly, r"coverage $\kappa$: share of test tool calls seen in training", size=S, ha="left", color=GREY)
    ax.add_patch(Rectangle((lx + 2.55, ly - 0.04), 0.16, 0.08, fc=DRIFT, ec=GREY, lw=0.4))
    txt(lx + 2.76, ly, "AgentDrift", size=S, ha="left", color=GREY)
    ax.add_patch(Rectangle((lx + 3.40, ly - 0.04), 0.16, 0.08, fc=DOJO, ec=GREY, lw=0.4))
    txt(lx + 3.61, ly, "AgentDojo", size=S, ha="left", color=GREY)
    save(fig, "fig_v06_overview")
    return KAP


if __name__ == "__main__":
    kap = fig_overview()
    fig_prevalence()
    fig_transfer()
    for ext in ("pdf", "png"):
        shutil.copy(os.path.join(ROOT, "paper", "fig_v04_coverage." + ext), os.path.join(HERE, "fig_v06_coverage." + ext))
    print("wrote fig_v06_overview, fig_v06_prevalence, fig_v06_transfer, fig_v06_coverage;",
          "kappa R1 %.3f R3 %.3f R4 %.3f WUT %.3f" % (kap["R1"], kap["R3_medical"], kap["R4"], kap["WUT"]))
