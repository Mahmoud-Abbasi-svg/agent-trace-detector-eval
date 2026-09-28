"""Emit the v0.4 results tables directly from the verified revision result files.
Every number in these tables is read from JSON; nothing is hand-typed. Constants that appear in cells are
definitions only (chance AUROC 0.5, chance precision = prevalence).

Sources (one stream per table, as in revision/RESULTS_SUMMARY.md Part 3):
  PO  revision/prevalence_ops_results.json     DS  revision/domain_shift_results.json
  OOV revision/oov_mechanism_results.json      CD  revision/content_detector_results.json
  FO  revision/features_obslen_results.json    R4D revision/r4_deep_results.json
  DG  revision/diagnostics_results.json
  plus revision/cache/linear/_validation/<setting>.json (tool-occurrence coverage kappa),
  archive_v0.3/gru_r3_diagnostic_results.json (v0.3 post hoc GRU diagnostic) and
  agentdojo_extract_manifest.json (short-run filter exclusions).

Writes tables/<name>.tex (header rows, \\midrule and body rows; md2tex adds \\toprule, \\bottomrule and the
column spec) and tables/<name>.md, plus tables/v04_colspecs.json with the (env, colspec, size) of every table.
Main-paper tables are named v04_*, supplement tables s_*.
Run from agent-ids/paper:  python gen_tables_v04.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "tables"); os.makedirs(OUT, exist_ok=True)


def J(*p):
    return json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))


PO = J("revision", "prevalence_ops_results.json")
DS = J("revision", "domain_shift_results.json")
OOV = J("revision", "oov_mechanism_results.json")
CD = J("revision", "content_detector_results.json")
FO = J("revision", "features_obslen_results.json")
R4D = J("revision", "r4_deep_results.json")
DG = J("revision", "diagnostics_results.json")
GDIAG = J("archive_v0.3", "gru_r3_diagnostic_results.json")
MANI = J("agentdojo_extract_manifest.json")
KAPPA = {}
for _s in ["R1", "R1wd_s0", "R1wd_s1", "R1wd_s2", "R1wd_s3", "R1wd_s4", "R2", "R3_medical", "R3_banking", "R3_web",
           "R3_email", "R3_coding", "R4", "WUT", "WUT_F", "WIT", "LOSO_banking", "LOSO_slack", "LOSO_travel", "LOSO_workspace"]:
    KAPPA[_s] = J("revision", "cache", "linear", "_validation", _s + ".json")["sizes"]["kappa"]

DOMS = ["medical", "banking", "web", "email", "coding"]
SUITES = ["banking", "slack", "travel", "workspace"]
MODELS = ["gemini-1.5-pro-002", "gemini-2.0-flash-001", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"]
PK = ["pi01", "pi02", "pi05", "pi10"]
PVAL = {"pi01": 0.01, "pi02": 0.02, "pi05": 0.05, "pi10": 0.10}      # design prevalences (definitions)
LAB = {"LR-A": "LR-A (tool identity)", "LR-B": "LR-B (behavioral)", "LR-C": "LR-C (combined)", "GRU": "GRU",
       "TFIDF": "TF-IDF (content)", "Markov": "Markov", "IForest": "IForest"}
HT = {"LR-A": r"\makecell{LR-A\\(tool identity)}", "LR-B": r"\makecell{LR-B\\(behavioral)}",
      "LR-C": r"\makecell{LR-C\\(combined)}", "GRU": "GRU", "TFIDF": r"\makecell{TF-IDF\\(content)}",
      "Markov": "Markov", "IForest": "IForest"}
TH = r"$\theta_{r80}$"
# display names of the within-AgentDojo designs, as in the manuscript (the result-file keys stay WUT, WUT_F and WIT)
WLAB = {"WUT": "Within UT", "WUT_F": "Within UT-F", "WIT": "Within IT"}
COLSPECS = {}
MINUS = "−"


# ------------------------------------------------------------------ value helpers
def tri(x):
    """Normalise the stream formats to (point, lo, hi)."""
    if isinstance(x, (list, tuple)):
        if len(x) == 3:
            return tuple(x)
        raise ValueError(x)
    if isinstance(x, dict):
        for k in ("point", "delta", "weighted_point", "mean"):
            if k in x:
                p = x[k]; break
        else:
            raise ValueError(x)
        if "lo" in x:
            return (p, x["lo"], x["hi"])
        if "ci" in x:
            return (p, x["ci"][0], x["ci"][1])
        if "min" in x:
            return (p, x["min"], x["max"])
        return (p, None, None)
    return (x, None, None)


def fnum(x, sign=False, dec=3):
    if x == 0:
        x = 0.0
    s = f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"
    if s in ("-0.000", "+0.000") and not sign:
        s = "0.000"
    return s


MTEX = {"*": "*", "†": r"\dagger", "‡": r"\ddagger", "↓": r"\downarrow", "§": r"\S"}


class Cell:
    def __init__(self, v=None, lo=None, hi=None, sign=False, dec=3, mark="", raw=None):
        self.v, self.lo, self.hi, self.sign, self.dec, self.mark, self.raw = v, lo, hi, sign, dec, mark, raw

    def _m_tex(self):
        return "" if not self.mark else "^{%s}" % "".join(MTEX[c] for c in self.mark)

    def tex(self):
        if self.raw is not None:
            return self.raw[0]
        f = lambda x: fnum(x, self.sign, self.dec)
        if self.lo is None:
            return "$%s%s$" % (f(self.v), self._m_tex())
        return r"\cit{%s%s}{%s}{%s}" % (f(self.v), self._m_tex(), f(self.lo), f(self.hi))

    def md(self):
        if self.raw is not None:
            return self.raw[1]
        f = lambda x: fnum(x, self.sign, self.dec).replace("-", MINUS)
        if self.lo is None:
            return f(self.v) + self.mark
        return f"{f(self.v)}{self.mark} [{f(self.lo)}, {f(self.hi)}]"


def C(x, sign=False, dec=3, mark=""):
    p, lo, hi = tri(x)
    return Cell(p, lo, hi, sign=sign, dec=dec, mark=mark)


def CP(p, lo, hi, sign=False, dec=3, mark=""):
    return Cell(p, lo, hi, sign=sign, dec=dec, mark=mark)


def P(v, sign=False, dec=3, mark=""):
    return Cell(v, sign=sign, dec=dec, mark=mark)


def R(tex, md=None):
    return Cell(raw=(tex, md if md is not None else tex))


NA = R("n/a", "n/a")


def pt(c):
    """Drop the interval of a Cell (point only)."""
    return Cell(c.v, sign=c.sign, dec=c.dec, mark=c.mark) if c.raw is None else c


def stack(vals, dec=3, sign=False):
    """Per-seed values stacked in one cell (LaTeX), slash-separated (Markdown)."""
    return R(r"\makecell{" + r"\\[-1pt]".join("$%s$" % fnum(v, sign, dec) for v in vals) + "}",
             " / ".join(num_md(v, sign, dec) for v in vals))


def CS(n, sep="3.5pt", first="l"):
    """Column spec with a compact inter-column gap."""
    return "@{}%s*{%d}{@{\hspace{%s}}c}@{}" % (first, n, sep)


def two_line(lt):
    """Split 'LR-B (behavioral)...' row labels over two lines in LaTeX."""
    for v in ("LR-A (tool identity)", "LR-B (behavioral)", "LR-C (combined)", "TF-IDF (content)"):
        if lt.startswith(v):
            a, b = v.split(" (", 1)
            return r"\makecell[l]{%s\\(%s%s}" % (a, b, lt[len(v):])
    return lt


def num_md(x, sign=False, dec=3):
    return fnum(x, sign, dec).replace("-", MINUS)


def rng(vals, dec=3, sign=False):
    """min to max of a list of points, as a raw cell (stacked in LaTeX)."""
    a, b = min(vals), max(vals)
    return R(r"\makecell{$%s$\\to $%s$}" % (fnum(a, sign, dec), fnum(b, sign, dec)), "%s to %s" % (num_md(a, sign, dec), num_md(b, sign, dec)))


def slash(vals, dec=3, sign=False):
    return R("$" + r"\,/\,".join(fnum(v, sign, dec) for v in vals) + "$", " / ".join(num_md(v, sign, dec) for v in vals))


def frac(k, n, dec=3):
    return R("%d/%d ($%s$)" % (k, n, fnum(k / n, dec=dec)), "%d/%d (%s)" % (k, n, fnum(k / n, dec=dec)))


# ------------------------------------------------------------------ writer
def mdrow(cells):
    return "| " + " | ".join(cells) + " |"


def write(name, header_tex, header_md, rows, env, colspec, size="\\scriptsize", ncols=None):
    """rows: ('row', label_tex, label_md, [Cell]) | ('mid',) | ('sub', text_tex, text_md)."""
    ncols = ncols or (1 + max(len(r[3]) for r in rows if r[0] == "row"))
    tex = ["% generated by gen_tables_v04.py", header_tex.strip(), r"\midrule"]
    md = [mdrow(header_md), mdrow(["---"] * len(header_md))]
    for r in rows:
        if r[0] == "mid":
            tex.append(r"\midrule")
        elif r[0] == "hdr":
            tex.append(r"\midrule")
            tex.append(" & ".join(r"\textbf{%s}" % c if c else "" for c in r[1]) + r" \\")
            tex.append(r"\midrule")
            md.append(mdrow(["**" + c + "**" if c else "" for c in r[2]]))
        elif r[0] == "sub":
            tex.append(r"\multicolumn{%d}{@{}l}{\emph{%s}} \\" % (ncols, r[1]))
            md.append(mdrow(["*" + r[2] + "*"] + [""] * (ncols - 1)))
        else:
            _, lt, lm, cells = r
            lt = two_line(lt)
            assert len(cells) == ncols - 1, (name, lm, len(cells), ncols)
            tex.append(" & ".join([lt] + [c.tex() for c in cells]) + r" \\")
            md.append(mdrow([lm] + [c.md() for c in cells]))
    assert len(header_md) == ncols, (name, len(header_md), ncols)
    open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    COLSPECS[name] = {"env": env, "colspec": colspec, "size": size, "ncols": ncols}


def row(lt, cells, lm=None):
    return ("row", lt, lm if lm is not None else lt, cells)


def sub(t, m=None):
    return ("sub", t, m if m is not None else t)


MID = ("mid",)


def hdr(tex_cells, md_cells):
    return ("hdr", tex_cells, md_cells)


# ================================================================== MAIN TABLES
# ---------------- v04_prevalence (PO) ----------------
def t_prevalence():
    M8 = PO["M8_R1"]; OP = PO["R1_operating_points"]["detectors"]

    def prec_cell(x):
        if x["point"] == 1.0 and "lower_bound_if_FPR_at_CP_upper" in x:
            b = x["lower_bound_if_FPR_at_CP_upper"]
            return R(r"\makecell{$1.000$\\[-2pt]{\scriptsize$(\ge %s)$}}" % fnum(b), "1.000 (≥ %s)" % fnum(b))
        return C(x)

    def fpr_cell(x):
        if x["point"] == 0.0 and "clopper_pearson_95" in x:
            u = x["clopper_pearson_95"][1]
            return R(r"\makecell{$0.000$\\[-2pt]{\scriptsize$(\le %s)$}}" % fnum(u), "0.000 (CP upper %s)" % fnum(u))
        return C(x)

    rows = []
    for d in ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF", None, "Markov", "IForest"]:
        if d is None:
            rows.append(MID); continue
        m = M8["detectors"][d]; op = OP[d]["t_r80"]
        cells = [C(m["AUROC"])] + [C(m["wAUPRC"][p]) for p in PK]
        cells += [prec_cell(op["precision_pi02"]), prec_cell(op["precision_pi10"]), pt(C(op["TPR"])), pt(fpr_cell(op["FPR"])),
                  pt(C(op["alerts_per_1000_pi02"], dec=0))]
        rows.append(row(LAB[d], cells))
    ch = M8["chance_wAUPRC_random_scorer"]
    rows.append(MID)
    rows.append(row("Random scorer", [P(0.5)] + [CP(ch[p]["mean"], ch[p]["p2.5"], ch[p]["p97.5"]) for p in PK]
                    + [P(PVAL["pi02"]), P(PVAL["pi10"]), NA, NA, NA]))
    ht = (r" & & \multicolumn{4}{c}{wAUPRC at attack prevalence} & \multicolumn{2}{c}{Precision at %s} & & & Alerts \\" % TH + "\n"
          r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}" "\n"
          r"Detector & AUROC & 1\% & 2\% & 5\% & 10\% & 2\% & 10\% & Recall & FPR & \makecell{per 1,000\\(2\%)} \\")
    hm = ["Detector", "AUROC", "wAUPRC 1%", "wAUPRC 2%", "wAUPRC 5%", "wAUPRC 10%", "Precision at θr80, 2%",
          "Precision at θr80, 10%", "Recall at θr80", "FPR at θr80", "Alerts per 1,000 at 2%"]
    write("v04_prevalence", ht, hm, rows, "table*", "@{}l*{10}{c}@{}")


# ---------------- v04_r3 (DS M3) ----------------
def t_r3():
    M3 = DS["M3"]
    rows = []
    # pooled R1 reference row
    cells = []
    for d in ["LR-B", "LR-C", "GRU"]:
        cells += [pt(C(M3["R1_pooled"][d]["AUROC"])), R("")]
    cells += [pt(C(M3["R1_pooled"]["TFIDF"]["AUROC"]))]
    for pr in ["LR-C-minus-LR-B", "GRU-minus-LR-C"]:
        x = M3["R1_pooled_paired"][pr]["AUROC"]
        cells += [C(x, sign=True, mark="*" if x["detected"] else ""), R("")]
    rows.append(row("R1 pooled", cells))
    rows.append(MID)
    for dom in DOMS:
        cells = []
        for d in ["LR-B", "LR-C", "GRU"]:
            dec = M3["R3_minus_R1dm"][d]["AUROC"][dom]
            mk = "↓" if (dec["detected_holm"] and dec["point"] < 0) else ""
            cells += [pt(C(M3["R1_domain_matched"][dom][d]["AUROC"])), pt(C(M3["R3"][dom][d]["AUROC"], mark=mk))]
        cells += [pt(C(M3["R3"][dom]["TFIDF"]["AUROC"]))]
        for pr in ["LR-C-minus-LR-B", "GRU-minus-LR-C"]:
            a = M3["R1_domain_paired"][pr]["AUROC"][dom]; b = M3["R3_paired"][pr]["AUROC"][dom]
            est = M3["reversal"][pr]["AUROC"]["per_fold"][dom]["established"]
            cells += [C(a, sign=True, mark="*" if a["detected_holm"] else ""),
                      C(b, sign=True, mark=("*" if b["detected_holm"] else "") + ("†" if est else ""))]
        rows.append(row(dom, cells))
    ht = (r" & \multicolumn{2}{c}{LR-B (behavioral)} & \multicolumn{2}{c}{LR-C (combined)} & \multicolumn{2}{c}{GRU} & TF-IDF"
          r" & \multicolumn{2}{c}{LR-C $-$ LR-B} & \multicolumn{2}{c}{GRU $-$ LR-C} \\" "\n"
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){9-10}\cmidrule(lr){11-12}" "\n"
          r"Held-out domain & R1dm & R3 & R1dm & R3 & R1dm & R3 & (content) & R1dm & R3 & R1dm & R3 \\")
    hm = ["Held-out domain", "LR-B (behavioral) R1dm", "LR-B (behavioral) R3", "LR-C (combined) R1dm", "LR-C (combined) R3",
          "GRU R1dm", "GRU R3", "TF-IDF (content) R3", "LR-C − LR-B, R1dm", "LR-C − LR-B, R3", "GRU − LR-C, R1dm", "GRU − LR-C, R3"]
    write("v04_r3", ht, hm, rows, "table*", "@{}l*{11}{c}@{}")


# ---------------- v04_thresholds (DS M3 values, M4 status) ----------------
def direction(key):
    st = DS["M4"]["status"][key]["vs_domain_matched_R1"]["holm"]
    toks = set()
    for dom in DOMS:
        if st[dom] != "holds":
            toks |= set(st[dom].split(","))
    if not toks:
        return "holds"
    if "FPR+" in toks and "rec+" in toks:
        return "over-alerting (FPR and recall up)"
    if "rec-" in toks and "FPR+" not in toks:
        fails = [d for d in DOMS if st[d] != "holds"]
        s = "under-alerting (recall down)"
        return s if len(fails) > 1 else "recall down (%s only)" % fails[0]
    return ", ".join(sorted(toks))


def t_thresholds():
    M3 = DS["M3"]; M4 = DS["M4"]; der = DS["derived"]["M4"]
    rows = []
    for d in ["LR-B", "LR-C", "GRU", "TFIDF"]:
        key = d + "@t_r80"
        nf = der["n_folds_failing_vs_dm_holm"][key]
        for metric, mk, dk in [("Recall", "recall@t_r80", "recall"), ("FPR", "FPR@t_r80", "FPR")]:
            cells = []
            for dom in DOMS:
                a = M3["R1_domain_matched"][dom][d][mk]["point"]; b = M3["R3"][dom][d][mk]["point"]
                h = M4["deltas"][key]["vs_domain_matched_R1"][dk][dom]["detected_holm"]
                cells.append(R(r"$%s\,{\rightarrow}\,%s%s$" % (fnum(a), fnum(b), "^{*}" if h else ""),
                               "%s → %s%s" % (fnum(a), fnum(b), "*" if h else "")))
            last = R("%d of 5" % nf) if metric == "Recall" else R(direction(key))
            rows.append(row(LAB[d] if metric == "Recall" else "", [R(metric)] + cells + [last],
                            lm=LAB[d] if metric == "Recall" else LAB[d]))
        if d == "GRU":
            cells = []
            for dom in DOMS:
                n = sum(M4["status"]["GRU#seed%d@t_r80" % s]["vs_domain_matched_R1"]["holm"][dom] != "holds" for s in range(3))
                cells.append(R("%d of 3 seeds fail" % n))
            nseed = min(der["n_folds_failing_vs_dm_holm"]["GRU#seed%d@t_r80" % s] for s in range(3))
            rows.append(row("", [R("Per seed")] + cells + [R("%d of 5 for every seed" % nseed)], lm="GRU"))
        if d != "TFIDF":
            rows.append(MID)
    ht = (r"Detector & Rate & medical & banking & web & email & coding & \makecell{Failing folds (Holm)\\and direction} \\")
    hm = ["Detector", "Rate", "medical", "banking", "web", "email", "coding", "Failing folds (Holm) and direction"]
    write("v04_thresholds", ht, hm, rows, "table*", "@{}ll*{5}{c}>{\\raggedright\\arraybackslash}p{2.6cm}@{}")


# ---------------- v04_oov (OOV) ----------------
def t_oov():
    S1 = OOV["s1_lrc_offset"]["settings"]; S3 = OOV["s3_dropout"]; S4 = OOV["s4_gru_offset"]["settings"]
    RR = S3["reversal_rule_3_4"]

    def off(setting, det):
        g = S4[setting][det]
        lo, hi = g["offset_neg_seedmean_ci"]
        return P(g["mean_over_seeds"]["offset_neg"], sign=True)

    def pair_r1(pr):
        x = RR[pr]["R1_pooled"]
        return C(x, sign=True, mark="*" if x["detected"] else "")

    def pair_r3(pr, dom):
        x = RR[pr]["per_fold"][dom]
        return C(x["R3"], sign=True, mark=("*" if x["R3"]["detected_holm"] else "") + ("†" if x["reversal_holm"] else ""))

    pops = S3["populations"]
    rows = [row("R1", [NA, NA, off("R1", "GRU"), off("R1", "GRU-drop0.2"),
                                       pt(C(pops["R1"]["detectors"]["LR-C-drop"]["AUROC"])), pair_r1("LR-C-drop-minus-LR-B"),
                                       pair_r1("LR-C-drop-minus-LR-C"),
                                       pt(C(pops["R1"]["detectors"]["GRU-drop0.2"]["AUROC"])), pair_r1("GRU-drop0.2-minus-LR-C"),
                                       pair_r1("GRU-drop0.2-minus-GRU")]), MID]
    for dom in DOMS:
        s = "R3_" + dom
        rows.append(row("R3 " + dom, [P(S1[s]["LR-C"]["offset"], sign=True), P(S1[s]["LR-C-drop"]["offset"], sign=True),
                                      off(s, "GRU"), off(s, "GRU-drop0.2"),
                                      pt(C(pops[s]["detectors"]["LR-C-drop"]["AUROC"])), pair_r3("LR-C-drop-minus-LR-B", dom),
                                      pair_r3("LR-C-drop-minus-LR-C", dom),
                                      pt(C(pops[s]["detectors"]["GRU-drop0.2"]["AUROC"])), pair_r3("GRU-drop0.2-minus-LR-C", dom),
                                      pair_r3("GRU-drop0.2-minus-GRU", dom)]))
    ht = (r" & \multicolumn{2}{c}{Tool-block offset $w_A^{\top}c$} & \multicolumn{2}{c}{GRU tool-pathway offset} & \multicolumn{3}{c}{Tool-block dropout} & \multicolumn{3}{c}{Token dropout ($p=0.2$)} \\" "\n"
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}" "\n"
          r"Setting & LR-C & LR-C-drop & GRU & GRU-drop0.2 & \makecell{LR-C-drop\\AUROC} & \makecell{LR-C-drop\\$-$ LR-B} & \makecell{LR-C-drop\\$-$ LR-C} & \makecell{GRU-drop0.2\\AUROC} & \makecell{GRU-drop0.2\\$-$ LR-C} & \makecell{GRU-drop0.2\\$-$ GRU} \\")
    hm = ["Setting", "LR-C offset w_A^T c", "LR-C-drop offset", "GRU tool-pathway offset", "GRU-drop0.2 tool-pathway offset",
          "LR-C-drop AUROC", "LR-C-drop − LR-B", "LR-C-drop − LR-C", "GRU-drop0.2 AUROC", "GRU-drop0.2 − LR-C", "GRU-drop0.2 − GRU"]
    write("v04_oov", ht, hm, rows, "table*", CS(10, "3pt"))


# ---------------- v04_r4 (R4D M6) ----------------
def outcome_text(d):
    c = R4D["M6"]["decisions"]["plan"][d]["conditions"]
    if c["A"]:
        s = "Transfer failure under UT"
        s += ", also reasonable under IT" if c["IT_P0_reasonable"] else ", but within not reasonable under IT"
    elif c["B"] and c["C"]:
        s = "Mixed (weak representation and population-dependent, reasonable on %s only)" % ", ".join(
            x.replace("_UT", "") for x in c["C_controls_reasonable"])
    elif c["B"]:
        s = "Weak representation (no population reaches the bar)"
    elif c["C"]:
        s = "Population-dependent"
    else:
        s = "Other"
    return s


def t_r4():
    cf = R4D["M6"]["configs"]; holm = R4D["M6"]["decisions"]["holm_within_minus_transfer_cluster"]["by_config"]

    def pc(cfg, key, metric="AUROC"):
        c = cf[cfg]; p = c["point"][key][metric]; _, lo, hi = c["cluster"]["ci"][key][metric]
        return CP(p, lo, hi)

    rows = []
    for d in ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]:
        dl = cf["P0_UT"]["cluster"]["paired"]["within:%s-minus-transfer:%s" % (d, d)]["AUROC"]
        cells = [pc("P0_UT", "transfer:" + d), pc("P0_UT", "within:" + d),
                 CP(dl["mean"], dl["lo"], dl["hi"], sign=True, mark="*" if holm["P0_UT"][d]["detected_holm"] else ""),
                 pc("P0_IT", "within:" + d), pc("P0F_UT", "transfer:" + d), pc("P0_UT", "within:" + d, "APw10"),
                 R(outcome_text(d))]
        rows.append(row(LAB[d], cells))
    ht = (r" & Transfer & \multicolumn{2}{c}{Within AgentDojo, UT} & Within, IT & Transfer, & Within UT & \\" "\n"
          r"\cmidrule(lr){3-4}" "\n"
          r"Detector & AUROC & AUROC & $-$ transfer & AUROC & \makecell{unsuccessful\\as negatives} & wAUPRC 10\% & Pre-specified outcome \\")
    hm = ["Detector", "Transfer AUROC (P0)", "Within-AgentDojo AUROC (UT)", "Within (UT) − transfer, paired",
          "Within-AgentDojo AUROC (IT)", "Transfer AUROC, unsuccessful attacks also negatives (P0F)",
          "Within (UT) wAUPRC at 10%", "Pre-specified outcome (Sec. IV-F)"]
    write("v04_r4", ht, hm, rows, "table*", "@{}l*{6}{c}>{\\raggedright\\arraybackslash}p{3.3cm}@{}")


# ================================================================== SUPPLEMENT TABLES
def s_breakdown():
    M9 = PO["M9"]
    rows = [sub("R1 (pool E with 831 attacked, 600 benign and 225 hard-negative trajectories)")]

    def pm(t, pi, mix):
        x = t["precision_eq6"][pi][mix]
        if x["point"] is None:
            return R("no alert")
        if x["point"] == 1.0 and "lower_bound_if_FPR_at_CP_upper" in x:
            b = x["lower_bound_if_FPR_at_CP_upper"]
            return R(r"\makecell{$1.000$\\[-2pt]{\scriptsize$(\ge %s)$}}" % fnum(b), "1.000 (≥ %s)" % fnum(b))
        return C(x)

    def rowdet(S, d, lab):
        t = M9[S]["schemes"]["traj_strat"]["detectors"][d]["t_r80"]
        cells = [C(t["FPR_benign"]), C(t["FPR_hard_negative"]) if "FPR_hard_negative" in t else NA,
                 C(t["flag_failed_attack"]), C(t["recall_poolE"])]
        cells += [C(t[k]) if k in t else NA for k in ["recall_full", "recall_partial", "recall_delayed"]]
        cells += [pt(pm(t, "pi02", "benign_only")), pt(pm(t, "pi02", "poolE_mix")),
                  pt(pm(t, "pi02", "hard_only")) if "hard_only" in t["precision_eq6"]["pi02"] else NA]
        return row(lab, cells)

    for d in ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF", "Markov", "IForest"]:
        rows.append(rowdet("R1", d, LAB[d]))
    rows.append(sub("R3 held-out domains"))
    for d in ["LR-B", "LR-C"]:
        for dom in DOMS:
            rows.append(rowdet("R3_" + dom, d, "%s, %s" % (d, dom)))
    rows.append(sub("AgentDojo P0 (392 benign negatives, with unsuccessful attacks in the failed column)"))
    for S, sl in [("R4", "transfer"), ("WUT", "within UT")]:
        for d in ["LR-C", "GRU", "TFIDF"]:
            rows.append(rowdet(S, d, "%s, %s" % (LAB[d], sl)))
    ht = (r" & \multicolumn{2}{c}{FPR} & Failed & \multicolumn{4}{c}{Recall} & \multicolumn{3}{c}{Precision at 2\%, negative mix} \\" "\n"
          r"\cmidrule(lr){2-3}\cmidrule(lr){5-8}\cmidrule(lr){9-11}" "\n"
          r"Detector & benign & hard neg. & flagged & pool E & full & partial & delayed & benign only & pool E & hard only \\")
    hm = ["Detector", "FPR benign", "FPR hard negative", "Failed attacks flagged", "Recall pool E", "Recall full",
          "Recall partial", "Recall delayed", "Precision 2%, benign only", "Precision 2%, pool-E mix", "Precision 2%, hard only"]
    write("s_breakdown", ht, hm, rows, "table*", CS(10, "2.2pt"))


def s_splits():
    m5 = DS["m5"]; ov = DS["m5_overlap_sensitivity"]; m2 = DS["m2"]["estimates"]
    dets = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
    rows = [sub("World-disjoint control (R1-wd), pool E, five split seeds"),
            row("R1 AUROC", [C(m5["R1"][d]["AUROC"]) for d in dets])]
    for k in ["0", "1", "2", "3", "4"]:
        cells = []
        for d in dets:
            x = m5["seeds"][k]["minus_R1"][d]["AUROC"]
            cells.append(C(x, sign=True, mark="*" if x["detected"] else ""))
        rows.append(row("R1-wd s%s $-$ R1" % k, cells, lm="R1-wd s%s − R1" % k))
    acr = m5["across_seeds_AUROC"]
    rows.append(row("R1-wd AUROC, mean (range)",
                    [R(r"\makecell{$%s$\\[-2pt]{\scriptsize$(%s$ to $%s)$}}" % (fnum(acr[d]["mean"]), fnum(acr[d]["min"]), fnum(acr[d]["max"])),
                       "%s (%s to %s)" % (fnum(acr[d]["mean"]), fnum(acr[d]["min"]), fnum(acr[d]["max"]))) for d in dets]))
    cells = []
    for d in dets:
        x = ov["seeds"]["1"]["pooled"][d]["AUROC"]
        cells.append(C(x, sign=True, mark="*" if x["detected"] else ""))
    rows.append(row("R1-wd s1 $-$ R1, overlap-aware", cells, lm="R1-wd s1 − R1, overlap-aware"))
    rows.append(sub("Task-disjoint split (R2), 35 test tasks, task-cluster bootstrap"))

    def m2c(d, metric, key, sign=False):
        if d not in m2:
            return NA
        x = m2[d][metric][key] if key != "diff" else m2[d][metric]["R2_minus_R1"]["primary_R2taskcluster_minus_R1trajectory"]
        return C(x, sign=sign, mark="*" if (key == "diff" and x["detected"]) else "")
    rows.append(row("R2 AUROC", [m2c(d, "AUROC", "R2_taskcluster") for d in dets]))
    rows.append(row("R2 $-$ R1 AUROC", [m2c(d, "AUROC", "diff", True) for d in dets], lm="R2 − R1 AUROC"))
    rows.append(row("R2 wAUPRC 10\\%", [m2c(d, "AUPRC_p10", "R2_taskcluster") for d in dets], lm="R2 wAUPRC 10%"))
    rows.append(row("R2 $-$ R1 wAUPRC 10\\%", [m2c(d, "AUPRC_p10", "diff", True) for d in dets], lm="R2 − R1 wAUPRC 10%"))
    ht = "Quantity & " + " & ".join(HT[d] for d in dets) + r" \\"
    hm = ["Quantity"] + [LAB[d] for d in dets]
    write("s_splits", ht, hm, rows, "table*", "@{}l*{5}{c}@{}")


def s_shift():
    m12 = DS["m12"]["settings"]; ex = DS["m12_extra"]; ctx = DS["m12_context"]
    rows = []
    order = ["R1", "R1wd_s0", "R1wd_s1", "R1wd_s2", "R1wd_s3", "R1wd_s4", "R2"] + ["R3_" + d for d in DOMS] + ["R4"] + ["LOSO_" + s for s in SUITES]
    for s in order:
        src = m12[s] if s in m12 else ex["full"][s]
        a = src["all_rows"]; e = src["poolE_rows"]
        cx = dict(ctx.get(s, {}))
        if s.startswith("R1wd_s") and "LR-B" not in cx:
            mm = DS["m5"]["seeds"][s[-1]]["metrics"]
            cx = {"LR-B": mm["LR-B"]["AUROC"], "LR-C": mm["LR-C"]["AUROC"]}
        cells = [R("%d / %d" % (a["n_train"], a["n_test"])), P(KAPPA[s]),
                 CP(a["AUROC_oof"], a["lo"], a["hi"]), CP(e["AUROC_oof"], e["lo"], e["hi"])]
        for k in ["LR-B", "LR-C"]:
            if k in cx and cx[k].get("lo") is not None:
                cells.append(C(cx[k]))
            elif k in cx:
                cells.append(P(cx[k]["point"]))
            else:
                cells.append(NA)
        cells.append(C(cx["LR-C_R3_minus_R1dm"], sign=True) if "LR-C_R3_minus_R1dm" in cx else NA)
        lab = s.replace("_", " ").replace("R1wd", "R1-wd")
        rows.append(row(lab, cells))
    rows.append(MID)
    for s in ["WUT", "WUT_F", "WIT"]:
        w = ex["within_agentdojo_cv"][s]
        rows.append(row(WLAB[s] + " (25 CV pairs)", [R("%d / %d" % (round(w["n_train_mean"]), round(w["n_test_mean"]))), P(KAPPA[s]),
                        R(r"\makecell{$%s$\\[-2pt]{\scriptsize$(%s$ to $%s)$}}" % (fnum(w["mean"]), fnum(w["min"]), fnum(w["max"])),
                          "%s (%s to %s)" % (fnum(w["mean"]), fnum(w["min"]), fnum(w["max"]))), NA, NA, NA, NA], lm=WLAB[s] + " (25 CV pairs)"))
    ht = (r"Setting & \makecell{$n$ train\\/ test} & \makecell{Tool\\coverage $\kappa$} & \makecell{C2ST AUROC\\all rows} & \makecell{C2ST AUROC\\pool E} & \makecell{LR-B\\AUROC} & \makecell{LR-C\\AUROC} & \makecell{LR-C\\R3 $-$ R1dm} \\")
    hm = ["Setting", "n train / test", "Tool coverage κ", "C2ST AUROC, all rows", "C2ST AUROC, pool-E rows", "LR-B AUROC",
          "LR-C AUROC", "LR-C R3 − R1dm AUROC"]
    write("s_shift", ht, hm, rows, "table*", "@{}l*{7}{c}@{}")


def s_content():
    S = CD["settings"]
    rows = []

    def base(s, lab):
        det = S[s]["detectors"]
        t = det["TFIDF"]
        pr = S[s]["paired"]["TFIDF - LR-C"]["AUROC"]
        cells = [C(t["AUROC"]), C(t["AP_w02"]), C(t["recall_t_r80"]), C(t["FPR_t_r80"]),
                 C(pr, sign=True, mark="*" if pr["detected"] else ""),
                 pt(C(det["TFIDF-nodigit"]["AUROC"])) if "TFIDF-nodigit" in det else NA,
                 pt(C(det["TFIDF-toolname"]["AUROC"])) if "TFIDF-toolname" in det else NA,
                 pt(C(S[s]["per_class_t_r80"]["TFIDF"]["flag_failed_attack"]["rate"])) if "flag_failed_attack" in S[s]["per_class_t_r80"].get("TFIDF", {}) else NA]
        return row(lab, cells)

    rows.append(sub("AgentDrift ladder"))
    rows.append(base("R1", "R1"))
    wd = CD["R1wd_summary"]["TFIDF"]

    def mr(q, dd="TFIDF", dec=3):
        x = CD["R1wd_summary"][dd][q]
        a, b, c = fnum(x["mean_over_seeds"], dec=dec), fnum(x["min"], dec=dec), fnum(x["max"], dec=dec)
        return R(r"\makecell{$%s$\\[-2pt]{\scriptsize$(%s$ to $%s)$}}" % (a, b, c), "%s (%s to %s)" % (a, b, c))
    deltas = [S["R1wd_s%d" % k]["paired"]["TFIDF - LR-C"]["AUROC"]["delta"] for k in range(5)]
    rows.append(row("R1-wd (5 seeds)", [mr("AUROC", dec=4), mr("AP_w02"), mr("recall_t_r80"), mr("FPR_t_r80"),
                                                         rng(deltas, sign=True), mr("AUROC", "TFIDF-nodigit", dec=4), NA, NA]))
    rows.append(base("R2", "R2 (task clusters)"))
    for dom in DOMS:
        rows.append(base("R3_" + dom, "R3 " + dom))
    rows.append(sub("AgentDojo"))
    for s, lab in [("R4", "R4 transfer"), ("WUT", WLAB["WUT"]), ("WUT_F", WLAB["WUT_F"]), ("WIT", WLAB["WIT"])]:
        r_ = base(s, lab)
        rows.append(("row", lab, lab, r_[3]))
    for su in SUITES:
        rows.append(base("LOSO_" + su, "LOSO " + su))
    rows.append(sub("R1 masking variants (fifth column gives the paired difference to TF-IDF)"))
    for v, lab in [("TFIDF-worldmask", "World mask (plan)"), ("TFIDF-anon", "README anonymization"),
                   ("TFIDF-anon-nodigit", "Anonymization and no digits"), ("TFIDF-nodigit", "No digits")]:
        t = S["R1"]["detectors"][v]; pr = S["R1"]["paired"]["%s - TFIDF" % v]["AUROC"]
        rows.append(row(lab, [C(t["AUROC"], dec=4), C(t["AP_w02"]), C(t["recall_t_r80"]), C(t["FPR_t_r80"]),
                              C(pr, sign=True, dec=4, mark="*" if pr["detected"] else ""), NA, NA, NA]))
    ht = (r" & \multicolumn{4}{c}{TF-IDF (content)} & TF-IDF & \makecell{TF-IDF\\no digits} & \makecell{TF-IDF\\tool names} & Failed \\" "\n"
          r"\cmidrule(lr){2-5}" "\n"
          r"Setting & AUROC & wAUPRC 2\% & Recall & FPR & $-$ LR-C & AUROC & AUROC & flagged \\")
    hm = ["Setting", "TF-IDF AUROC", "TF-IDF wAUPRC 2%", "TF-IDF recall at θr80", "TF-IDF FPR at θr80", "TF-IDF − LR-C AUROC",
          "TF-IDF no digits AUROC", "TF-IDF tool names AUROC", "Unsuccessful attacks flagged (TF-IDF)"]
    write("s_content", ht, hm, rows, "table*", CS(8))


def s_obslen():
    M52 = FO["M5_2_without_observation_length"]; pad = FO["M5_3_padding"]
    rows = []
    for s, lab in [("R1", "R1")] + [("R3_" + d, "R3 " + d) for d in DOMS] + [("R4", "R4 transfer"), ("WUT", WLAB["WUT"])]:
        m = M52[s]["primary"]["metrics"]
        rows.append(row(lab, [C(m[d]["AUROC"]) for d in ["LR-B", "LR-B-noobs", "LR-C", "LR-C-noobs", "LR-obsonly"]]))
    rows.append(hdr(["Padding (recall at %s, AUROC)" % TH, "$+0$", "$+100$", "$+300$", "$+600$", "$+1{,}200$"],
                    ["Padding (recall at θr80, AUROC)", "+0", "+100", "+300", "+600", "+1,200"]))
    L = pad["R1_linear"]["eval"]["metrics"]; G = pad["R1_GRU"]["eval"]["metrics"]; W = pad["WUT"]["eval"]["primary"]["metrics"]
    for src, d, lab in [(L, "LR-B", "R1 LR-B"), (L, "LR-C", "R1 LR-C"), (G, "GRU", "R1 GRU"), (W, "LR-B", "Within UT LR-B"), (W, "LR-C", "Within UT LR-C")]:
        for mk, ml in [("recall@t_r80", "recall"), ("AUROC", "AUROC")]:
            rows.append(row("%s, %s" % (lab, ml), [C(src["%s|pad%d" % (d, n)][mk]) for n in [0, 100, 300, 600, 1200]]))
    ht = r"AUROC, pool E & LR-B & LR-B-noobs & LR-C & LR-C-noobs & LR-obsonly \\"
    hm = ["AUROC, pool E", "LR-B", "LR-B-noobs", "LR-C", "LR-C-noobs", "LR-obsonly"]
    write("s_obslen", ht, hm, rows, "table*", "@{}l*{5}{c}@{}")


def s_keyword():
    K = FO["m8_keyword_flag"]; A = FO["m8_LRB_annotation_features"]
    rows = []
    for k, lab in [("keyword_vs_annotation_agentdrift", "AgentDrift (train and test)"),
                   ("keyword_vs_annotation_agentdojo", "AgentDojo (P0 and unsuccessful)"),
                   ("keyword_vs_annotation_agentdojo_P0", "AgentDojo P0")]:
        x = K[k]; tl = x["tool_level"]
        rows.append(row(lab, [frac(tl["tp"], tl["tp"] + tl["fp"]), frac(tl["tp"], tl["tp"] + tl["fn"]),
                              C(x["occurrence_precision_ci"]), C(x["occurrence_recall_ci"])]))
    rows.append(hdr(["LR-B AUROC", "LR-B", "LR-B-annot", "LR-B-minus", r"LR-B-annot $-$ LR-B"],
                    ["LR-B AUROC", "LR-B", "LR-B-annot", "LR-B-minus", "LR-B-annot − LR-B"]))
    holm = A["R3_holm"]["LR-B-annot minus LR-B"]["AUROC"]
    for s, lab in [("R1", "R1")] + [("R3_" + d, "R3 " + d) for d in DOMS] + [("R4", "R4 transfer")]:
        p = A[s]["primary"]; m = p["metrics"]; ct = p["contrasts"]["LR-B-annot minus LR-B"]["AUROC"]
        mk = "*" if (s.startswith("R3_") and holm[s[3:]]["detected_holm"]) else ""
        rows.append(row(lab, [C(m["LR-B"]["AUROC"]), C(m["LR-B-annot"]["AUROC"]), C(m["LR-B-minus"]["AUROC"]),
                              C(ct, sign=True, mark=mk)]))
    ht = r"Keyword flag vs annotation & \makecell{Tool-level\\precision} & \makecell{Tool-level\\recall} & \makecell{Call-level\\precision} & \makecell{Call-level\\recall} \\"
    hm = ["Keyword flag vs annotation", "Tool-level precision", "Tool-level recall", "Call-level precision", "Call-level recall"]
    write("s_keyword", ht, hm, rows, "table*", "@{}l*{4}{c}@{}")


def s_r4_controls():
    cf = R4D["M6"]["configs"]; dec = R4D["M6"]["decisions"]["plan"]
    dets = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
    rows = []

    def m6row(cfg, lab):
        c = cf[cfg]; cells = []
        for d in dets:
            for side in ["within", "transfer"]:
                k = "%s:%s" % (side, d)
                p = c["point"][k]["AUROC"]; _, lo, hi = c["cluster"]["ci"][k]["AUROC"]
                mk = "§" if (side == "within" and dec[d]["per_config"][cfg]["within_reasonable"]) else ""
                cells.append(CP(p, lo, hi, mark=mk) if side == "within" else P(p))
        return row(lab, cells)

    for cfg, lab in [("P0_UT", "P0"), ("C1_UT", "C1"), ("MB_UT", "Model-balanced"),
                     ("WUTF_P0", "UT-F"), ("P0F_UT", "P0F"), ("WUTF_P0F", "UT-F on P0F")]:
        r_ = m6row(cfg, lab); rows.append(("row", lab, lab.replace("\\_", "_"), r_[3]))
    wd = R4D["m7_Q4"]["with_without_dos"]["P0noDoS_UT"]
    cells = []
    for d in dets:
        for side in ["within", "transfer"]:
            k = "%s:%s" % (side, d); p = wd["point"][k]["AUROC"]; _, lo, hi = wd["cluster_ci"][k]["AUROC"]
            cells.append(CP(p, lo, hi) if side == "within" else P(p))
    rows.append(row("P0 without DoS", cells))
    ii = R4D["Q2"]["II_only"]["pooled"]; cells = []
    for d in dets:
        for side, k in [("within", "WUT:" + d), ("transfer", "transfer:" + d)]:
            p = ii["point"][k]["AUROC"]; _, lo, hi = ii["strat_ci"][k]["AUROC"]
            cells.append(CP(p, lo, hi) if side == "within" else P(p))
    rows.append(row(r"\makecell[l]{important\_instructions\\only (traj.)}", cells, lm="important_instructions only (trajectory bootstrap)"))
    rngT = R4D["Q2"]["per_template_range_transfer"]; cells = []
    for d in dets:
        cells += [NA, R(r"\makecell{$%s$\\to $%s$}" % (fnum(rngT[d][0]), fnum(rngT[d][1])), "%s to %s" % (fnum(rngT[d][0]), fnum(rngT[d][1])))]
    rows.append(row(r"\makecell[l]{Per template,\\gpt-4o (range)}", cells, lm="Per template, gpt-4o (range over 15 templates)"))
    for cfg, lt, lm in [("ATTEMPT_UT", r"\makecell[l]{Attempt target,\\UT (exploratory)}", "Attempt target, UT (exploratory)"),
                        ("ATTEMPT_IT", r"\makecell[l]{Attempt target,\\IT (exploratory)}", "Attempt target, IT (exploratory)")]:
        rows.append(("row", lt, lm, m6row(cfg, lm)[3]))
    rows.append(sub("Per model (UT, cluster intervals)"))
    pm = R4D["M6"]["per_model"]
    for mdl in MODELS:
        x = pm[mdl]; cells = []
        for d in dets:
            for side, k in [("within", "within_UT:" + d), ("transfer", "transfer:" + d)]:
                p = x["point"][k]["AUROC"]; _, lo, hi = x["cluster_UT"]["ci"][k]["AUROC"]
                mk = "§" if (side == "within" and x["within_UT_reasonable_cluster"][d]) else ""
                cells.append(CP(p, lo, hi, mark=mk) if side == "within" else P(p))
        short = mdl.replace("-2024-04-09", "").replace("-2024-05-13", "").replace("-002", "").replace("-001", "")
        rows.append(row(r"\makecell[l]{\texttt{%s}\\(%d/%d)}" % (short, x["n_neg"], x["n_pos"]), cells, lm="`%s` (%d/%d)" % (mdl, x["n_neg"], x["n_pos"])))
    ht = (" & " + " & ".join(r"\multicolumn{2}{c}{%s}" % (LAB[d] if d not in ("LR-A", "TFIDF") else {"LR-A": "LR-A (tool identity)", "TFIDF": "TF-IDF (content)"}[d]) for d in dets) + r" \\" "\n"
          + "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(5)) + "\n"
          + "Population & " + " & ".join(["Within & Transfer"] * 5) + r" \\")
    hm = ["Population"] + ["%s %s" % (LAB[d], s) for d in dets for s in ["within", "transfer"]]
    write("s_r4_controls", ht, hm, rows, "table*", CS(10))


def s_intervals():
    m2 = R4D["m2"]; pt = m2["point"]
    dets = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF", "Markov", "IForest"]
    hU = m2["cluster_UT"]["holm_WUT_minus_transfer"]; hI = m2["cluster_IT"]["holm_WIT_minus_transfer"]
    rows = []
    for d in dets:
        def ci(scheme, key):
            _, lo, hi = m2[scheme]["ci"][key]["AUROC"]
            return CP(pt[key]["AUROC"], lo, hi)
        dU = m2["cluster_UT"]["paired"]["WUT:%s-minus-transfer:%s" % (d, d)]["AUROC"]
        dI = m2["cluster_IT"]["paired"]["WIT:%s-minus-transfer:%s" % (d, d)]["AUROC"]
        cells = [ci("strat", "transfer:" + d), ci("cluster_UT", "transfer:" + d), ci("cluster_IT", "transfer:" + d),
                 ci("strat", "WUT:" + d), ci("cluster_UT", "WUT:" + d), ci("strat", "WIT:" + d), ci("cluster_IT", "WIT:" + d),
                 CP(dU["mean"], dU["lo"], dU["hi"], sign=True, mark="*" if hU[d] < 0.05 else ""),
                 CP(dI["mean"], dI["lo"], dI["hi"], sign=True, mark="*" if hI[d] < 0.05 else "")]
        rows.append(row(LAB[d], cells))
    # prevalence-weighted AUPRC at 10% and 5% (review m17.3), with the v0.3 subsample estimator at 5% as secondary
    m17 = R4D["m17_3"]
    rows.append(hdr(["wAUPRC", r"Transfer 10\%", r"Transfer 5\%", r"UT 10\%", r"UT 5\%", r"IT 10\%", r"IT 5\%",
                     r"\makecell{Subsample 5\%\\transfer}", r"\makecell{Subsample 5\%\\UT}", r"\makecell{Subsample 5\%\\IT}"],
                    ["wAUPRC", "Transfer 10%", "Transfer 5%", "Within UT 10%", "Within UT 5%", "Within IT 10%", "Within IT 5%",
                     "v0.3 subsample 5%, transfer", "v0.3 subsample 5%, within UT", "v0.3 subsample 5%, within IT"]))
    for d in dets:
        cells = []
        for key, cik in [("transfer:" + d, "cluster_UT_ci"), ("WUT:" + d, "cluster_UT_ci"), ("WIT:" + d, "cluster_IT_ci")]:
            for q in ["APw10", "APw05"]:
                lo, hi = m17[cik][key][q]
                cells.append(CP(m17["point"][key][q], lo, hi))
        for key in ["transfer:" + d, "WUT:" + d, "WIT:" + d]:
            cells.append(C(m17["subsample_v03"]["p05"]["ci"][key]))
        rows.append(row(LAB[d], cells))
    ht = (r" & \multicolumn{3}{c}{Transfer AUROC} & \multicolumn{2}{c}{Within UT AUROC} & \multicolumn{2}{c}{Within IT AUROC} & \makecell{UT $-$\\transfer} & \makecell{IT $-$\\transfer} \\" "\n"
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
          r"Detector & traj. & UT clust. & IT clust. & traj. & UT clust. & traj. & IT clust. & UT clust. & IT clust. \\")
    hm = ["Detector", "Transfer, trajectory", "Transfer, UT clusters", "Transfer, IT clusters", "Within UT, trajectory", "Within UT, UT clusters",
          "Within IT, trajectory", "Within IT, IT clusters", "Within UT − transfer, UT clusters", "Within IT − transfer, IT clusters"]
    write("s_intervals", ht, hm, rows, "table*", CS(9))


def s_oppoints():
    M8 = PO["M8_R1"]["detectors"]; B = PO["R1_operating_points"]["m10_fpr_budgets"]["detectors"]; T5 = PO["m4_theta05"]["table"]
    rows = []
    for d in ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF", "Markov", "IForest"]:
        b = B[d]["benign_hard"]
        cells = [C(M8[d]["FPR_at_test_TPR80"]["FPR_at_TPR80"]),
                 C(b["fpr01"]["test_TPR"]), C(b["fpr01"]["test_FPR"]), C(b["fpr05"]["test_TPR"]), C(b["fpr05"]["test_FPR"])]
        if d in T5["R1"]:
            cells += [C(T5["R1"][d]["recall_poolE"]), C(T5["R1"][d]["FPR_poolE"]),
                      rng([T5["R3_" + x][d]["recall_poolE"]["point"] for x in DOMS]),
                      rng([T5["R3_" + x][d]["FPR_poolE"]["point"] for x in DOMS]),
                      pt(C(T5["R4"][d]["recall_poolE"])), pt(C(T5["R4"][d]["FPR_poolE"]))]
        else:
            cells += [NA] * 6
        rows.append(row(LAB[d], cells))
    g = PO["m4_gru_t_val80"]["table"]["R1"]["GRU"]["t_val80"]
    rows.append(MID)
    rows.append(row(r"GRU at $\theta_{val80}$ (R1)", [NA, NA, NA, NA, NA, C(g["recall_poolE"] if "recall_poolE" in g else g["recall"]),
                                                   C(g["FPR_poolE"] if "FPR_poolE" in g else g["FPR"]), NA, NA, NA, NA], lm="GRU at θval80 (R1)"))
    ht = (r" & \makecell{FPR at test\\TPR 0.8} & \multicolumn{2}{c}{1\% budget} & \multicolumn{2}{c}{5\% budget} & \multicolumn{2}{c}{$\theta=0.5$, R1} & \multicolumn{2}{c}{$\theta=0.5$, R3 (range)} & \multicolumn{2}{c}{$\theta=0.5$, R4} \\" "\n"
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}" "\n"
          r"Detector & (descriptive) & TPR & FPR & TPR & FPR & Recall & FPR & Recall & FPR & Recall & FPR \\")
    hm = ["Detector", "FPR at test TPR 0.8 (descriptive)", "1% budget TPR", "1% budget FPR", "5% budget TPR", "5% budget FPR",
          "θ=0.5 R1 recall", "θ=0.5 R1 FPR", "θ=0.5 R3 recall (range)", "θ=0.5 R3 FPR (range)", "θ=0.5 R4 recall", "θ=0.5 R4 FPR"]
    write("s_oppoints", ht, hm, rows, "table*", CS(11, "2.2pt"))


def s_labelfree():
    m = PO["m11_label_free"]["folds"]
    rows = []

    def sm(x):
        return CP(x["mean"], x["min"], x["max"])
    for dom in DOMS:
        for d in ["LR-C", "LR-B", "GRU"]:
            x = m[dom]["detectors"][d]; a = x["primary_poolE"]["summary_over_draws"]; b = x["deployment_prev02"]["summary_over_draws"]
            t = x["primary_poolE"]["theta_r80_same_fold_full_poolE"]
            rows.append(row("%s, %s" % (dom, LAB[d]), [sm(a["FPR"]), sm(a["recall"]), sm(b["FPR"]), sm(b["recall"]), sm(b["precision_pi02"]),
                                                       P(t["FPR"]), P(t["recall"])]))
        if dom != "coding":
            rows.append(MID)
    ht = (r" & \multicolumn{2}{c}{(a) native sample, pre-specified} & \multicolumn{3}{c}{(b) 2\%% traffic, post hoc} & \multicolumn{2}{c}{%s, same fold} \\" % TH + "\n"
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}" "\n"
          r"Fold and detector & FPR & Recall & FPR & Recall & Precision 2\% & FPR & Recall \\")
    hm = ["Fold and detector", "(a) FPR", "(a) recall", "(b) FPR", "(b) recall", "(b) precision at 2%", "θr80 FPR", "θr80 recall"]
    write("s_labelfree", ht, hm, rows, "table*", "@{}l*{7}{c}@{}")


def s_probes():
    m = DG["m15_probes"]; r = R4D["m15"]
    rows = [sub("AgentDrift domain and world probes (accuracy)")]
    for k, lab in [("domain", "Domain, all pooled"), ("domain_benign_only", "Domain, benign only"), ("world", "World, 54 worlds of 30 or more"),
                   ("world_nonemail_supplementary", "World, non-email (supplementary)")]:
        u = m[k]["units"]
        cells = [C(u[x]["accuracy"]) if x in u else NA for x in ["A", "B", "B-minus", "C", "majority", "domain_only"]]
        rows.append(row("%s ($n=%s$)" % (lab, format(m[k]["n"], ",")), cells, lm="%s (n=%s)" % (lab, format(m[k]["n"], ","))))
    rows.append(sub("AgentDojo model-identity probe on benign P0 (AUROC, with the suite floor in the last column)"))
    a = r["AUROC"]
    rows.append(row("gpt-4o vs other models (92 vs 300)", [NA, C(a["B"]), NA, C(a["C"]), C(a["majority"]), C(a["suite"])]))
    ht = r"Probe & A (tools) & B (behavioral) & B-minus & C (combined) & Majority & \makecell{Domain only\\/ suite} \\"
    hm = ["Probe", "A (tools)", "B (behavioral)", "B-minus", "C (combined)", "Majority floor", "Domain-only floor / suite floor"]
    write("s_probes", ht, hm, rows, "table*", "@{}l*{6}{c}@{}")


def s_coverage():
    L = OOV["s2_graded"]["levels"]
    rows = []
    dets = ["LR-C", "LR-C-drop", "LR-B", "GRU", "GRU-drop0.2"]
    contr = ["LR-C-minus-LR-B", "LR-C-drop-minus-LR-B", "GRU-minus-LR-B", "GRU-drop0.2-minus-LR-B", "GRU-minus-LR-C"]
    levels = ["1.0", "0.75", "0.5", "0.25", "0.0"]
    lab = lambda lv: "%s (%s)" % (lv, fnum(L[lv]["coverage_mean"]))
    for lv in levels:
        rows.append(row(lab(lv), [C(L[lv]["detectors"][d]["AUROC"]) for d in dets]))
    rows.append(hdr(["FPR at %s" % TH] + dets, ["FPR at θr80"] + dets))
    for lv in levels:
        rows.append(row(lab(lv), [C(L[lv]["detectors"][d]["FPR@t_r80"]) for d in dets]))
    rows.append(hdr(["Paired AUROC"] + [c.replace("-minus-", " $-$ ") for c in contr], ["Paired AUROC"] + [c.replace("-minus-", " − ") for c in contr]))
    for lv in levels:
        rows.append(row(lab(lv), [C(L[lv]["contrasts"][c]["AUROC"], sign=True, mark="*" if L[lv]["contrasts"][c]["AUROC"]["detected"] else "") for c in contr]))
    ht = r"\makecell{AUROC by target coverage\\(achieved coverage)} & LR-C & LR-C-drop & LR-B & GRU & GRU-drop0.2 \\"
    hm = ["AUROC by target coverage (achieved coverage)"] + dets
    write("s_coverage", ht, hm, rows, "table*", "@{}l*{5}{c}@{}")


def s_loso():
    S5 = OOV["s5_loso"]; S1 = OOV["s1_lrc_offset"]["settings"]
    rows = []
    for su in SUITES:
        x = S5["suites"][su]; lc = x["LOSO_cluster"]; ct = x["contrasts"]["LR-C-minus-LR-B"]
        cells = [R("%d / %d (%d)" % (x["n_pos"], x["n_neg"], x["n_clusters"] if isinstance(x["n_clusters"], int) else x["n_clusters"].get("joint", 0))),
                 P(x["coverage"]["tool_occurrence_cov"]),
                 C(lc["LR-B"]["AUROC"]), C(lc["LR-C"]["AUROC"]), C(lc["LR-C-drop"]["AUROC"]),
                 C(ct["LOSO_cluster"], sign=True, mark="*" if ct["LOSO_cluster"]["detected_holm"] else ""),
                 C(ct["LOSO_stratified"], sign=True, mark="*" if ct["LOSO_stratified"]["detected_holm"] else ""),
                 C(ct["WUT_suite_matched_cluster"], sign=True, mark="*" if ct["WUT_suite_matched_cluster"]["detected_holm"] else ""),
                 P(S1["LOSO_" + su]["LR-C"]["offset"], sign=True)]
        rows.append(row(su, cells))
    w = S5["WUT_pooled"]; wc = w["contrasts"]["LR-C-minus-LR-B"]
    rows.append(MID)
    rows.append(row("Within UT, pooled", [R("%d / %d" % (w["n_pos"], w["n_neg"])), NA, NA, NA, NA,
                                                     C(wc["cluster"] if "cluster" in wc else wc, sign=True, mark="*" if (wc.get("cluster", wc).get("detected")) else ""),
                                                     C(wc["stratified"], sign=True, mark="*" if wc["stratified"]["detected"] else "") if "stratified" in wc else NA, NA, NA]))
    ht = (r" & & & \multicolumn{3}{c}{Held-out suite AUROC (cluster)} & \multicolumn{3}{c}{LR-C $-$ LR-B} & \\" "\n"
          r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}" "\n"
          r"Held-out suite & \makecell{pos / neg\\(clusters)} & Coverage & LR-B & LR-C & LR-C-drop & \makecell{LOSO\\cluster} & \makecell{LOSO\\traj.} & \makecell{Within UT,\\suite-matched} & \makecell{LR-C\\offset} \\")
    hm = ["Held-out suite", "pos / neg (clusters)", "Tool-occurrence coverage", "LR-B AUROC", "LR-C AUROC", "LR-C-drop AUROC",
          "LR-C − LR-B, LOSO cluster", "LR-C − LR-B, LOSO trajectory", "LR-C − LR-B, within UT suite-matched", "LR-C tool-block offset"]
    write("s_loso", ht, hm, rows, "table*", CS(9))


def s_gru():
    F = OOV["s7_q1"]["folds"]; SS = R4D["m16"]["seed_spread"]
    rows = []
    for s, lab in [("R1", "R1")] + [("R3_" + d, "R3 " + d) for d in DOMS]:
        g = F[s]["GRU"]; ps = SS[s]["GRU"]["per_seed"]
        cells = [stack(g["AUROC_per_seed"]), stack([x["point"] for x in g["minus_LR-C_per_seed"]], sign=True),
                 C(g["minus_LR-C"]["two_level"], sign=True), C(g["minus_LR-C"]["mixture"], sign=True),
                 C(g["minus_LR-C"]["seed_averaged"], sign=True),
                 stack([x["FPR@t_r80"] for x in ps]), stack([x["recall@t_r80"] for x in ps])]
        if s.startswith("R3_"):
            m = GDIAG[s[3:]]["by_prev"]["native"]["metrics"]
            cells += [P(m["GRU-B_beh"]["AUROC"][0]), P(m["GRU-A_tool"]["AUROC"][0])]
        else:
            cells += [NA, NA]
        rows.append(row(lab, cells))
    ht = (r" & \multicolumn{2}{c}{Per seed (0 / 1 / 2)} & \multicolumn{3}{c}{GRU $-$ LR-C AUROC, estimand} & \multicolumn{2}{c}{Per seed at %s} & \multicolumn{2}{c}{v0.3 post hoc diagnostic} \\" % TH + "\n"
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}" "\n"
          r"Setting & AUROC & $-$ LR-C & two-level & mixture & seed-avg. & FPR & Recall & \makecell{LR head on B,\\GRU training} & \makecell{GRU, tool\\sequence only} \\")
    hm = ["Setting", "GRU AUROC per seed", "GRU − LR-C per seed", "GRU − LR-C, two-level", "GRU − LR-C, mixture (pre-specified)",
          "GRU − LR-C, seed-averaged", "FPR at θr80 per seed", "Recall at θr80 per seed",
          "v0.3 post hoc LR head on B, GRU training (AUROC)", "v0.3 post hoc GRU on tool sequence only (AUROC)"]
    write("s_gru", ht, hm, rows, "table*", "@{}l*{9}{c}@{}")


def s_m7():
    T = R4D["M7"]["table"]; dets = ["LR-A", "LR-B", "LR-C", "GRU"]
    rows = []

    def cell(v03, cor, sign=False):
        p, lo, hi = cor
        if v03 is None:
            return R(r"\cit{\text{not run}\rightarrow %s}{%s}{%s}" % (fnum(p, sign), fnum(lo, sign), fnum(hi, sign)),
                     "not run → %s [%s, %s]" % (num_md(p, sign), num_md(lo, sign), num_md(hi, sign)))
        return R(r"\cit{%s\rightarrow %s}{%s}{%s}" % (fnum(v03[0], sign), fnum(p, sign), fnum(lo, sign), fnum(hi, sign)),
                 "%s → %s [%s, %s]" % (num_md(v03[0], sign), num_md(p, sign), num_md(lo, sign), num_md(hi, sign)))
    for q, lab, sg in [("transfer_AUROC", "Transfer AUROC", False), ("WUT_AUROC", "Within UT AUROC", False), ("WIT_AUROC", "Within IT AUROC", False),
                       ("WUT_minus_transfer", "Within UT $-$ transfer", True), ("WIT_minus_transfer", "Within IT $-$ transfer", True),
                       ("transfer_FPR@t_r80", "Transfer FPR at %s" % TH, False)]:
        rows.append(row(lab, [cell(T[d]["v03"][q], T[d]["corrected"][q], sg) for d in dets],
                        lm=lab.replace("$-$", "−").replace(TH, "θr80")))
    for mdl in MODELS:
        rows.append(row(r"Transfer, \texttt{%s}" % mdl, [cell(T[d]["v03"]["per_model_transfer"][mdl], T[d]["corrected"]["per_model_transfer"][mdl]) for d in dets],
                        lm="Transfer, `%s`" % mdl))
    rows.append(hdr(["Short-run filter, excluded / all", "Benign", "Compromised", "Unsuccessful", "Kept (b / c / u)"],
                    ["Short-run filter, excluded / all", "Benign", "Compromised", "Unsuccessful", "Kept (benign / compromised / unsuccessful)"]))
    bm = MANI["by_model"]
    for mdl in ["gpt-4o-2024-05-13", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-002", "gpt-4-turbo-2024-04-09", "gemini-2.0-flash-001"]:
        x = bm[mdl]
        rows.append(row(r"\texttt{%s}" % mdl, [frac(x["benign_excluded_short"], x["benign_all"]),
                                              frac(x.get("attacked_excluded_short", 0), x["attacked_all"]),
                                              frac(x.get("attack_failed_excluded_short", 0), x["attack_failed_all"]),
                                              R("%d / %d / %d" % (x["benign"], x["attacked"], x["attack_failed"]))], lm="`%s`" % mdl))
    ex = MANI["excluded_fewer_than_two_calls"]; kp = MANI["counts_kept"]
    rows.append(row("All models", [frac(ex["benign_excluded_short"], ex["benign_excluded_short"] + kp["benign"]),
                                   frac(ex["attacked_excluded_short"], ex["attacked_excluded_short"] + kp["attacked"]),
                                   frac(ex["attack_failed_excluded_short"], ex["attack_failed_excluded_short"] + kp["attack_failed"]),
                                   R("%d / %d / %d" % (kp["benign"], kp["attacked"], kp["attack_failed"]))]))
    ht = r"v0.3 $\rightarrow$ corrected & LR-A (tool identity) & LR-B (behavioral) & LR-C (combined) & GRU \\"
    hm = ["v0.3 → corrected", "LR-A (tool identity)", "LR-B (behavioral)", "LR-C (combined)", "GRU"]
    write("s_m7", ht, hm, rows, "table*", "@{}l*{4}{c}@{}")


def s_thresholds_all():
    M4 = DS["M4"]; der = DS["derived"]["M4"]
    arrow = lambda s: "holds" if s == "holds" else " ".join(t.replace("FPR+", "FPR↑").replace("FPR-", "FPR↓").replace("rec+", "rec↑").replace("rec-", "rec↓") for t in s.split(","))
    arrow_tex = lambda s: "holds" if s == "holds" else " ".join(
        t.replace("FPR+", r"FPR$\uparrow$").replace("FPR-", r"FPR$\downarrow$").replace("rec+", r"rec$\uparrow$").replace("rec-", r"rec$\downarrow$") for t in s.split(","))
    keys = ["LR-A@t_r80", "LR-B@t_r80", "LR-B-minus@t_r80", "LR-B-noobs@t_r80", "LR-C@t_r80", "LR-C-drop@t_r80", "LR-C-noobs@t_r80",
            "GRU@t_r80", "GRU#seed0@t_r80", "GRU#seed1@t_r80", "GRU#seed2@t_r80", "GRU@t_val80", "GRU-drop0.2@t_r80", "GRU-drop0.1@t_r80",
            "TFIDF@t_r80", "TFIDF-nodigit@t_r80", "Markov@t_r80", "IForest@t_r80"]
    rows = [sub("Status relative to domain-matched R1 after Holm over five folds")]
    for k in keys:
        st = M4["status"][k]["vs_domain_matched_R1"]["holm"]
        cells = [R(arrow_tex(st[d]), arrow(st[d])) for d in DOMS]
        cells.append(R("%d / %d" % (der["n_folds_failing_vs_dm_holm"][k], der["n_folds_failing_vs_dm_unadjusted"][k])))
        base = k.split("@")[0].split("#")[0]
        nice = LAB.get(base, base.replace("TFIDF-nodigit", "TF-IDF no digits"))
        rest = k[len(base):].replace("@t_r80", "").replace("#seed", ", seed ")
        lt = nice + rest.replace("@t_val80", r" at $\theta_{val80}$")
        lm = nice + rest.replace("@t_val80", " at θval80")
        rows.append(row(lt, cells, lm=lm))
    rows.append(sub("R3 minus domain-matched R1 at %s (paired rates, independent bootstraps)" % TH, "R3 minus domain-matched R1 at θr80"))
    for d in ["LR-B", "LR-C", "GRU", "TFIDF"]:
        for dk, ml in [("recall", "recall"), ("FPR", "FPR")]:
            dd = M4["deltas"][d + "@t_r80"]["vs_domain_matched_R1"][dk]
            cells = [C(dd[x], sign=True, mark="*" if dd[x]["detected_holm"] else "") for x in DOMS] + [R("")]
            rows.append(row("%s, $\\Delta$ %s" % (LAB[d], ml), cells, lm="%s, Δ %s" % (LAB[d], ml)))
    ht = r"Detector & medical & banking & web & email & coding & \makecell{Folds failing\\(Holm / unadj.)} \\"
    hm = ["Detector", "medical", "banking", "web", "email", "coding", "Folds failing (Holm / unadjusted)"]
    write("s_thresholds_all", ht, hm, rows, "table*", "@{}l*{6}{c}@{}")


if __name__ == "__main__":
    for f in [t_prevalence, t_r3, t_thresholds, t_oov, t_r4, s_breakdown, s_splits, s_shift, s_content, s_obslen, s_keyword,
              s_r4_controls, s_intervals, s_oppoints, s_labelfree, s_probes, s_coverage, s_loso, s_gru, s_m7, s_thresholds_all]:
        f()
    json.dump(COLSPECS, open(os.path.join(OUT, "v04_colspecs.json"), "w", encoding="utf-8"), indent=1)
    print("wrote", len(COLSPECS), "tables:", ", ".join(COLSPECS))
