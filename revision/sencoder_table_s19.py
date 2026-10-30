"""Table S19 (proposed) from revision/sencoder_results.json and revision/sencoder_table6_row.json.

Formatting only, no computation beyond averaging the five mask draws of the embedding-drift check.
Writes revision/sencoder_s19/s_sencoder.md and s_sencoder.tex in the format of paper/gen_tables_v04.py
(\\cit{point}{lo}{hi} cells in LaTeX, "point [lo, hi]" with a true minus sign in Markdown).
Nothing in paper/, MANUSCRIPT_SOURCE.md or SUPPLEMENT_SOURCE.md is read for numbers or written.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sencoder_s19")
DOM = ["medical", "banking", "web", "email", "coding"]
MINUS = "\u2212"
NCOL = 8   # label + 7 value columns


def fnum(x, sign=False, dec=3):
    s = f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"
    return "0.000" if s in ("-0.000", "+0.000") and not sign else s


class C:
    """One cell. v, lo, hi numbers (lo/hi optional), or raw=(tex, md)."""
    def __init__(self, v=None, lo=None, hi=None, sign=False, dec=3, mark="", raw=None):
        self.v, self.lo, self.hi, self.sign, self.dec, self.mark, self.raw = v, lo, hi, sign, dec, mark, raw

    def tex(self):
        if self.raw is not None:
            return self.raw[0]
        f = lambda x: fnum(x, self.sign, self.dec)
        m = "^{*}" if self.mark == "*" else ""
        if self.lo is None:
            return "$%s%s$" % (f(self.v), m)
        return r"\cit{%s%s}{%s}{%s}" % (f(self.v), m, f(self.lo), f(self.hi))

    def md(self):
        if self.raw is not None:
            return self.raw[1]
        f = lambda x: fnum(x, self.sign, self.dec).replace("-", MINUS)
        if self.lo is None:
            return f(self.v) + self.mark
        return f"{f(self.v)}{self.mark} [{f(self.lo)}, {f(self.hi)}]"


def raw(t, m=None):
    return C(raw=(t, t if m is None else m))


BLANK = raw("", "")
NA = raw("n/a")


def ci(x, sign=False, mark=""):
    """x is [p, lo, hi] or {point|delta, lo, hi}."""
    if isinstance(x, dict):
        x = [x.get("point", x.get("delta")), x["lo"], x["hi"]]
    return C(x[0], x[1], x[2], sign=sign, mark=mark)


def rng(a, b, sign=True):
    t = r"\makecell{$%s$\\to $%s$}" % (fnum(a, sign), fnum(b, sign))
    m = "%s to %s" % (fnum(a, sign).replace("-", MINUS), fnum(b, sign).replace("-", MINUS))
    return raw(t, m)


def build(R, T6):
    rows = []
    L = R["ladder"]; dr = R["embedding_drift"]["rungs"]
    # ---------------- panel a: ladder
    head_a = ["(a) Rung", "κ", "AUROC", "SE − LR-B, paired", "AUPRC at 10%", "recall at θr80", "FPR at θr80", "Drift"]
    head_a_tex = ["(a) Rung", r"$\kappa$", "AUROC", r"SE $-$ LR-B, paired", r"AUPRC at 10\%", r"recall at $\theta_{r80}$",
                  r"FPR at $\theta_{r80}$", "Drift"]
    names = [("R1", "R1")] + [(f"R1wd_s{k}", f"R1-wd split {k}") for k in range(5)] + [("R2", "R2")] + \
            [(f"R3_{d}", f"R3 {d}") for d in DOM]
    for s, lab in names:
        bp = L[s]["by_prev"]; m = bp["native"]["metrics"]["SE"]; p = bp["native"]["paired"]["SE-minus-LR-B"]["AUROC"]
        mark = "*" if (p["lo"] > 0 or p["hi"] < 0) else ""
        rows.append(("row", lab, lab, [C(L[s]["kappa"]), ci(m["AUROC"]), ci(p, sign=True, mark=mark),
                                      ci(bp["p10"]["metrics"]["SE"]["AUPRC"]), ci(m["recall@t_r80"]), ci(m["FPR@t_r80"]),
                                      C(dr[s]["pool_E"]["mean"])]))
    # ---------------- panel b: reversal rule and offsets
    rows.append(("hdr", ["(b) Held-out domain", r"R1dm SE $-$ LR-B", r"R3 SE $-$ LR-B", r"R3 SE $-$ LR-B, AUPRC at 10\%",
                         r"SE R3 $-$ R1dm", "SE offset", "SE offset, range over seeds", r"GRU offset$^{\dagger}$"],
                 ["(b) Held-out domain", "R1dm SE − LR-B", "R3 SE − LR-B", "R3 SE − LR-B, AUPRC at 10%", "SE R3 − R1dm",
                  "SE offset", "SE offset, range over seeds", "GRU offset†"]))
    V = R["reversal"]; O = R["offset_analogue"]; G = R["post_hoc_not_preregistered"]["GRU10_full_logit_offset"]["values"]
    hm = lambda x: "*" if x["detected_holm"] else ""
    for d in DOM:
        a = V["reversal"]["SE-minus-LR-B"]["AUROC"]["per_fold"][d]; b = V["reversal"]["SE-minus-LR-B"]["AUPRC_p10"]["per_fold"][d]
        g = V["R3_minus_R1dm"]["SE"]["AUROC"][d]; o = O["R3"][d]
        rows.append(("row", d, d, [ci(a["R1dm"], True, hm(a["R1dm"])), ci(a["R3"], True, hm(a["R3"])),
                                  ci(b["R3"], True, hm(b["R3"])), ci(g, True, hm(g)), C(o["mean"], sign=True),
                                  rng(o["min"], o["max"]), C(G[f"R3_{d}"]["mean"], sign=True)]))
    o = O["R1_reference"]
    rows.append(("row", "R1 (reference)", "R1 (reference)", [NA, NA, NA, NA, C(o["mean"], sign=True), rng(o["min"], o["max"]),
                                                             C(G["R1"]["mean"], sign=True)]))
    # ---------------- panel c: graded coverage inside R1
    rows.append(("hdr", ["(c) Target coverage (achieved)", "AUROC", r"FPR at $\theta_{r80}$", r"recall at $\theta_{r80}$",
                         r"SE $-$ LR-B, paired", r"LR-C$^{\ddagger}$", r"GRU$^{\ddagger}$", "Drift"],
                 ["(c) Target coverage (achieved)", "AUROC", "FPR at θr80", "recall at θr80", "SE − LR-B, paired", "LR-C‡",
                  "GRU‡", "Drift"]))
    lv = R["graded_and_padding"]["graded"]["levels"]; md = R["embedding_drift"]["R1_masked"]
    for k, v in lv.items():
        draws = [x["mean"] for kk, x in md.items() if kk.split("|")[0] == k]
        pub = v["published_same_masks"]; p = v["AUROC_minus_LR-B"]
        mark = "*" if (p["lo"] > 0 or p["hi"] < 0) else ""
        lab = f"{k} ({v['coverage_mean']:.3f})"
        rows.append(("row", lab, lab, [ci(v["AUROC"]), ci(v["FPR@t_r80"]), ci(v["recall@t_r80"]), ci(p, True, mark),
                                      C(pub["LR-C"]["point"]), C(pub["GRU"]["point"]), C(sum(draws) / len(draws))]))
    # ---------------- panel d: observation-length padding
    rows.append(("hdr", ["(d) Filler characters", r"SE recall at $\theta_{r80}$", "SE AUROC", r"SE recall change",
                         r"LR-C recall$^{\ddagger}$", r"LR-B recall$^{\ddagger}$", r"GRU recall$^{\ddagger}$", ""],
                 ["(d) Filler characters", "SE recall at θr80", "SE AUROC", "SE recall change", "LR-C recall‡",
                  "LR-B recall‡", "GRU recall‡", ""]))
    P = R["graded_and_padding"]["padding"]; pub = P["published_same_protocol"]
    for k, lab in (("pad0", "0"), ("pad100", "100"), ("pad1200", "1,200")):
        v = P["levels"][k]
        chg = ci(v["recall_change_vs_pad0"], True, "*") if "recall_change_vs_pad0" in v else NA
        rows.append(("row", lab, lab, [ci(v["recall@t_r80"]), ci(v["AUROC"]), chg,
                                      C(pub["LR-C"][k]["recall@t_r80"][0]), C(pub["LR-B"][k]["recall@t_r80"][0]),
                                      C(pub["GRU"][k]["recall@t_r80"][0]), BLANK]))
    # ---------------- panel e: AgentDrift to AgentDojo
    rows.append(("hdr", ["(e) Population", "n pos / n neg", "SE transfer", r"SE transfer, cluster$^{\S}$", "SE within (UT)",
                         r"within $-$ transfer", "SE-W0 transfer", "SE-W0 within"],
                 ["(e) Population", "n pos / n neg", "SE transfer", "SE transfer, cluster§", "SE within (UT)",
                  "within − transfer", "SE-W0 transfer", "SE-W0 within"]))
    R4 = R["R4"]
    clus = {"P0": T6["decisions_plan_procedure"]["P0_UT"], "C1": T6["decisions_plan_procedure"]["C1_UT"],
            "P0F": T6["decisions_plan_procedure"]["P0F_UT"], "MB": T6["decisions_plan_procedure"]["MB_UT"]}
    def ccell(d):
        return C(d["transfer_AUROC"], d["transfer_ci"][0], d["transfer_ci"][1])
    for pop, lab in (("P0", "P0"), ("C1", "C1"), ("P0F", "P0F")):
        m = R4[pop]["by_prev"]["native"]["metrics"]; p = R4[pop]["by_prev"]["native"]["paired"]["SE_within-minus-SE_transfer"]["AUROC"]
        mark = "*" if (p["lo"] > 0 or p["hi"] < 0) else ""
        nn = f"{R4[pop]['n_pos']:,} / {R4[pop]['n_neg']:,}"
        rows.append(("row", lab, lab, [raw(nn.replace(",", "{,}"), nn), ci(m["SE_transfer"]["AUROC"]), ccell(clus[pop]),
                                      ci(m["SE_within"]["AUROC"]), ci(p, True, mark), ci(m["SE-W0_transfer"]["AUROC"]),
                                      ci(m["SE-W0_within"]["AUROC"])]))
    mb = R4["model_balanced"]; m = mb["native_counts"]["metrics"]; p = mb["native_counts"]["paired"]["SE_within-minus-SE_transfer"]["AUROC"]
    nn = f"{mb['n_pos']:,} / {mb['n_neg']:,}"
    rows.append(("row", "model-balanced", "model-balanced", [raw(nn.replace(",", "{,}"), nn), ci(m["SE_transfer"]["AUROC"]),
                                                             ccell(clus["MB"]), ci(m["SE_within"]["AUROC"]),
                                                             ci(p, True, "*" if (p["lo"] > 0 or p["hi"] < 0) else ""),
                                                             ci(m["SE-W0_transfer"]["AUROC"]), ci(m["SE-W0_within"]["AUROC"])]))
    # ---------------- panel f: world-feature support per AgentDojo suite
    rows.append(("hdr", ["(f) AgentDojo suite", "Traces", "Any address in arguments", "Any URL in arguments",
                         "Any non-zero world feature", "", "", ""],
                 ["(f) AgentDojo suite", "Traces", "Any address in arguments", "Any URL in arguments",
                  "Any non-zero world feature", "", "", ""]))
    for s, v in R4["agentdojo_feature_support_by_suite"]["all"].items():
        n = f"{v['n_traces']:,}"
        rows.append(("row", s, s, [raw(n.replace(",", "{,}"), n), C(v["share_any_email_in_args"]), C(v["share_any_url_in_args"]),
                                  C(v["share_any_nonzero_world_feature"]), BLANK, BLANK, BLANK]))
    # ---------------- panel g: R2 against the published single run
    rows.append(("hdr", ["(g) R2, threshold 0.5", "Ours, mean of 10 seeds", "Ours, min", "Ours, max",
                         r"Published, single run$^{\P}$", "", "", ""],
                 ["(g) R2, threshold 0.5", "Ours, mean of 10 seeds", "Ours, min", "Ours, max", "Published, single run¶",
                  "", "", ""]))
    Q = R["R2_paper_comparison_point"]; pubq = Q["published_reference_not_ours"]
    for k, lab in (("precision", "precision"), ("recall", "recall"), ("F1", "F1")):
        rows.append(("row", lab, lab, [C(Q[k]["mean"]), C(Q[k]["min"]), C(Q[k]["max"]), C(pubq[k]), BLANK, BLANK, BLANK]))
    for k, lab in (("benign", "flag rate, benign"), ("failed_attack", "flag rate, failed attack"),
                   ("hard_negative", "flag rate, hard negative")):
        f = Q["flag_rate"][k]
        rows.append(("row", lab, lab, [C(f["mean"]), C(f["min"]), C(f["max"]), C(pubq["flag_rate"][k]), BLANK, BLANK, BLANK]))
    return head_a_tex, head_a, rows


def wrap(h, width=14):
    """Break a long LaTeX header cell over two lines at the space nearest its middle (never inside $...$)."""
    if len(h) <= width or " " not in h:
        return h
    spaces = [i for i, ch in enumerate(h) if ch == " " and h[:i].count("$") % 2 == 0]
    if not spaces:
        return h
    k = min(spaces, key=lambda i: abs(i - len(h) / 2))
    return r"\makecell{%s\\%s}" % (h[:k], h[k + 1:])


def write(head_tex, head_md, rows):
    os.makedirs(OUT, exist_ok=True)
    head_tex = [wrap(h) for h in head_tex]
    rows = [("hdr", [wrap(c) for c in r[1]], r[2]) if r[0] == "hdr" else r for r in rows]
    tex = ["% generated by revision/sencoder_table_s19.py", " & ".join(head_tex) + r" \\", r"\midrule"]
    mdl = ["| " + " | ".join(head_md) + " |", "| " + " | ".join(["---"] * NCOL) + " |"]
    for r in rows:
        if r[0] == "hdr":
            tex += [r"\midrule", " & ".join(r"\textbf{%s}" % c if c else "" for c in r[1]) + r" \\", r"\midrule"]
            mdl.append("| " + " | ".join("**" + c + "**" if c else "" for c in r[2]) + " |")
        else:
            _, lt, lm, cells = r
            assert len(cells) == NCOL - 1, (lm, len(cells))
            tex.append(" & ".join([lt] + [c.tex() for c in cells]) + r" \\")
            mdl.append("| " + " | ".join([lm] + [c.md() for c in cells]) + " |")
    open(os.path.join(OUT, "s_sencoder.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    open(os.path.join(OUT, "s_sencoder.md"), "w", encoding="utf-8").write("\n".join(mdl) + "\n")
    json.dump({"env": "table*", "colspec": "@{}l*{7}{@{\\hspace{3pt}}c}@{}", "size": "\\scriptsize", "ncols": NCOL},
              open(os.path.join(OUT, "s_sencoder_colspec.json"), "w"), indent=1)


if __name__ == "__main__":
    R = json.load(open(os.path.join(HERE, "sencoder_results.json"), encoding="utf-8"))
    T6 = json.load(open(os.path.join(HERE, "sencoder_table6_row.json"), encoding="utf-8"))
    write(*build(R, T6))
    print("wrote", OUT)
