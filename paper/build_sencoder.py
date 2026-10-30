"""Build the SE-revised manuscript and supplement without touching the existing build outputs.

Run from agent-ids/paper:  python build_sencoder.py
Inputs (read only)
  ../MANUSCRIPT_SOURCE.revised.md, ../SUPPLEMENT_SOURCE.revised.md   (from revision/apply_sencoder_revision.py)
  gen_tables_v04.py, build_manuscript.py, md2tex_v04.py              (imported or executed, never modified)
  ../revision/gru10_tool_offset.json, ../revision/sencoder_table6_row.json, ../revision/sencoder_s19/s_sencoder.*
Outputs (all new names)
  tables_sencoder/            every generated table, with v04_oov and v04_r4 changed and s_sencoder added
  sencoder_build/             synced sources, MANUSCRIPT_DRAFT.md, SUPPLEMENT.md, build_report.json
  main_v04_sencoder.tex, supplement_v04_sencoder.tex   (written next to the figures so relative paths resolve)
Changes to generated tables
  v04_oov  GRU tool-pathway offset column = ten-seed means (same estimator as oov_mechanism.py s4).
  v04_r4   SE row added. Holm correction over six representations (published p values plus SE's).
  s_sencoder  copied from revision/sencoder_s19 (written by revision/sencoder_table_s19.py).
Every other generated table must come out byte-identical to tables/, which the script checks.
"""
import copy, json, os, re, shutil, sys, importlib

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
TOUT = os.path.join(HERE, "tables_sencoder"); BUILD = os.path.join(HERE, "sencoder_build")
os.makedirs(TOUT, exist_ok=True); os.makedirs(BUILD, exist_ok=True)
REV = os.path.join(ROOT, "revision")
report = {}

# ------------------------------------------------------------------ 1. tables
gt = importlib.import_module("gen_tables_v04")
gt.OUT = TOUT
G10 = json.load(open(os.path.join(REV, "gru10_tool_offset.json"), encoding="utf-8"))
T6 = json.load(open(os.path.join(REV, "sencoder_table6_row.json"), encoding="utf-8"))


def t_oov_10seed():
    orig = gt.OOV
    oov = copy.deepcopy(orig)
    for s, v in G10["offset_mean_10_seeds"].items():
        oov["s4_gru_offset"]["settings"][s]["GRU"]["mean_over_seeds"]["offset_neg"] = v
    gt.OOV = oov
    try:
        gt.t_oov()
    finally:
        gt.OOV = orig


def t_r4_with_se():
    cf = gt.R4D["M6"]["configs"]
    holm5 = gt.R4D["M6"]["decisions"]["holm_within_minus_transfer_cluster"]["by_config"]["P0_UT"]
    dets = ["LR-A", "LR-B", "LR-C", "GRU", "TFIDF"]
    se = T6["P0_UT"]
    ps = [holm5[d]["p_boot"] for d in dets] + [se["within_minus_transfer"]["p_boot"]]
    adj = gt.holm(ps) if hasattr(gt, "holm") else None
    if adj is None:   # Holm step-down, as in r4_deep.holm
        import numpy as np
        p = np.asarray(ps, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
        for k, i in enumerate(order):
            run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    det6 = {d: bool(adj[i] < 0.05) for i, d in enumerate(dets + ["SE"])}
    report["tableVI_holm6"] = {"p_boot": dict(zip(dets + ["SE"], ps)), "p_holm": dict(zip(dets + ["SE"], [float(x) for x in adj])),
                               "detected_changes_vs_holm5": [d for d in dets if det6[d] != holm5[d]["detected_holm"]]}

    def pc(cfg, key, metric="AUROC"):
        c = cf[cfg]; p = c["point"][key][metric]; _, lo, hi = c["cluster"]["ci"][key][metric]
        return gt.CP(p, lo, hi)

    rows = []
    for d in dets:
        dl = cf["P0_UT"]["cluster"]["paired"]["within:%s-minus-transfer:%s" % (d, d)]["AUROC"]
        cells = [pc("P0_UT", "transfer:" + d), pc("P0_UT", "within:" + d),
                 gt.CP(dl["mean"], dl["lo"], dl["hi"], sign=True, mark="*" if det6[d] else ""),
                 pc("P0_IT", "within:" + d), pc("P0F_UT", "transfer:" + d), pc("P0_UT", "within:" + d, "APw10"),
                 gt.R(gt.outcome_text(d))]
        rows.append(gt.row(gt.LAB[d], cells))
    g = se["within_minus_transfer"]; dpf = T6["P0F_UT"]["transfer_AUROC"]
    rows.append(gt.row("SE (reimplementation)", [
        gt.CP(*se["transfer_AUROC"]), gt.CP(*se["within_AUROC"]),
        gt.CP(g["mean"], g["lo"], g["hi"], sign=True, mark="*" if det6["SE"] else ""),
        gt.R("not run"), gt.CP(*dpf), gt.CP(*se["within_APw10"]), gt.R("Transfer failure under UT, IT not run")]))
    # header strings are those of gen_tables_v04.t_r4 (unchanged)
    src = open(os.path.join(HERE, "gen_tables_v04.py"), encoding="utf-8").read()
    i = src.index("def t_r4():"); blk = src[i:src.index("# ===", i)]
    ns = {}
    exec(blk[blk.index("    ht = ("):blk.index("    write(")].replace("\n    ", "\n").strip(), {}, ns)
    gt.write("v04_r4", ns["ht"], ns["hm"], rows, "table*", "@{}l*{6}{c}>{\\raggedright\\arraybackslash}p{3.3cm}@{}")


funcs = [gt.t_prevalence, gt.t_r3, gt.t_thresholds, t_oov_10seed, t_r4_with_se, gt.s_breakdown, gt.s_splits, gt.s_shift,
         gt.s_content, gt.s_obslen, gt.s_keyword, gt.s_r4_controls, gt.s_intervals, gt.s_oppoints, gt.s_labelfree,
         gt.s_probes, gt.s_coverage, gt.s_loso, gt.s_gru, gt.s_m7, gt.s_thresholds_all]
for f in funcs:
    f()
S19 = os.path.join(REV, "sencoder_s19")
for ext in (".tex", ".md"):
    shutil.copy(os.path.join(S19, "s_sencoder" + ext), os.path.join(TOUT, "s_sencoder" + ext))
gt.COLSPECS["s_sencoder"] = json.load(open(os.path.join(S19, "s_sencoder_colspec.json"), encoding="utf-8"))
json.dump(gt.COLSPECS, open(os.path.join(TOUT, "v04_colspecs.json"), "w", encoding="utf-8"), indent=1)

# identity check against the existing generated tables
PUB = os.path.join(HERE, "tables")
same, changed, new = [], [], []
for fn in sorted(os.listdir(TOUT)):
    a = os.path.join(PUB, fn); b = os.path.join(TOUT, fn)
    if not os.path.exists(a):
        new.append(fn)
    elif open(a, "rb").read() == open(b, "rb").read():
        same.append(fn)
    else:
        changed.append(fn)
report["tables"] = {"identical": len(same), "changed": changed, "new": new}
allowed = {"v04_oov.tex", "v04_oov.md", "v04_r4.tex", "v04_r4.md", "v04_colspecs.json"}
assert set(changed) <= allowed, f"unexpected table changes {set(changed) - allowed}"

# ------------------------------------------------------------------ 2. sync tables into copies of the revised sources
mismatch = {}
for src_name, dst_name in (("MANUSCRIPT_SOURCE.revised.md", "MANUSCRIPT_SOURCE.md"), ("SUPPLEMENT_SOURCE.revised.md", "SUPPLEMENT_SOURCE.md")):
    text = open(os.path.join(ROOT, src_name), encoding="utf-8").read()
    for name in re.findall(r"<!-- BEGIN ([\w\-]+) -->", text):
        table = open(os.path.join(TOUT, name + ".md"), encoding="utf-8").read().strip()
        old = re.search(r"<!-- BEGIN %s -->\n(.*?)\n<!-- END %s -->" % (re.escape(name), re.escape(name)), text, re.S).group(1)
        if old.strip() != table:
            mismatch.setdefault(src_name, []).append(name)
        text, n = re.subn(r"(<!-- BEGIN %s -->).*?(<!-- END %s -->)" % (re.escape(name), re.escape(name)),
                          lambda m_: m_.group(1) + "\n" + table + "\n" + m_.group(2), text, flags=re.S)
        assert n == 1, (src_name, name)
    open(os.path.join(BUILD, dst_name), "w", encoding="utf-8").write(text)
report["sync_differences_vs_revised_sources"] = mismatch

# ------------------------------------------------------------------ 3. numbered Markdown
bm = importlib.import_module("build_manuscript")
bm.ROOT = BUILD
bm.main()

# ------------------------------------------------------------------ 4. LaTeX
code = open(os.path.join(HERE, "md2tex_v04.py"), encoding="utf-8").read()
subs = [('os.path.join(ROOT, "MANUSCRIPT_DRAFT.md" if MODE == "main" else "SUPPLEMENT.md")',
         'os.path.join(BUILD_DIR, "MANUSCRIPT_DRAFT.md" if MODE == "main" else "SUPPLEMENT.md")'),
        ('os.path.join(HERE, "main_v04.tex" if MODE == "main" else "supplement_v04.tex")',
         'os.path.join(HERE, "main_v04_sencoder.tex" if MODE == "main" else "supplement_v04_sencoder.tex")'),
        ('os.path.join(HERE, "tables", "v04_colspecs.json")', 'os.path.join(HERE, "tables_sencoder", "v04_colspecs.json")'),
        ('os.path.join(HERE, "tables", name + ".tex")', 'os.path.join(HERE, "tables_sencoder", name + ".tex")')]
for a, b in subs:
    assert code.count(a) == 1, a
    code = code.replace(a, b)
for mode in ("main", "supp"):
    sys.argv = ["md2tex_v04.py", mode]
    exec(compile(code, os.path.join(HERE, "md2tex_v04.py"), "exec"),
         {"__file__": os.path.join(HERE, "md2tex_v04.py"), "__name__": "__main__", "BUILD_DIR": BUILD})
json.dump(report, open(os.path.join(BUILD, "build_report.json"), "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(report, indent=1, default=float))
