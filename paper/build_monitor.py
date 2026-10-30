"""Build the monitor-revised manuscript and supplement (v0.5) without touching any existing build output.

Run from agent-ids/paper:  python build_monitor.py [--synthetic]
Inputs (read only)
  ../MANUSCRIPT_SOURCE.monitor.md, ../SUPPLEMENT_SOURCE.monitor.md   (from revision/apply_monitor_revision.py)
  gen_tables_v04.py, build_manuscript.py, md2tex_v04.py, build_sencoder.py  (imported, executed in part, never modified)
  ../revision/monitor_tables/ (from revision/monitor_tables.py), ../revision/sencoder_s19/, ../revision/*.json
Outputs (all new names)
  tables_monitor/   every generated table. Changes against tables/ are limited to
                    v04_oov, v04_r4 (SE row, as in build_sencoder.py), v04_prevalence and v04_r4 (monitor rows),
                    s_sencoder split into s_sencoder (panels a to d) and s_sencoder_b (panels e to g), so that
                    Table S19 fits its page, and the new s_monitor_ladder and s_monitor_arms.
  monitor_build/    synced sources, MANUSCRIPT_DRAFT.md, SUPPLEMENT.md, build_report.json
  main_v05_monitor.tex, supplement_v05_monitor.tex
"""
import copy, json, os, re, shutil, sys, importlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REV = os.path.join(ROOT, "revision")
sys.path.insert(0, HERE); sys.path.insert(0, REV); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); A = ap.parse_args()
SUF = "_synthetic" if A.synthetic else ""
TOUT = os.path.join(HERE, "tables_monitor" + SUF); BUILD = os.path.join(HERE, "monitor_build" + SUF)
os.makedirs(TOUT, exist_ok=True); os.makedirs(BUILD, exist_ok=True)
report = {"synthetic": A.synthetic}

# ------------------------------------------------------------------ 1. tables
gt = importlib.import_module("gen_tables_v04")
gt.OUT = TOUT
import monitor_tables as mt                                   # noqa: E402
mt.SYN = A.synthetic
_write = gt.write


def write_with_monitor(name, header_tex, header_md, rows, env, colspec, size="\\scriptsize", ncols=None):
    if name == "v04_prevalence":
        rows = list(rows) + mt.prevalence_rows(gt)
    elif name == "v04_r4":
        rows = list(rows) + [gt.MID] + mt.r4_rows(gt)
    return _write(name, header_tex, header_md, rows, env, colspec, size, ncols)


gt.write = write_with_monitor
# t_oov_10seed and t_r4_with_se exactly as in build_sencoder.py (source executed, not copied by hand)
bsrc = open(os.path.join(HERE, "build_sencoder.py"), encoding="utf-8").read()
seg = bsrc[bsrc.index("G10 = json.load("):bsrc.index("funcs = [")]
ns = {"gt": gt, "json": json, "os": os, "copy": copy, "REV": REV, "HERE": HERE, "report": report}
exec(seg, ns)
funcs = [gt.t_prevalence, gt.t_r3, gt.t_thresholds, ns["t_oov_10seed"], ns["t_r4_with_se"], gt.s_breakdown, gt.s_splits, gt.s_shift,
         gt.s_content, gt.s_obslen, gt.s_keyword, gt.s_r4_controls, gt.s_intervals, gt.s_oppoints, gt.s_labelfree,
         gt.s_probes, gt.s_coverage, gt.s_loso, gt.s_gru, gt.s_m7, gt.s_thresholds_all]
for f in funcs:
    f()
gt.write = _write

# Table S19 split at panel (e), so that each part fits a page
S19 = os.path.join(REV, "sencoder_s19")
spec19 = json.load(open(os.path.join(S19, "s_sencoder_colspec.json"), encoding="utf-8"))
tex = open(os.path.join(S19, "s_sencoder.tex"), encoding="utf-8").read().splitlines()
k = next(i for i, l in enumerate(tex) if l.startswith(r"\textbf{(e) Population}"))
assert tex[k - 1] == r"\midrule" and tex[k + 1] == r"\midrule", "unexpected S19 layout"
part_a = tex[:k - 1]; part_b = [tex[0], tex[k], r"\midrule"] + tex[k + 2:]
while part_a and part_a[-1] == r"\midrule":
    part_a.pop()
open(os.path.join(TOUT, "s_sencoder.tex"), "w", encoding="utf-8").write("\n".join(part_a) + "\n")
open(os.path.join(TOUT, "s_sencoder_b.tex"), "w", encoding="utf-8").write("\n".join(part_b) + "\n")
mdl = open(os.path.join(S19, "s_sencoder.md"), encoding="utf-8").read().splitlines()
km = next(i for i, l in enumerate(mdl) if l.startswith("| **(e) Population**"))
ncol = mdl[0].count("|") - 1
open(os.path.join(TOUT, "s_sencoder.md"), "w", encoding="utf-8").write("\n".join(mdl[:km]) + "\n")
open(os.path.join(TOUT, "s_sencoder_b.md"), "w", encoding="utf-8").write(
    "\n".join([mdl[km].replace("**", ""), "| " + " | ".join(["---"] * ncol) + " |"] + mdl[km + 1:]) + "\n")
gt.COLSPECS["s_sencoder"] = spec19; gt.COLSPECS["s_sencoder_b"] = dict(spec19)
report["s19_split"] = {"rows_part_a": len(part_a), "rows_part_b": len(part_b)}

# monitor supplement tables
MT = os.path.join(REV, "monitor_tables", "synthetic" if A.synthetic else "")
for nm in ("s_monitor_ladder", "s_monitor_arms"):
    for ext in (".tex", ".md"):
        shutil.copy(os.path.join(MT, nm + ext), os.path.join(TOUT, nm + ext))
gt.COLSPECS.update(json.load(open(os.path.join(MT, "colspecs.json"), encoding="utf-8")))
json.dump(gt.COLSPECS, open(os.path.join(TOUT, "v04_colspecs.json"), "w", encoding="utf-8"), indent=1)

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
allowed = {"v04_oov.tex", "v04_oov.md", "v04_r4.tex", "v04_r4.md", "v04_prevalence.tex", "v04_prevalence.md", "v04_colspecs.json"}
assert set(changed) <= allowed, f"unexpected table changes {set(changed) - allowed}"
assert set(new) <= {"s_sencoder.tex", "s_sencoder.md", "s_sencoder_b.tex", "s_sencoder_b.md", "s_monitor_ladder.tex", "s_monitor_ladder.md",
                    "s_monitor_arms.tex", "s_monitor_arms.md"}, new

# ------------------------------------------------------------------ 2. sync tables into copies of the monitor sources
mismatch = {}
for src_name, dst_name in (("MANUSCRIPT_SOURCE.monitor%s.md" % SUF, "MANUSCRIPT_SOURCE.md"), ("SUPPLEMENT_SOURCE.monitor%s.md" % SUF, "SUPPLEMENT_SOURCE.md")):
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
report["sync_differences_vs_monitor_sources"] = mismatch

# ------------------------------------------------------------------ 3. numbered Markdown
bm = importlib.import_module("build_manuscript")
bm.ROOT = BUILD
extra_refs = json.load(open(os.path.join(REV, "references_monitor.json"), encoding="utf-8"))
assert not set(extra_refs) & set(bm.REFS), "monitor reference keys clash"
bm.REFS.update(extra_refs); report["added_references"] = sorted(extra_refs)
bm.main()

# ------------------------------------------------------------------ 4. LaTeX
code = open(os.path.join(HERE, "md2tex_v04.py"), encoding="utf-8").read()
subs = [('os.path.join(ROOT, "MANUSCRIPT_DRAFT.md" if MODE == "main" else "SUPPLEMENT.md")',
         'os.path.join(BUILD_DIR, "MANUSCRIPT_DRAFT.md" if MODE == "main" else "SUPPLEMENT.md")'),
        ('os.path.join(HERE, "main_v04.tex" if MODE == "main" else "supplement_v04.tex")',
         'os.path.join(HERE, "main_v05_monitor%s.tex" if MODE == "main" else "supplement_v05_monitor%s.tex")' % (SUF, SUF)),
        ('os.path.join(HERE, "tables", "v04_colspecs.json")', 'os.path.join(HERE, "%s", "v04_colspecs.json")' % os.path.basename(TOUT)),
        ('os.path.join(HERE, "tables", name + ".tex")', 'os.path.join(HERE, "%s", name + ".tex")' % os.path.basename(TOUT))]
for a, b in subs:
    assert code.count(a) == 1, a
    code = code.replace(a, b)
for mode in ("main", "supp"):
    sys.argv = ["md2tex_v04.py", mode]
    exec(compile(code, os.path.join(HERE, "md2tex_v04.py"), "exec"),
         {"__file__": os.path.join(HERE, "md2tex_v04.py"), "__name__": "__main__", "BUILD_DIR": BUILD})
json.dump(report, open(os.path.join(BUILD, "build_report.json"), "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(report, indent=1, default=float))
