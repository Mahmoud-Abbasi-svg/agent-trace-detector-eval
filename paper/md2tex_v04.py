"""Render the v0.4 Markdown manuscript or supplement as IEEEtran LaTeX.

  python md2tex_v04.py main   -> reads ../MANUSCRIPT_DRAFT.md, writes main_v04.tex
  python md2tex_v04.py supp   -> reads ../SUPPLEMENT.md,       writes supplement_v04.tex

Both inputs are produced by build_manuscript.py (citations already numbered). Generated tables are
read from tables/<name>.tex with the env, column spec and size in tables/v04_colspecs.json; literal
Markdown tables get proportional paragraph columns. Figures use the PDF next to each PNG, and a figure
spans both columns when its PDF is wider than 5 in. Differences from md2tex.py (v0.3): no draft header,
footnote or revision appendix (review m21); the indicator is written \\mathbf{1} in the source (m25);
S-numbered tables, figures and sections for the supplement.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
SRC = os.path.join(ROOT, "MANUSCRIPT_DRAFT.md" if MODE == "main" else "SUPPLEMENT.md")
OUT = os.path.join(HERE, "main_v04.tex" if MODE == "main" else "supplement_v04.tex")
MD = open(SRC, encoding="utf-8").read()
# table floats: top of page in the main paper; the table-heavy supplement may also use float pages
PLACE = "[!t]" if MODE == "main" else "[!tp]"
SPECS = json.load(open(os.path.join(HERE, "tables", "v04_colspecs.json"), encoding="utf-8"))

UNI = {"—": "---", "–": "--", "−": "$-$", "≤": "$\\le$", "≥": "$\\ge$", "≈": "$\\approx$", "→": "$\\rightarrow$",
       "±": "$\\pm$", "†": "$^{\\dagger}$", "‡": "$^{\\ddagger}$", "×": "$\\times$", "κ": "$\\kappa$", "τ": "$\\tau$",
       "θ": "$\\theta$", "π": "$\\pi$", "Δ": "$\\Delta$", "↓": "$\\downarrow$", "↑": "$\\uparrow$", "§": "\\S{}",
       "’": "'", "‘": "`", "\u00a0": "~", "·": "$\\cdot$", "≠": "$\\neq$", "∈": "$\\in$", "μ": "$\\mu$"}


def esc_text(s):
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("^", "\\^{}"), ("~", "\\textasciitilde{}")]:
        s = s.replace(a, b)
    s = s.replace("<", "\\textless{}").replace(">", "\\textgreater{}")
    for a, b in UNI.items():
        s = s.replace(a, b)
    return s


def inline(s):
    out, pos = [], 0
    for m in re.finditer(r"(\$[^$]+\$)|(`[^`]+`)", s):
        out.append(("t", s[pos:m.start()]))
        out.append(("m", m.group(1)) if m.group(1) else ("c", m.group(2)[1:-1]))
        pos = m.end()
    out.append(("t", s[pos:]))
    res = []
    for kind, x in out:
        if kind == "m":
            res.append(x.replace("\\mathbb{1}", "\\mathbf{1}"))
        elif kind == "c":
            res.append("\\texttt{" + esc_text(x).replace("-", "-\\allowbreak{}") + "}")
        else:
            x = esc_text(x.replace("\\*", "\x00"))   # Markdown-escaped asterisk is a literal "*", not emphasis
            x = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", x)
            x = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", x)
            x = re.sub(r'"([^"]+)"', r"``\1''", x)
            # citation ranges "[a]–[b]" (en dash already escaped to --) become one \cite so cite.sty prints [a]–[b]
            x = re.sub(r"\[(\d{1,3})\]\s*--\s*\[(\d{1,3})\]",
                       lambda m: "\\cite{" + ",".join(f"r{k}" for k in range(int(m.group(1)), int(m.group(2)) + 1)) + "}", x)
            x = re.sub(r"\[(\d{1,3})\]", r"\\cite{r\1}", x)
            # long hexadecimal digests (SHA-256) may break every 16 characters
            x = re.sub(r"\b[0-9a-f]{32,}\b", lambda m: "\\allowbreak{}".join(m.group(0)[k:k + 16] for k in range(0, len(m.group(0)), 16)), x)
            x = x.replace("\\textless{}UNK\\textgreater{}", "\\unk{}").replace("\x00", "*")
            res.append(x)
    s = "".join(res)
    return s.replace("\\texttt{\\unk{}}", "\\unk{}").replace("\\texttt{\\textless{}UNK\\textgreater{}}", "\\unk{}")


def titlecase(s):
    small = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "versus", "under", "with", "from"}
    out = []
    for i, w in enumerate(s.lower().split()):
        parts = [p if (i > 0 and p in small) else p[:1].upper() + p[1:] for p in w.split("-")]
        out.append("-".join(parts))
    s = " ".join(out)
    for a, b in [("Llm", "LLM"), ("Gru", "GRU"), ("Agentdrift", "AgentDrift"), ("Agentdojo", "AgentDojo"),
                 ("Lr-", "LR-"), ("Tf-idf", "TF-IDF"), ("Auroc", "AUROC"), ("Auprc", "AUPRC"), ("Oov", "OOV"),
                 ("Dos", "DoS"), ("Ieee", "IEEE"), ("Gpt", "GPT"), ("Ii", "II")]:
        s = re.sub(r"\b%s" % re.escape(a), b, s)
    return s


P = lambda w: ">{\\raggedright\\arraybackslash}p{%.2fcm}" % w


def md_table(lines, env):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|[-:|\s]+\|$", l.strip())]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    total = 17.4 if env == "table*" else 8.3
    total -= 0.141 * (n - 1)   # 2 pt \tabcolsep on each side of a column gap
    lens = [max(3.0, sum(min(len(r[j]), 80) for r in rows) / len(rows)) for j in range(n)]
    # a column must hold its longest unbreakable word (about 0.12 cm per character at \scriptsize, bold headers wider)
    def longest(j):
        toks = [(t, i == 0) for i, r in enumerate(rows) for t in re.split(r"(?<=[-/])|\s+", re.sub(r"\[\d+\]|`", "", r[j])) if t]
        return max([len(t) * (1.1 if hdr else 1.0) for t, hdr in toks] or [0])
    # only the first column (names, which TeX cannot hyphenate) gets a word-based minimum
    minw = [max(0.9, 0.105 * longest(j) + 0.1) if j == 0 else 0.9 for j in range(n)]
    widths = [total * L / sum(lens) for L in lens]
    for _ in range(10):   # raise narrow columns to their minimum and shrink the others proportionally
        fixed = [j for j in range(n) if widths[j] < minw[j] - 1e-9]
        if not fixed:
            break
        for j in fixed:
            widths[j] = minw[j]
        free = [j for j in range(n) if widths[j] > minw[j] + 1e-9]
        excess = sum(widths) - total
        tot_free = sum(widths[j] for j in free)
        for j in free:
            widths[j] -= excess * widths[j] / tot_free
    scale = total / sum(widths); widths = [w * scale for w in widths]
    spec = "@{}" + "".join(P(w) for w in widths) + "@{}"
    # 7 pt type on a 7.5 pt baseline (IEEEtran \scriptsize uses 8 pt) keeps the largest literal tables within one page
    size = "\\fontsize{7}{7.5}\\selectfont\\sloppy" if env == "table*" else "\\scriptsize"
    body = [" & ".join("\\textbf{%s}" % inline(c) for c in rows[0]) + " \\\\", "\\midrule"]
    body += [" & ".join(inline(c) for c in r) + " \\\\" for r in rows[1:]]
    return "{%s\\renewcommand{\\arraystretch}{1.0}\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{%s}\n\\toprule\n%s\n\\bottomrule\n\\end{tabular}}" % (
        size, spec, "\n".join(body))


def gen_tabular(name):
    sp = SPECS[name]
    body = open(os.path.join(HERE, "tables", name + ".tex"), encoding="utf-8").read()
    body = "\n".join(l for l in body.splitlines() if not l.startswith("%"))
    return sp.get("env", "table*"), "{%s\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{%s}\n\\toprule\n%s\n\\bottomrule\n\\end{tabular}}" % (
        sp.get("size", "\\scriptsize"), sp["colspec"], body)


def note(t):
    t = t.strip()
    if t.startswith("*") and t.endswith("*"):
        t = t[1:-1]
    return "\\par\\vspace{2pt}{\\scriptsize\\raggedright " + inline(t) + "\\par}"


def fig_wide(stem):
    try:
        import fitz
        r = fitz.open(os.path.join(HERE, stem + ".pdf"))[0].rect
        return r.width / 72.0 > 5.0
    except Exception:
        return False


def listblock(b):
    items = [re.sub(r"^\s*[-*]\s+", "", l) for l in b.splitlines() if re.match(r"^\s*[-*]\s+", l)]
    return "\\begin{itemize}\n" + "\n".join("\\item " + inline(x) for x in items) + "\n\\end{itemize}"


blocks = re.split(r"\n\s*\n", MD.strip())
out, refs, pending, cont = [], [], None, False
title = author = affil = None
in_refs = in_appx = False
i = 0
while i < len(blocks):
    b = blocks[i].strip(); i += 1
    if not b or b == "---" or b.startswith(">"):
        continue
    if b.startswith("# ") and title is None:
        title = b[2:].strip(); continue
    if title and author is None and re.match(r"^\*\*[^*]+\*\*$", b) and b not in ("**Abstract**", "**Index Terms**"):
        author = b.strip("*"); continue
    if author and affil is None and b.startswith("*") and "e-mail" in b:
        affil = b.strip("*").strip(); continue
    if b == "**Abstract**":
        out.append("\\begin{abstract}\n" + inline(blocks[i].strip().replace("\n", " ")) + "\n\\end{abstract}"); i += 1; continue
    if b == "**Index Terms**":
        out.append("\\begin{IEEEkeywords}\n" + inline(blocks[i].strip().rstrip(".")) + ".\n\\end{IEEEkeywords}"); i += 1; continue
    if b.startswith("## "):
        h = b[3:].strip()
        if h.upper() == "REFERENCES":
            in_refs = True; continue
        if h.upper().startswith("APPENDIX"):
            in_refs = False; in_appx = True; out.append("\\appendices"); continue
        h = re.sub(r"^(S?[IVX]+|S?\d+)\.\s+", "", h)
        out.append("\\section{" + titlecase(h) + "}"); continue
    if b.startswith("### "):
        h = re.sub(r"^(S?\d+\.)?[A-Z]\.\s+", "", b[4:].strip())
        out.append(("\\section{" if in_appx else "\\subsection{") + inline(h) + "}"); continue
    if in_refs:
        if b.strip() == "<!-- REFERENCES -->":
            continue
        for chunk in re.split(r"\n(?=\[\d+\]\s)", b):
            m = re.match(r"\[(\d+)\]\s+(.*)", chunk.strip(), re.S)
            if m:
                refs.append((m.group(1), inline(m.group(2).replace("\n", " "))))
        continue
    if b.startswith("**TABLE "):
        ls = b.splitlines()
        cap = ls[1].strip("*").strip() if len(ls) > 1 else ""
        pending = cap
        cont = "CONTINUED" in ls[0].upper()   # "**TABLE S3 (CONTINUED)**" keeps the number of the previous table
        rest = "\n".join(ls[2:]).strip()
        if rest:
            blocks.insert(i, rest)
        continue
    if pending is not None and (b.startswith("|") or b.startswith("<!-- BEGIN")):
        cap = pending; pending = None
        lines = b.splitlines()
        nt = ""
        if b.startswith("|"):
            tl = [l for l in lines if l.strip().startswith("|")]
            tail = [l for l in lines if not l.strip().startswith("|")]
            if tail:
                nt = " ".join(tail)
            elif i < len(blocks) and blocks[i].strip().startswith("*") and not blocks[i].strip().startswith("**"):
                nt = blocks[i].strip(); i += 1
            ncol = len(tl[0].strip().strip("|").split("|"))
            # in the supplement, long literal tables span both columns even with few columns
            env = "table*" if ncol >= 4 or (MODE != "main" and len("".join(tl)) > 1500) else "table"
            tab = md_table(tl, env)
        else:
            name = re.match(r"<!-- BEGIN ([\w\-]+) -->", b).group(1)
            env, tab = gen_tabular(name)
            m = re.search(r"<!-- END %s -->\s*\n?(.*)" % re.escape(name), b, re.S)
            nt = m.group(1).strip() if m and m.group(1).strip() else ""
            if not nt and i < len(blocks) and blocks[i].strip().startswith("*") and not blocks[i].strip().startswith("**"):
                nt = blocks[i].strip(); i += 1
        out.append(("\\addtocounter{table}{-1}\n" if cont else "") +
                   "\\begin{%s}%s\n\\centering\n\\caption{%s}\n%s%s\n\\end{%s}" % (env, PLACE, inline(cap), tab, ("\n" + note(nt)) if nt else "", env))
        cont = False
        continue
    m = re.match(r"!\[Fig\. (S?\d+)\]\((?:paper/)?([\w\-]+)\.png\)", b)
    if m:
        stem = m.group(2)
        cap = blocks[i].strip() if i < len(blocks) else ""; i += 1
        cap = re.sub(r"^\*Fig\. S?\d+\.\s*", "", cap).rstrip("*")
        wide = fig_wide(stem); env = "figure*" if wide else "figure"
        out.append("\\begin{%s}[!t]\n\\centering\n\\includegraphics[width=%s]{%s.pdf}\n\\caption{%s}\n\\end{%s}" % (
            env, "\\textwidth" if wide else "\\columnwidth", stem, inline(cap), env))
        continue
    if b.startswith("$$"):
        eq = re.sub(r"\\tag\{[^}]*\}", "", b.strip().strip("$")).strip().replace("\\mathbb{1}", "\\mathbf{1}")
        out.append("\\begin{equation}\n" + eq + "\n\\end{equation}"); continue
    if re.match(r"^\s*[-*]\s+", b) and all(re.match(r"^\s*[-*]\s+", l) or l.startswith("  ") for l in b.splitlines()):
        out.append(listblock(b)); continue
    if b.startswith("<!--"):
        continue
    if re.match(r"^\d+\.\s+", b) and all(re.match(r"^\d+\.\s+", l) or l.startswith("  ") for l in b.splitlines()):
        items = re.split(r"\n(?=\d+\.\s)", b)
        out.append("\\begin{enumerate}\n" + "\n".join("\\item " + inline(re.sub(r"^\d+\.\s+", "", x).replace("\n", " ")) for x in items) + "\n\\end{enumerate}")
        continue
    out.append(inline(b.replace("\n", " ")))

body = "\n\n".join(out)
body = re.sub(r"(\\end\{equation\})\n\n(?=[a-z(])", r"\1\n\\noindent ", body)
body = re.sub(r"\n\n(\\begin\{equation\})", r"\n\1", body)
# bare web addresses in running text become \url{} so they can break across lines
body = re.sub(r"(?<=[\s(])(https://github\.com/[A-Za-z0-9./_-]*[A-Za-z0-9/_-])", r"\\url{\1}", body)
if refs:
    bib = "\\begin{thebibliography}{99}\n" + "\n".join(("\\bibitem{r%s} %s" % (n, t)) if MODE == "main" else ("\\bibitem[%s]{r%s} %s" % (n, n, t)) for n, t in refs) + "\n\\end{thebibliography}"
    k = body.find("\\appendices")
    body = body[:k] + bib + "\n\n" + body[k:] if k >= 0 else body + "\n\n" + bib

short = "Evaluating Baseline Detectors of Hijacked LLM Agents"
supp_cmds = "\\renewcommand{\\thetable}{S\\arabic{table}}\n\\renewcommand{\\thefigure}{S\\arabic{figure}}\n\\renewcommand{\\thesection}{S\\arabic{section}}\n\\renewcommand{\\theequation}{S\\arabic{equation}}\n" if MODE != "main" else ""
PRE = r"""\documentclass[journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{makecell}
\usepackage{cite}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\newcommand{\unk}{\texttt{<UNK>}}
\newcommand{\cit}[3]{\begin{tabular}[c]{@{}c@{}}$#1$\\[-2pt]{\scriptsize$[#2,\,#3]$}\end{tabular}}
\hyphenation{Agent-Drift Agent-Dojo Drift-Net Traj-AD Llama-Fire-wall Sentinel-Agent}
""" + supp_cmds + r"""\begin{document}
\title{%s}
\author{%s%%
\thanks{%s}}
\markboth{Manuscript submitted to IEEE Transactions}{%s}
\maketitle
""" % (inline(title or ""), inline(author or "Mahmoud Abbasi"), inline(affil or ""), short)
open(OUT, "w", encoding="utf-8").write(PRE + body + "\n\n\\end{document}\n")
print(f"wrote {os.path.basename(OUT)}: {len(body)} chars, {len(refs)} references")
