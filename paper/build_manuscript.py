"""Resolve citation keys and build the numbered Markdown manuscript and supplement.

Sources (edited by hand):   ../MANUSCRIPT_SOURCE.md, ../SUPPLEMENT_SOURCE.md
References database:        ../revision/references_v04.json  {key: {"ieee": "...", ...}}
Outputs (generated):        ../MANUSCRIPT_DRAFT.md, ../SUPPLEMENT.md

Citation syntax in the sources: [@key] or [@key1; @key2; @key3].
Numbers follow IEEE practice: order of first appearance in the main manuscript (the supplement
reuses the main numbering and appends keys that only it cites). Groups print as "[4], [6], [8]",
and runs of three or more consecutive numbers print as "[2]–[12]".
The line "<!-- REFERENCES -->" in a source is replaced by its numbered reference list.
Run from agent-ids/paper after sync_tables.py has filled the table markers of the sources:
  python build_manuscript.py
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REFS = json.load(open(os.path.join(ROOT, "revision", "references_v04.json"), encoding="utf-8"))
CITE = re.compile(r"\[(@[\w:.\-]+(?:\s*[;,]\s*@[\w:.\-]+)*)\]")


def keys_in(text):
    out = []
    for m in CITE.finditer(text):
        for k in re.split(r"\s*[;,]\s*", m.group(1)):
            k = k.lstrip("@")
            if k not in out:
                out.append(k)
    return out


def fmt(nums):
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"[{nums[i]}]–[{nums[j]}]")
        else:
            parts.extend(f"[{n}]" for n in nums[i:j + 1])
        i = j + 1
    return ", ".join(parts)


def resolve(text, order):
    def sub(m):
        ks = [k.lstrip("@") for k in re.split(r"\s*[;,]\s*", m.group(1))]
        missing = [k for k in ks if k not in order]
        if missing:
            raise KeyError(f"citation keys not in the reference database or order: {missing}")
        return fmt([order[k] for k in ks])
    return CITE.sub(sub, text)


def ref_list(order, only=None):
    items = sorted(order.items(), key=lambda kv: kv[1])
    lines = [f"[{n}] {REFS[k]['ieee']}" for k, n in items if only is None or k in only]
    return "\n\n".join(lines)


def main():
    src = open(os.path.join(ROOT, "MANUSCRIPT_SOURCE.md"), encoding="utf-8").read()
    main_keys = keys_in(src)
    unknown = [k for k in main_keys if k not in REFS]
    if unknown:
        sys.exit(f"unknown citation keys in MANUSCRIPT_SOURCE.md: {unknown}")
    order = {k: i + 1 for i, k in enumerate(main_keys)}
    out = resolve(src, order).replace("<!-- REFERENCES -->", ref_list(order))
    open(os.path.join(ROOT, "MANUSCRIPT_DRAFT.md"), "w", encoding="utf-8").write(out)
    print(f"MANUSCRIPT_DRAFT.md: {len(order)} references")

    sp = os.path.join(ROOT, "SUPPLEMENT_SOURCE.md")
    if os.path.exists(sp):
        sup = open(sp, encoding="utf-8").read()
        extra = [k for k in keys_in(sup) if k not in order]
        unknown = [k for k in extra if k not in REFS]
        if unknown:
            sys.exit(f"unknown citation keys in SUPPLEMENT_SOURCE.md: {unknown}")
        sorder = dict(order); nxt = len(order)
        for k in extra:
            nxt += 1; sorder[k] = nxt
        cited = set(keys_in(sup))
        sout = resolve(sup, sorder).replace("<!-- REFERENCES -->", ref_list(sorder, only=cited))
        open(os.path.join(ROOT, "SUPPLEMENT.md"), "w", encoding="utf-8").write(sout)
        print(f"SUPPLEMENT.md: {len(cited)} references cited ({len(extra)} only in the supplement)")


if __name__ == "__main__":
    main()
