"""Reproducibility checks for the revision (review comment m14). Writes, under revision/:
  requirements-lock.txt   exact installed versions (importlib.metadata) of the third-party packages the code imports,
                          with their installed runtime dependencies
  repro_checks.json       (1) result of tests/test_agentdojo_alignment.py, run as a script and under pytest;
                          (2) a parse check (ast) of every project script; (3) per script: SHA-256, the environment
                          variables it reads, its command-line flags, and every line that sets a seed, a bootstrap size,
                          a LogisticRegression / IsolationForest / TfidfVectorizer configuration (with line numbers),
                          so that REPRODUCIBILITY.md can be checked against the code.
Nothing is fitted and no existing file is modified. The test run uses -B and PYTHONDONTWRITEBYTECODE=1 and pytest's
cache provider is disabled, so that no bytecode or cache files are written next to the scripts.
Run (Git Bash, from agent-ids/):
  AGENTDRIFT=<AgentDrift clone> AGENTDOJO=<agentdojo repo> C:/Python310/python.exe -B revision/repro_checks.py
"""
import os, sys, re, ast, json, time, hashlib, platform, subprocess, importlib.metadata as md

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LOCK = os.path.join(HERE, "requirements-lock.txt"); OUT = os.path.join(HERE, "repro_checks.json")
DIRECT = ["numpy", "scipy", "scikit-learn", "torch", "matplotlib"]      # third-party imports of the project code
OPTIONAL = ["pytest"]                                                   # only to run the unit test under pytest
STDLIB_NOTE = ("standard library only otherwise: os, sys, re, json, glob, math, time, pickle, argparse, hashlib, "
               "collections, itertools, copy, statistics, platform, warnings, subprocess, shutil, datetime")
PATTERNS = {
    "env": r"os\.environ(\.get)?\s*[\[\(]\s*[\"']([A-Z_]+)",
    "seed": r"seed|random_state|default_rng|manual_seed|np\.random\.seed|RandomState|Generator\(",
    "bootstrap": r"bootstrap\(|\bB\s*=\s*\d|\bB13\b\s*=|\bB4\b\s*=|\bN_SEED\s*=|\bBOOT\w*\s*=",
    "estimator": r"LogisticRegression\(|IsolationForest\(|TfidfVectorizer\(|GridSearchCV\(|StratifiedKFold\(|KFold\(",
    "cli": r"add_argument\(|sys\.argv",
}


def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def dist_tree(names):
    seen = {}

    def walk(n):
        k = n.lower().replace("_", "-")
        if k in seen:
            return
        try:
            d = md.distribution(n)
        except md.PackageNotFoundError:
            seen[k] = None; return
        seen[k] = d.metadata["Name"], d.version
        for req in d.requires or []:
            if "extra ==" in req:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)", req)
            if m:
                walk(m.group(1))
    for n in names:
        walk(n)
    return seen


def write_lock():
    tree = dist_tree(DIRECT); opt = dist_tree(OPTIONAL)
    direct = {n.lower(): tree[n.lower()] for n in DIRECT}
    deps = {k: v for k, v in tree.items() if k not in direct and v}
    L = ["# requirements-lock.txt -- exact versions installed on the machine that produced the revision results",
         f"# generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} by revision/repro_checks.py (importlib.metadata)",
         f"# Python {platform.python_version()} ({platform.python_implementation()}), {platform.platform()}",
         "# torch is the CPU build: pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu",
         f"# {STDLIB_NOTE}", "", "# direct imports of the project code"]
    for n in DIRECT:
        name, ver = direct[n.lower()]
        L.append(f"{name}=={ver}")
    L += ["", "# installed runtime dependencies of the packages above"]
    for k in sorted(deps):
        name, ver = deps[k]
        L.append(f"{name}=={ver}")
    L += ["", "# optional: test runner for tests/test_agentdojo_alignment.py (the test also runs as a plain script)"]
    for n in OPTIONAL:
        v = opt.get(n.lower())
        L.append(f"# {v[0]}=={v[1]}" if v else f"# {n} (not installed)")
    open(LOCK, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return {"direct": {n: direct[n.lower()][1] for n in DIRECT}, "dependencies": {v[0]: v[1] for v in deps.values()},
            "optional": {n: (opt.get(n.lower()) or (None, None))[1] for n in OPTIONAL}, "python": platform.python_version()}


def run_test():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = {}
    for name, cmd in [("script", [sys.executable, "-B", os.path.join("tests", "test_agentdojo_alignment.py")]),
                      ("pytest", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                  os.path.join("tests", "test_agentdojo_alignment.py")])]:
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=580)
        out[name] = {"command": " ".join(["python" if c == sys.executable else c for c in cmd]), "exit_code": p.returncode,
                     "seconds": round(time.time() - t0, 1), "stdout_tail": p.stdout.strip().splitlines()[-6:],
                     "stderr_tail": p.stderr.strip().splitlines()[-6:]}
        m = re.search(r"positional == id pairing on (\d+) runs", p.stdout)
        if m:
            out[name]["n_runs_with_usable_ids_checked"] = int(m.group(1))
    return out


def scripts():
    L = []
    for d in ["", "paper", "tests", "revision"]:
        base = os.path.join(ROOT, d)
        for f in sorted(os.listdir(base)):
            if f.endswith(".py") or f.endswith(".sh"):
                L.append(os.path.join(d, f).replace("\\", "/"))
    return L


def inspect(rel):
    p = os.path.join(ROOT, rel); txt = open(p, encoding="utf-8", errors="replace").read(); lines = txt.splitlines()
    info = {"sha256": sha(p), "bytes": os.path.getsize(p), "n_lines": len(lines)}
    if rel.endswith(".py"):
        try:
            tree = ast.parse(txt); info["parses"] = True
            info["docstring_head"] = (ast.get_docstring(tree) or "").splitlines()[:12]
        except SyntaxError as e:
            info["parses"] = False; info["syntax_error"] = f"line {e.lineno}: {e.msg}"
    info["env_vars"] = sorted({m.group(2) for m in re.finditer(PATTERNS["env"], txt)} |
                              ({"AGENTDRIFT", "AGENTDOJO"} & set(re.findall(r"export (AGENTDRIFT|AGENTDOJO)=", txt))))
    for k in ["seed", "bootstrap", "estimator", "cli"]:
        rx = re.compile(PATTERNS[k])
        hits = [{"line": i, "text": t.strip()[:220]} for i, t in enumerate(lines, 1) if rx.search(t)]
        info[k + "_lines"] = hits[:60]
        info[k + "_lines_total"] = len(hits)
    return info


def main():
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": os.path.abspath(__file__),
           "env": {"AGENTDRIFT": os.environ.get("AGENTDRIFT"), "AGENTDOJO": os.environ.get("AGENTDOJO")}}
    res["requirements_lock"] = write_lock(); res["requirements_lock_file"] = LOCK
    res["alignment_test"] = run_test()
    inv = {s: inspect(s) for s in scripts()}
    res["scripts"] = inv
    res["parse_failures"] = [s for s, v in inv.items() if v.get("parses") is False]
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", LOCK); print("wrote", OUT)
    print("alignment test:", {k: v["exit_code"] for k, v in res["alignment_test"].items()}, "parse failures:", res["parse_failures"])


if __name__ == "__main__":
    main()
