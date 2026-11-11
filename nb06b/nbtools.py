"""Parse the section files (percent format), fill {{key:fmt}} placeholders from facts, build the notebook, run cells for development."""
import json, re, sys
from pathlib import Path

SECTIONS = Path(__file__).parent / "sections"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*(?::\s*([A-Za-z0-9]+))?\s*\}\}")


def parse_cells():
    cells = []
    for path in sorted(SECTIONS.glob("*.py")):
        kind, buffer = None, []
        def flush():
            if kind is None:
                return
            text = "\n".join(buffer).strip("\n")
            if kind == "md":
                lines = text.split("\n")
                if lines and lines[0].strip() == '"""':
                    lines = lines[1:]
                if lines and lines[-1].strip() == '"""':
                    lines = lines[:-1]
                text = "\n".join(lines).strip("\n")
            cells.append({"kind": kind, "source": text, "file": path.name})
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.startswith("# %% [md]"):
                flush(); kind, buffer = "md", []
            elif line.startswith("# %%"):
                flush(); kind, buffer = "code", []
            else:
                buffer.append(line)
        flush()
    return cells


def fmt_value(value, spec):
    if spec in (None, ""):
        if isinstance(value, bool): return str(value)
        if isinstance(value, int): return f"{value:,}"
        if isinstance(value, float): return f"{value:.2f}"
        return str(value)
    if spec == "n": return f"{int(round(value)):,}"
    if spec == "s": return str(value)
    if spec.startswith("pct"): return f"{100 * value:.{int(spec[3:])}f}%"
    if spec.startswith("sf"): return f"{value:+.{int(spec[2:])}f}"
    if spec.startswith("f"): return f"{value:.{int(spec[1:])}f}"
    raise ValueError(f"unknown format {spec}")


def fill(text, facts, missing):
    def repl(match):
        key, spec = match.group(1), match.group(2)
        if facts is None:
            return match.group(0)
        if key not in facts:
            missing.add(key)
            return match.group(0)
        return fmt_value(facts[key], spec)
    return PLACEHOLDER.sub(repl, text)


def build(out_path, facts_path=None, allow_markers=False):
    import nbformat
    facts = json.loads(Path(facts_path).read_text(encoding="utf-8")) if facts_path else None
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    missing, cells = set(), []
    for cell in parse_cells():
        if cell["kind"] == "md":
            cells.append(nbformat.v4.new_markdown_cell(fill(cell["source"], facts, missing)))
        else:
            cells.append(nbformat.v4.new_code_cell(cell["source"]))
    for i, c in enumerate(cells):
        c["id"] = f"c{i:03d}"
    nb.cells = cells
    if missing:
        raise SystemExit(f"MISSING FACTS ({len(missing)}): {sorted(missing)}")
    markers = [c for c in cells if "[[" in c.source and c.cell_type == "markdown"]
    if markers and not allow_markers:
        raise SystemExit(f"{len(markers)} cell(s) still hold [[MARKER]] text")
    nbformat.write(nb, out_path)
    return len(cells), sum(c.cell_type == "code" for c in cells)


def run_dev(project, dump_facts=None, upto=None):
    """Execute the code cells in one namespace, in the notebooks folder, the way a kernel would."""
    import os, time
    dump_facts = str(Path(dump_facts).resolve()) if dump_facts else None
    os.chdir(Path(project) / "notebooks")
    ns = {"__name__": "__main__"}
    exec("from IPython.display import display", ns)
    t0 = time.time()
    for i, cell in enumerate(c for c in parse_cells() if c["kind"] == "code"):
        t1 = time.time()
        try:
            exec(compile(cell["source"], f"{cell['file']}[code {i}]", "exec"), ns)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"FAILED in {cell['file']} code cell #{i}"); break
        print(f"  [{i:02d}] {cell['file']:28s} {time.time() - t1:5.1f}s", flush=True)
        if upto is not None and i >= upto: break
    print(f"total {time.time() - t0:.1f}s")
    if dump_facts and "FACTS" in ns:
        from pathlib import Path as P
        P(dump_facts).write_text(json.dumps(dict(sorted(ns["FACTS"].items())), indent=1, ensure_ascii=False), encoding="utf-8")
        print("facts:", len(ns["FACTS"]))
    return ns


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        out, facts = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None)
        print(build(out, facts, allow_markers="--allow-markers" in sys.argv))
    elif cmd == "dev":
        run_dev(sys.argv[2], dump_facts=sys.argv[3] if len(sys.argv) > 3 else None)
