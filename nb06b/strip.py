"""Remove outputs and execution counts: python3 strip.py <in.ipynb> <out.ipynb>"""
import sys
import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs, cell.execution_count = [], None
nbformat.write(nb, sys.argv[2])
print("stripped", sum(c.cell_type == "code" for c in nb.cells), "code cells")
