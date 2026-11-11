"""Execute a notebook in a fresh kernel with nbclient: python3 runnb.py <in.ipynb> <out.ipynb> <working-folder> [kernel]"""
import sys, time
import nbformat
from nbclient import NotebookClient

src, dst, cwd = sys.argv[1:4]
kernel = sys.argv[4] if len(sys.argv) > 4 else "python3"
nb = nbformat.read(src, as_version=4)
start = time.time()
NotebookClient(nb, timeout=3600, kernel_name=kernel, resources={"metadata": {"path": cwd}}).execute()
nbformat.write(nb, dst)
print(f"executed {len(nb.cells)} cells in {time.time() - start:.1f} s with kernel {kernel}")
