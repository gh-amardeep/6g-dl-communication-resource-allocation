"""Execute notebooks top-to-bottom in the `exercise1` kernel (in place, outputs saved).

    python experiments/run_notebooks.py            # all
    python experiments/run_notebooks.py 00 03 07   # selected prefixes
"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NB = Path(__file__).resolve().parents[1] / "notebooks"
want = sys.argv[1:]
for f in sorted(NB.glob("*.ipynb")):
    if want and not any(f.name.startswith(w) for w in want):
        continue
    nb = nbformat.read(f, as_version=4)
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=3600, kernel_name="exercise1", resources={"metadata": {"path": str(NB)}}).execute()
        status = "OK"
    except Exception as e:                      # keep partial outputs for debugging
        status = f"FAILED: {str(e)[-1500:]}"
    nbformat.write(nb, f)
    print(f"{f.name:45s} {status}  ({time.time()-t0:.0f}s)", flush=True)
