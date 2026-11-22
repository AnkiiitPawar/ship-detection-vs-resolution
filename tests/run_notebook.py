"""Run the notebook top to bottom on the fake data, with settings small enough for a CPU.

The notebook file itself is not changed: the smaller settings are applied to an in-memory copy.

Usage:
    pip install -r tests/requirements.txt
    python tests/make_fake_data.py /tmp/fake_input
    python tests/run_notebook.py /tmp/fake_input /tmp/nb_work

Takes about 5-10 minutes on a laptop CPU. The executed copy is saved as <work>/executed.ipynb.
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

repo = Path(__file__).resolve().parent.parent
nb = nbformat.read(repo / "ship_detection_vs_resolution.ipynb", as_version=4)


def edit(find, replace):
    """Replace text in every code cell that contains it; fail loudly if no cell does."""
    cells = [c for c in nb.cells if c.cell_type == "code" and find in c.source]
    assert cells, f"not found in the notebook: {find!r}"
    for c in cells:
        c.source = c.source.replace(find, replace)


edit("N_SHIP_TILES = 4000", "N_SHIP_TILES = 40")   # 40 fake tiles with ships instead of 4,000 real ones
edit("EPOCHS = 15", "EPOCHS = 2")                  # 2 passes are enough to check that everything runs
edit("batch=16", "batch=4")                        # a CPU with a few GB of memory needs smaller batches
for c in nb.cells:                                 # libraries come from tests/requirements.txt instead
    if c.cell_type == "code":
        c.source = "\n".join("# " + line if line.startswith("!pip") else line for line in c.source.splitlines())

input_dir, work_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work_dir.mkdir(parents=True, exist_ok=True)
os.environ["AIRBUS_INPUT"] = str(input_dir)
os.environ["NB_WORK"] = str(work_dir)

t0 = time.time()
client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(work_dir)}})
try:
    client.execute()
    status = "OK"
except Exception as e:  # say which cell failed and why
    status = f"FAILED: {type(e).__name__}: {str(e)[-3000:]}"
nbformat.write(nb, work_dir / "executed.ipynb")

for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    texts = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            texts.append(o.get("text", ""))
        elif o.get("output_type") == "error":
            texts.append("ERROR " + o.get("ename", "") + ": " + o.get("evalue", ""))
        elif o.get("output_type") in ("display_data", "execute_result"):
            texts.append("[" + ", ".join(o.get("data", {}).keys()) + "]")
    print(f"--- cell {i} ---")
    print("".join(texts)[-1500:])
print(status)
print(f"Elapsed: {time.time() - t0:.0f} s")
sys.exit(0 if status == "OK" else 1)
