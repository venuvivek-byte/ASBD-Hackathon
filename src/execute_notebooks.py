"""
Execute and Populate Jupyter Notebooks
Author: ASBD Hackathon Team
Executes all project Jupyter notebooks sequentially, captures execution counts,
stdout/stderr text, dataframe displays, and embeds base64 figures directly into cell outputs.
"""

import os
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    "01_load_and_inspect.ipynb",
    "02_preprocessing.ipynb",
    "03_descriptive_analytics.ipynb",
    "04_predictive_and_mining.ipynb",
    "05_pyspark_benchmark.ipynb",
    "06_final_report.ipynb",
    "sports_analytics_pipeline.ipynb"
]

def execute_notebook(nb_filename, notebooks_dir="notebooks"):
    nb_path = os.path.join(notebooks_dir, nb_filename)
    print("=" * 80)
    print(f"EXECUTING NOTEBOOK: {nb_path}")
    print("=" * 80)

    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook file not found: {nb_path}")

    # Read original notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Initialize preprocessor with ample timeout
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    start_time = time.time()
    try:
        # Execute cells using the notebooks directory as CWD so relative paths resolve cleanly
        abs_nb_dir = os.path.abspath(notebooks_dir)
        ep.preprocess(nb, {'metadata': {'path': abs_nb_dir}})
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Executed in {elapsed:.2f} seconds.")
    except Exception as e:
        print(f"[ERROR] Execution failed for {nb_filename}: {e}")
        raise e

    # Count populated cells and output types
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    total_outputs = sum(len(c.get('outputs', [])) for c in code_cells)
    image_outputs = 0
    for c in code_cells:
        for out in c.get('outputs', []):
            if 'data' in out and 'image/png' in out['data']:
                image_outputs += 1

    print(f"  * Code Cells Executed: {len(code_cells)}")
    print(f"  * Total Cell Outputs:  {total_outputs}")
    print(f"  * Embedded Images:     {image_outputs}")

    # Verify execution counts
    for idx, c in enumerate(code_cells):
        assert c.execution_count is not None, f"Cell {idx} execution_count is null!"

    # Save executed notebook back to disk
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[OK] Saved fully populated notebook: {nb_path}\n")

    return {
        "notebook": nb_filename,
        "code_cells": len(code_cells),
        "total_outputs": total_outputs,
        "image_outputs": image_outputs,
        "elapsed_sec": round(elapsed, 2)
    }

def main():
    print("STARTING BATCH EXECUTION OF ALL JUPYTER NOTEBOOKS...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    notebooks_dir = os.path.join(project_root, "notebooks")

    results = []
    for nb_name in NOTEBOOKS:
        res = execute_notebook(nb_name, notebooks_dir=notebooks_dir)
        results.append(res)

    print("=" * 80)
    print("ALL NOTEBOOKS EXECUTED AND FULLY POPULATED!")
    print("=" * 80)
    for r in results:
        print(f"  - {r['notebook']}: {r['code_cells']} cells, {r['total_outputs']} outputs, {r['image_outputs']} plots ({r['elapsed_sec']}s)")

if __name__ == "__main__":
    main()
