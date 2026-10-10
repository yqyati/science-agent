import importlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

modules = ["numpy", "scipy", "sympy", "mpmath", "pandas", "networkx", "requests", "httpx", "bs4", "lxml", "pypdf", "fitz", "pdfplumber", "pdfminer", "pyarrow", "openpyxl", "PIL", "matplotlib", "IPython", "PyPDF2", "datasets", "huggingface_hub", "polars", "duckdb", "fastparquet", "rdkit", "z3", "pulp", "ortools", "cvxpy", "pikepdf"]
for name in modules:
    importlib.import_module(name)
print("imports: passed", flush=True)
import sympy as sp
import mpmath as mp
import scipy.integrate
x = sp.Symbol("x")
assert sp.integrate(x**2, (x, 0, 1)) == sp.Rational(1, 3)
mp.mp.dps = 50
assert abs(mp.quad(lambda t: t**2, [0, 1]) - mp.mpf(1)/3) < mp.mpf("1e-45")
assert abs(scipy.integrate.quad(lambda t: t**2, 0, 1)[0] - 1/3) < 1e-12
print("symbolic/high-precision/numerical calculations: passed", flush=True)
import fitz
import pypdf
import pdfplumber
from pdfminer.high_level import extract_text
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/"test.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Science sandbox PDF test 42")
    document.save(path)
    document.close()
    assert "42" in pypdf.PdfReader(path).pages[0].extract_text()
    import PyPDF2
    import pikepdf
    assert "42" in PyPDF2.PdfReader(path).pages[0].extract_text()
    with pikepdf.open(path) as pdf:
        assert len(pdf.pages) == 1
    with pdfplumber.open(path) as pdf:
        assert "42" in pdf.pages[0].extract_text()
    assert "42" in extract_text(str(path))
    assert "42" in subprocess.check_output(["pdftotext", str(path), "-"], text=True)
    import pandas as pd
    table = pd.DataFrame({"answer": [42]})
    parquet = Path(directory)/"test.parquet"
    table.to_parquet(parquet)
    assert pd.read_parquet(parquet).iloc[0,0] == 42
    table.to_parquet(parquet, engine="fastparquet")
    assert pd.read_parquet(parquet, engine="fastparquet").iloc[0,0] == 42
print("PDF extraction and parquet round trip: passed", flush=True)
subprocess.run(["python3", "-m", "pip", "check"], check=True)

from datasets import Dataset
assert Dataset.from_dict({"answer": [42]})[0]["answer"] == 42
import polars as pl
assert pl.DataFrame({"answer": [42]}).select(pl.col("answer").sum()).item() == 42
import duckdb
assert duckdb.sql("select 6*7").fetchone()[0] == 42
from rdkit import Chem
assert Chem.MolFromSmiles("CCO").GetNumAtoms() == 3
from z3 import Int, Solver, sat
answer = Int("answer")
solver = Solver()
solver.add(answer * 2 == 84)
assert solver.check() == sat and solver.model()[answer].as_long() == 42
import pulp
problem = pulp.LpProblem("test", pulp.LpMinimize)
x = pulp.LpVariable("x", lowBound=0)
problem += x
problem += x >= 42
assert problem.solve(pulp.PULP_CBC_CMD(msg=False)) == 1 and x.value() == 42
from ortools.sat.python import cp_model
model = cp_model.CpModel()
x = model.new_int_var(0, 100, "x")
model.add(x == 42)
solver = cp_model.CpSolver()
assert solver.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
import cvxpy as cp
x = cp.Variable()
problem = cp.Problem(cp.Minimize(x), [x >= 42])
problem.solve()
assert abs(x.value - 42) < 1e-5
assert "GLOP" in cp.installed_solvers() and "PDLP" in cp.installed_solvers()
problem.solve(solver="GLOP")
assert abs(x.value - 42) < 1e-5
problem.solve(solver="PDLP")
assert abs(x.value - 42) < 1e-5
print("datasets/dataframes/SQL/RDKit/Z3/CBC/OR-Tools/CVXPY: passed", flush=True)
