"""
Aggregate results_smacv2.tsv → compute mean ± std per condition
and patch the \newcommand macros in the paper.
"""
import csv, re, pathlib, math

TSV  = pathlib.Path("/Users/tasha/Documents/Github/tribal-village/results_smacv2.tsv")
TEX  = pathlib.Path("/Users/tasha/Documents/Github/698bd9a65fba5c04a962e794/samples/main.tex")

rows = []
with open(TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

def stats(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    if n == 1:
        return mean, float("nan")
    var  = sum((v - mean)**2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)

def fmt(mean, std, n):
    if math.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} \\pm {std:.3f}"

for cond in ["individual", "shared"]:
    cond_rows = [r for r in rows if r["reward_type"] == cond]
    effranks  = [float(r["effrank_n"]) for r in cond_rows]
    probes    = [float(r["probe_acc"])  for r in cond_rows]
    d_acts    = [float(r["d_act"])      for r in cond_rows]
    seeds     = [r["seed"] for r in cond_rows]
    m_er, s_er = stats(effranks)
    m_pr, s_pr = stats(probes)
    m_da, s_da = stats(d_acts)
    n = len(cond_rows)
    print(f"{cond} ({n} seeds {seeds}):")
    print(f"  effrank_n = {fmt(m_er, s_er, n)}")
    print(f"  probe_acc = {fmt(m_pr, s_pr, n)}")
    print(f"  d_act     = {fmt(m_da, s_da, n)}")
    print()

# Patch the paper macros
ind_rows = [r for r in rows if r["reward_type"] == "individual"]
shr_rows = [r for r in rows if r["reward_type"] == "shared"]

def macro_val(rows, key):
    vals = [float(r[key]) for r in rows]
    m, s = stats(vals)
    if math.isnan(s):
        return f"{m:.3f}"
    return f"{m:.3f} \\pm {s:.3f}"

macros = {
    "SMACINDEFFRANK": macro_val(ind_rows, "effrank_n"),
    "SMACINDDAQ":     macro_val(ind_rows, "d_act"),
    "SMACINDPROBE":   macro_val(ind_rows, "probe_acc"),
    "SMACSHREFFRANK": macro_val(shr_rows, "effrank_n"),
    "SMACSHRDAQ":     macro_val(shr_rows, "d_act"),
    "SMACSHRPROBE":   macro_val(shr_rows, "probe_acc"),
}

tex = TEX.read_text()
for name, val in macros.items():
    # Lambda replacement avoids re.sub interpreting backslashes in the string.
    replacement = f"\\newcommand{{\\{name}}}{{{val}}}"
    tex = re.sub(
        rf"\\newcommand{{\\{name}}}{{[^}}]*}}",
        lambda m, r=replacement: r,
        tex,
    )
TEX.write_text(tex)
print("Patched paper macros:")
for k, v in macros.items():
    print(f"  \\{k} = {v}")
