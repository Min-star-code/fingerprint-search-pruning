"""Inline results/benchmark.json into site/template.html -> site/index.html."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "..", "results", "benchmark.json")))
tpl = open(os.path.join(HERE, "template.html")).read()
assert "/*DATA*/null" in tpl
out = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(os.path.join(HERE, "index.html"), "w").write(out)
# GitHub Pages serves /docs
docs = os.path.join(HERE, "..", "docs")
os.makedirs(docs, exist_ok=True)
open(os.path.join(docs, "index.html"), "w").write(out)
open(os.path.join(docs, ".nojekyll"), "w").write("")
print("wrote site/index.html and docs/index.html", len(out) // 1024, "KB")
