"""Build the written report (PDF) from results/benchmark.json.

Run:  .venv/bin/python report/build_report.py
Output: report/Fingerprint_Search_Report.pdf
"""

import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "results", "benchmark.json")))
OUT = os.path.join(HERE, "Fingerprint_Search_Report.pdf")
SITE_URL = os.environ.get("SITE_URL", "")  # filled in once the page is published

# Palette taken from neoplasm.ai (see site/template.html)
INK, MUTED, LINE, BAND = "#17161c", "#62616b", "#e2e2e7", "#f7f7f9"
CORAL, LAV, PURPLE, GRAY, SKIP = "#f2665c", "#a879f2", "#7917ef", "#9a99a3", "#dcdce2"

R = D["rows"]
row = {round(r["threshold"], 2): r for r in R}
TYP = 0.6
L = D["layout"]
H = D["popcount"]["histogram"]
N = D["dataset"]["database_size"]
qs = sorted(D["popcount"]["query_counts"])
QMED = qs[len(qs) // 2]


def fx(x):
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


def pct(x):
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------- figures ---
plt.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 9, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": .6, "axes.axisbelow": True,
})
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)


def window(a, t):
    import math
    return math.ceil(t * a - 1e-9), math.floor(a / t + 1e-9)


def fig_hist():
    lo, hi = window(QMED, TYP)
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    xs = list(range(len(H)))
    ax.bar(xs, H, width=.9, color=[PURPLE if lo <= b <= hi else SKIP for b in xs], linewidth=0)
    for v in (lo - .5, hi + .5):
        ax.axvline(v, color=PURPLE, ls=(0, (3, 3)), lw=1)
    ax.plot([QMED, QMED], [0, max(H) * 1.06], color=INK, lw=1.2)
    ax.set_ylim(0, max(H) * 1.16)
    ax.text(QMED, max(H) * 1.13, f"query · {QMED} bits", fontsize=8, color=INK, va="top", ha="center", weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("bits set in fingerprint")
    ax.set_ylabel("molecules")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-1, len(H))
    fig.tight_layout()
    p = os.path.join(FIG, "fig1_hist.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    kept = sum(c for b, c in enumerate(H) if lo <= b <= hi)
    return p, lo, hi, kept / N


def fig_perf():
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ts = [r["threshold"] for r in R]
    spec = [("rdkit_brute", "baseline (RDKit)", GRAY, "-"), ("rdkit_pruned", "pruning", CORAL, "--"),
            ("numpy_brute", "packed bits", LAV, "-"), ("numpy_pruned", "packed bits + pruning", PURPLE, "--")]
    for k, name, c, ls in spec:
        ax.plot(ts, [r[k + "_ms"] for r in R], ls=ls, color=c, lw=2, marker="o", ms=3, label=name)
    ax.set_xlabel("similarity threshold t")
    ax.set_ylabel("ms per query")
    ax.set_ylim(0, None)
    ax.set_xticks(ts)
    ax.set_xticklabels([f"{t:.2f}" for t in ts], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.16))
    fig.tight_layout()
    p = os.path.join(FIG, "fig2_perf.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def fig_layout():
    fig, ax = plt.subplots(figsize=(6.8, 1.25))
    ax.barh([1, 0], [L["scattered_ms"], L["contiguous_ms"]], color=[CORAL, PURPLE], height=.55)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["scattered in memory", "contiguous copies"], color=INK)
    for y, v in ((1, L["scattered_ms"]), (0, L["contiguous_ms"])):
        ax.text(v + 1, y, f"{v:.0f} ms", va="center", fontsize=8, color=INK)
    ax.set_xlabel("RDKit ms per query (same call, same data)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, L["scattered_ms"] * 1.12)
    fig.tight_layout()
    p = os.path.join(FIG, "fig3_layout.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


p1, LO, HI, KEPT = fig_hist()
p2 = fig_perf()
p3 = fig_layout()

# ------------------------------------------------------------------- text ---
S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=6),
    "kicker": ParagraphStyle("kicker", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=PURPLE, spaceAfter=4),
    "by": ParagraphStyle("by", fontName="Helvetica", fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=INK, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.6, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=8),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=INK, spaceAfter=3),
    "stat": ParagraphStyle("stat", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=INK, alignment=1),
    "statl": ParagraphStyle("statl", fontName="Helvetica", fontSize=7.8, leading=10, textColor=MUTED, alignment=1),
}
P = lambda t, s="body": Paragraph(t, S[s])  # noqa: E731


def bullets(items, style="body"):
    return ListFlowable([ListItem(P(i, style), leftIndent=10, value="•") for i in items],
                        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8, bulletColor=MUTED)


rt, last = row[TYP], R[-1]
first_useful = next(r for r in R if r["rdkit_pruned_speedup"] >= 1.5)
env, ds = D["environment"], D["dataset"]

stats = Table([[P(fx(rt["rdkit_pruned_speedup"]), "stat"), P(f'<font color="{PURPLE}">{fx(rt["numpy_brute_speedup"])}</font>', "stat"),
                P(f'{fx(last["numpy_pruned_speedup"])}', "stat"), P(str(D["correctness"]["mismatched_result_sets"]), "stat")],
               [P(f"pruning bound at t = {TYP:.2f}", "statl"), P("packed bits, every threshold", "statl"),
                P(f"both combined at t = {last['threshold']:.2f}", "statl"),
                P(f"mismatches in {D['correctness']['checks']:,} checks", "statl")]],
              colWidths=[1.7 * inch] * 4)
stats.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .6, colors.HexColor(LINE)),
                           ("LINEBEFORE", (1, 0), (-1, -1), .6, colors.HexColor(LINE)),
                           ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))

hdr = ["t", "baseline", "pruning", "packed", "packed + pruning", "library scored", "mean hits"]
tbl_rows = [hdr] + [[f'{r["threshold"]:.2f}', f'{r["rdkit_brute_ms"]:.1f}', f'{r["rdkit_pruned_ms"]:.1f}',
                     f'{r["numpy_brute_ms"]:.2f}', f'{r["numpy_pruned_ms"]:.2f}', pct(r["rdkit_pruned_scanned"]),
                     f'{r["mean_hits"]:.1f}'] for r in R]
table = Table(tbl_rows, colWidths=[.55 * inch, .9 * inch, .9 * inch, .9 * inch, 1.2 * inch, 1.1 * inch, .9 * inch], repeatRows=1)
table.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(MUTED)), ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(INK)),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor(LINE)),
    ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor(BAND)),
]))

link = (f'Interactive version: <a href="{SITE_URL}" color="{PURPLE}">{SITE_URL}</a>' if SITE_URL
        else "Interactive version: link to be added.")

story = [
    P("ALGORITHM OPTIMIZATION · CHEMINFORMATICS · RESEARCH NOTE", "kicker"),
    P("When pruning doesn't pay: faster exact similarity search on 250k drug-like molecules", "title"),
    P(f"Mincheng Li · University of Waterloo, Honours Mathematics · November 2026 · {link}", "by"),
    stats,
    Spacer(1, 10),
    P("Summary", "h2"),
    P(f"Exact Tanimoto similarity search was benchmarked on {ds['database_size']:,} drug-like molecules (ZINC) with "
      f"{ds['queries']} held-out queries. The popcount bound of Swamidass &amp; Baldi (2007) is exact, but on this library "
      f"it skips almost nothing at typical thresholds: at t = {TYP:.2f}, {pct(rt['rdkit_pruned_scanned'])} of molecules still "
      f"had to be scored and the speed-up was {fx(rt['rdkit_pruned_speedup'])}. The reason is visible in one histogram: "
      f"drug-like molecules have similar fingerprint sizes, so the bound's window covers most of the library. Changing the "
      f"representation instead, packing each fingerprint into 64-bit words and scoring the whole library in one vectorised "
      f"pass, was {fx(rt['numpy_brute_speedup'])} faster at every threshold. The two ideas stack to "
      f"{fx(last['numpy_pruned_speedup'])} for near-duplicate search (t = {last['threshold']:.2f}). All four methods returned "
      f"identical hits and scores in every one of {D['correctness']['checks']:,} checks."),
    P("Problem", "h2"),
    P("Similarity search asks which molecules in a library resemble a query. Each molecule becomes a 2,048-bit Morgan "
      "fingerprint (radius 2, the ECFP4 equivalent; Rogers &amp; Hahn, 2010), and two molecules are compared with the "
      "Tanimoto score: shared bits divided by total distinct bits (Bajusz et al., 2015 support it as a sound default). "
      "An exact search compares the query with every molecule, so cost grows with library size. The question here is "
      "which published speed-ups still hold on a realistic drug-like library, and by how much, without changing a single result."),
    P("Idea 1: skip molecules that cannot reach the threshold", "h2"),
    P("A fingerprint with <i>a</i> bits can never score above min(<i>a</i>, <i>b</i>) / max(<i>a</i>, <i>b</i>) against one "
      "with <i>b</i> bits. For threshold <i>t</i>, molecules with fewer than <i>t·a</i> or more than <i>a/t</i> bits are skipped. "
      "With the library sorted by bit count, the remaining candidates are one contiguous slice found by two binary searches."),
    KeepTogether([
        Image(p1, width=6.8 * inch, height=2.6 * inch),
        P(f"Figure 1. Fingerprint bit counts for all {N:,} library molecules (median {D['popcount']['db_median']:.0f}). For the "
          f"median query ({QMED} bits) at t = {TYP:.2f}, only molecules with {LO}–{HI} bits can match (purple): "
          f"{pct(KEPT)} of the library still has to be scored. Averaged over all {ds['queries']} queries at this threshold, "
          f"{pct(rt['rdkit_pruned_scanned'])} was scored.", "cap"),
    ]),
    P("Idea 2: make each comparison cheaper", "h2"),
    P(f"Each fingerprint is stored as 32 × 64-bit words in one contiguous {D['timing']['packed_db_megabytes']:.0f} MB "
      f"NumPy array; a query is scored against the whole library with bitwise AND, hardware popcount and one division. "
      f"This follows Dalke (2019), who argues that modern exact search is limited mainly by memory bandwidth rather than "
      f"arithmetic. The pruning slice can be applied to the packed array too."),
    KeepTogether([
        Image(p2, width=6.8 * inch, height=2.9 * inch),
        P(f"Figure 2. Mean time per query ({ds['queries']} queries, best of 3 runs each, one CPU thread). Pruning only beats "
          f"the baseline by 1.5× or more from t = {first_useful['threshold']:.2f}, where fewer than "
          f"{max(first_useful['mean_hits'], 0.1):.1f} library molecules per query pass the threshold on average. Packed bits are flat at about {rt['numpy_brute_ms']:.1f} ms.", "cap"),
    ]),
    P("Side finding: memory layout", "h2"),
    P(f"An early benchmark run reported a baseline of about 90 ms per query. The cause was not the algorithm: RDKit keeps "
      f"each fingerprint as its own object, and after the library was shuffled the list order no longer matched the order in "
      f"which the objects were created. The same call on the same data took {L['scattered_ms']:.0f} ms instead of "
      f"{L['contiguous_ms']:.0f} ms ({fx(L['scattered_ms'] / L['contiguous_ms'])}). All RDKit numbers in this report use fresh "
      f"copies laid out in scan order, so the baseline is shown at its best."),
    KeepTogether([
        Image(p3, width=6.8 * inch, height=1.25 * inch),
        P(f"Figure 3. RDKit BulkTanimotoSimilarity, mean over {L['queries']} queries, best of 3 runs.", "cap"),
    ]),
    P("Results", "h2"),
    table,
    P("Times in ms per query. 'Library scored' is the share of molecules the pruning bound could not skip.", "cap"),
    P("Implications", "h2"),
    bullets([
        "Check a published optimization against the shape of your own data first. Here a bit-count histogram predicted the "
        "pruning result before any timing was run.",
        "For libraries of similar-sized molecules, a better representation is the larger lever; pruning is worth adding for "
        "near-duplicate search, where it multiplies the gain.",
        "Memory layout can change measured speed by several times, so baselines need the same care as the method being tested.",
    ]),
    P("Limits and next steps", "h2"),
    bullets([
        "One library (drug-like ZINC subset) and one fingerprint type. A more diverse library, or a shorter fingerprint such as "
        "MACCS, would spread bit counts more widely and should make the bound more useful; this is the first thing to test.",
        "Single-threaded, one query at a time. Batching many queries per pass over memory and multi-threading are the obvious "
        "next steps for the packed method, given the memory-bandwidth view.",
        "Libraries larger than memory, approximate search, and GPU implementations were out of scope.",
    ]),
    P("Method and reproducibility", "h2"),
    bullets([
        f"Data: {ds['molecules_read']:,} SMILES from the 250k ZINC subset used by Gómez-Bombarelli et al. (2018); "
        f"{ds['heldout_pool']:,} held out, {ds['queries']} used as queries (seed {D['seed']}).",
        "Correctness: every method's hit list and floating-point scores were compared exactly with the RDKit baseline for every "
        "query and threshold; 30 unit tests also compare all four methods with a plain Python loop.",
        f"Environment: {env['cpu']}, Python {env['python']}, RDKit {env['rdkit']}, NumPy {env['numpy']}.",
        "Run: <font face='Courier'>python src/benchmark.py</font> then <font face='Courier'>pytest tests</font>.",
    ], "small"),
    P("How it was built", "h2"),
    P("Prototyped in about a day with a coding agent (Claude Code). I chose the question, the comparisons and what counted as "
      "correct; the agent drafted the code; no number was reported until it matched a deliberately slow reference "
      "implementation. Two misleading baselines (Python dictionary overhead, then the memory-layout effect above) were traced "
      "and fixed during the work."),
    P("References", "h2"),
    bullets([
        "Bajusz, D., Rácz, A. &amp; Héberger, K. (2015). Why is Tanimoto index an appropriate choice for fingerprint-based "
        "similarity calculations? <i>Journal of Cheminformatics</i>, 7, 20.",
        "Dalke, A. (2019). The chemfp project. <i>Journal of Cheminformatics</i>, 11, 76.",
        "Gómez-Bombarelli, R. et al. (2018). Automatic chemical design using a data-driven continuous representation of "
        "molecules. <i>ACS Central Science</i>, 4(2), 268–276.",
        "Rogers, D. &amp; Hahn, M. (2010). Extended-connectivity fingerprints. <i>Journal of Chemical Information and "
        "Modeling</i>, 50(5), 742–754.",
        "Swamidass, S. J. &amp; Baldi, P. (2007). Bounds and algorithms for fast exact searches of chemical fingerprints in "
        "linear and sublinear time. <i>Journal of Chemical Information and Modeling</i>, 47(2), 302–317.",
    ], "small"),
    Spacer(1, 10),
    P("Independent project. Not affiliated with or endorsed by Neoplasm AI.", "cap"),
]


def on_page(c, doc):
    c.saveState()
    c.setFillColor(colors.HexColor(PURPLE))
    c.rect(0, LETTER[1] - 4, LETTER[0], 4, stroke=0, fill=1)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor(MUTED))
    c.drawString(.8 * inch, .5 * inch, "Mincheng Li · Exact similarity search benchmark")
    c.drawRightString(LETTER[0] - .8 * inch, .5 * inch, str(doc.page))
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=LETTER, leftMargin=.8 * inch, rightMargin=.8 * inch,
                        topMargin=.7 * inch, bottomMargin=.75 * inch,
                        title="When pruning doesn't pay", author="Mincheng Li")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
