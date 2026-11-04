"""Render the token-highlight figure (Fig. 1) and the heads x tokens figure (Fig. 4) from figdata.json as LaTeX.

usage (from report/): python render_figs.py ../results/figdata.json data
"""
import json
import sys
from pathlib import Path

DIALOGUE = "w_lead2_k4_001"
PANELS = [  # config, title
    ("full_b0", "Full cache"),
    ("kvp_ft_b256", "KVP"),
    ("kvp_ft_offload_b256", "KVP + restore"),
]
MATRICES = [("kvp_ft_b256", "KVP"), ("kvp_ft_heads48_b256", "+ answer tokens"), ("kvp_ft_keyval_heads48_b256", "+ answer + link")]
QUESTION_CUT = "live?"
TEX_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
           "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\~{}", "'": r"\textquotesingle{}"}


def esc(s):
    return "".join(TEX_ESC.get(c, c) for c in s)


def tokens_tex(pieces, counts, n_heads, roles=None):
    out = []
    for i, (p, c) in enumerate(zip(pieces, counts)):
        p = p.replace("\n", " ")
        lead = " " if p.startswith(" ") else ""
        body = esc(p.lstrip(" "))
        role = roles[i] if roles else ""
        if role == "answer":
            body = r"\textbf{" + body + "}"
        elif role == "link":
            body = r"\textit{" + body + "}"
        pct = round(100 * c / n_heads)
        out.append(lead + r"\tk{" + str(pct) + "}{" + body + "}")
    return "".join(out)


def cut_question(q):
    text = ""
    for i, p in enumerate(q["pieces"]):
        text += p
        if text.rstrip().endswith(QUESTION_CUT):
            return q["pieces"][: i + 1], q["counts"][: i + 1]
    raise SystemExit(f"{QUESTION_CUT!r} not found in the question")


def check(row):
    d = row["configs"]
    if not d["full_b0"]["correct"]:
        raise SystemExit(f"{DIALOGUE}: full cache is wrong; pick another dialogue")
    ref = d["full_b0"]["fact"]["pieces"]
    for c, _ in PANELS:
        if d[c]["fact"]["pieces"] != ref:
            raise SystemExit(f"{DIALOGUE}: fact tokens differ for {c}")


def short_answer(a):
    a = a.strip().replace("\n", " ")
    return a if len(a) <= 100 else a[:97] + "..."


def render_fig1(row, n_heads):
    lines = []
    for i, (c, title) in enumerate(PANELS):
        v = row["configs"][c]
        qp, qc = cut_question(v["question"])
        mark = r"\cmark" if v["correct"] else r"\xmark"
        if i:
            lines.append(r"\par\medskip" if i % 3 == 0 else r"\hfill")
        lines.append(r"\tokpanel{" + title + "}{" + mark + "}%")
        lines.append("{" + tokens_tex(v["fact"]["pieces"], v["fact"]["counts"], n_heads, v["fact"]["roles"]) + "}%")
        lines.append("{" + tokens_tex(qp, qc, n_heads) + "}%")
        lines.append("{" + esc(short_answer(v["answer"])) + "}")
    return "\n".join(lines) + "\n"


def role_spans(roles):
    """Contiguous column ranges of the linking phrase and the answer parts: link, then name, then city."""
    spans, i = [], 0
    while i < len(roles):
        if roles[i]:
            j = i
            while j + 1 < len(roles) and roles[j + 1] == roles[i]:
                j += 1
            spans.append((i, j, roles[i]))
            i = j + 1
        else:
            i += 1
    names = {"link": iter(["link"]), "answer": iter(["name", "city"])}
    out = [(a, b, next(names[r])) for a, b, r in spans]
    if [n for _, _, n in out] != ["link", "name", "city"]:
        raise SystemExit(f"unexpected role spans {out}")
    return out


def render_fig4(row):
    out = []
    for c, title in MATRICES:
        v = row["configs"][c]
        f = v["fact"]
        if f["matrix"] is None:
            raise SystemExit(f"{c} has no kept matrix")
        cells = ",".join(f"{t}/{h}" for h, r in enumerate(f["matrix"]) for t, k in enumerate(r) if k)
        labels = ",".join(f"{a}/{b}/{name}" for a, b, name in role_spans(f["roles"]))
        mark = r"\cmark" if v["correct"] else r"\xmark"
        out.append(r"\headmatrix{" + title + "}{" + mark + "}{" + str(len(f["pieces"])) + "}{" + str(len(f["matrix"]))
                   + "}{" + labels + "}{" + cells + "}")
    return "%\n\\hfill\n".join(out) + "%\n"


HEAD_PANELS = [  # config, title, short answer, strings the full answer must contain
    ("kvp_ft_b256", "KVP", "``Bella'', New York", ["Bella", "New York"]),
    ("kvp_ft_heads48_b256", "KVP + answer tokens", "``Bella'', New York", ["Bella", "New York"]),
    ("kvp_ft_keyval_heads48_b256", "KVP + answer + linking phrase", "Lady Waffle, Bergen",
     ["Lady Waffle", "Bergen"]),
]


def words(fact):
    """Merge sub-word pieces into words: (text, token indices, role); punctuation-only pieces are dropped."""
    out = []
    for i, (p, role) in enumerate(zip(fact["pieces"], fact["roles"])):
        if not any(ch.isalnum() for ch in p):
            continue
        if i == 0 or p.startswith(" "):
            out.append([p.strip(), [i], role])
        else:
            out[-1][0] += p
            out[-1][1].append(i)
            out[-1][2] = out[-1][2] or role
    return out


def render_fig2(row):
    cw, rh, gap, lab = 0.09, 0.25, 0.45, 2.0
    ref = words(row["configs"]["full_b0"]["fact"])
    n_w = len(ref)
    key = [j for j, (_, _, role) in enumerate(ref) if role]
    out = [r"\begin{tikzpicture}[every node/.style={text=cink}]"]
    for j, (w, _, role) in enumerate(ref):
        y = -(j + 0.5) * rh
        txt = esc(w)
        txt = r"\textbf{" + txt + "}" if role == "answer" else r"\textit{" + txt + "}" if role == "link" else txt
        out.append(rf"\node[anchor=east, font=\scriptsize, inner sep=1pt] at (-0.08,{y:.3f}) {{{txt}}};")
    yc = -(n_w + 0.6 + 0.5) * rh
    out.append(rf"\node[anchor=east, font=\scriptsize\bfseries, inner sep=1pt, text=crest!70!black] at (-0.08,{yc:.3f}) {{complete fact}};")
    for p, (cfg, title, short, must) in enumerate(HEAD_PANELS):
        v = row["configs"][cfg]
        f = v["fact"]
        for m in must:
            if m not in v["answer"]:
                raise SystemExit(f"{cfg}: answer {v['answer']!r} lacks {m!r}")
        ws = words(f)
        if [w for w, _, _ in ws] != [w for w, _, _ in ref]:
            raise SystemExit(f"{cfg}: words differ from the full-cache run")
        m = f["matrix"]
        n_h = len(m)
        x0 = p * (n_h * cw + gap)
        out.append(rf"\fill[black!6] ({x0:.3f},0) rectangle ({x0 + n_h * cw:.3f},{-n_w * rh:.3f});")
        held = [[all(m[h][t] for t in toks) for h in range(n_h)] for _, toks, _ in ws]
        for j in range(n_w):
            for h in range(n_h):
                if held[j][h]:
                    x, y = x0 + h * cw, -j * rh
                    out.append(rf"\fill[ckvp] ({x + 0.008:.3f},{y - 0.03:.3f}) rectangle ({x + cw - 0.008:.3f},{y - rh + 0.03:.3f});")
        complete = [h for h in range(n_h) if all(held[j][h] for j in key)]
        yt = -(n_w + 0.6) * rh
        out.append(rf"\fill[black!6] ({x0:.3f},{yt:.3f}) rectangle ({x0 + n_h * cw:.3f},{yt - rh:.3f});")
        for h in complete:
            x = x0 + h * cw
            out.append(rf"\fill[crest] ({x + 0.008:.3f},{yt - 0.03:.3f}) rectangle ({x + cw - 0.008:.3f},{yt - rh + 0.03:.3f});")
        mid = x0 + n_h * cw / 2
        out.append(rf"\node[anchor=south, font=\scriptsize\bfseries, align=center, text width={n_h * cw + 0.3:.2f}cm] at ({mid:.3f},0.05) {{{title}}};")
        mark = r"\cmark" if v["correct"] else r"\xmark"
        out.append(rf"\node[anchor=north, font=\scriptsize, align=center, text width={n_h * cw + 0.3:.2f}cm] at ({mid:.3f},{yt - rh - 0.05:.3f}) "
                   rf"{{complete in \textbf{{{len(complete)} of {n_h}}} heads\\ {mark}\ answer: {short}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(src.read_text())
    row = d["dialogues"][DIALOGUE]
    check(row)
    (out / "fig1_panels.tex").write_text(render_fig1(row, d["n_heads"]))
    (out / "fig2_heads.tex").write_text(render_fig2(row))
    print(f"wrote {out}/fig1_panels.tex and {out}/fig2_heads.tex from {DIALOGUE}")


if __name__ == "__main__":
    main()
