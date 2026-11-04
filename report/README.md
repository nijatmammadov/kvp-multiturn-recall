# Technical report (work in progress)

`kvp-multiturn-report.pdf` is built from `main.tex` with pdflatex (TikZ figures, booktabs table).

From this folder:
- `python table1.py .. data/table1.tex` writes Table 1 (every configuration) from the fp32 JSONLs in `../results/`.
- `python render_figs.py ../results/figdata.json data` writes Figure 1 (token view) and Figure 2 (heads x words) from
  the export of `scripts/kaggle_figs.py`.
- `python report_numbers.py .. /tmp/x` prints the paired McNemar tests quoted in the text.
- `latexmk -pdf main.tex`.
