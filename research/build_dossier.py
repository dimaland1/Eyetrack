"""Assemble les chapitres du dossier de thèse en un document HTML unifié (prêt PDF).

Vérifie l'existence des chapitres et des figures référencées, génère une table des
matières automatique, formate la bibliographie depuis ``references.bib`` et embarque
MathJax pour le rendu des équations LaTeX. Si Microsoft Edge ou Chrome est présent
sous Windows, propose aussi l'export direct en PDF via ``--pdf``.

Usage :
    python -m research.build_dossier
    python -m research.build_dossier --pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import html as html_lib


ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = ROOT / "docs" / "these"
OUT_HTML = THESIS_DIR / "dossier_these_eyetrack.html"
OUT_PDF = THESIS_DIR / "dossier_these_eyetrack.pdf"


CHAPTER_ORDER = [
    "00_plan_et_resume.md",
    "01_introduction_problematique.md",
    "02_etat_de_l_art.md",
    "03_fondements_mathematiques_et_systeme.md",
    "04_acquisition_solo_et_protocole.md",
    "05_modeles_et_architectures.md",
    "06_experiences_et_ablations.md",
    "07_discussion_limites_et_perspectives.md",
]


def _inline_format(text: str) -> str:
    """Convertit la mise en forme inline Markdown en préservant les blocs LaTeX $...$."""
    math_tokens: list[str] = []

    def _save_math(m: re.Match) -> str:
        math_tokens.append(m.group(0))
        return f"@@MATH{len(math_tokens) - 1}@@"

    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$", _save_math, text, flags=re.DOTALL)
    text = html_lib.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*\n]+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    for i, tok in enumerate(math_tokens):
        text = text.replace(f"@@MATH{i}@@", tok)
    return text


def md_to_html(md: str, chapter_idx: int) -> tuple[str, list[tuple[int, str, str]]]:
    """Convertisseur Markdown -> HTML léger respectant tableaux, code, citations et LaTeX."""
    lines = md.splitlines()
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []

    in_code = False
    in_math = False
    math_buf: list[str] = []
    in_list: str | None = None
    in_quote = False
    table_buf: list[str] = []

    def _flush_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    def _flush_quote():
        nonlocal in_quote
        if in_quote:
            out.append("</blockquote>")
            in_quote = False

    def _flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows = [[c.strip() for c in r.strip("|").split("|")] for r in table_buf]
        out.append("<table>")
        if len(rows) >= 2 and all(set(c) <= set("-: ") for c in rows[1]):
            out.append("<thead><tr>" + "".join(f"<th>{_inline_format(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for r in rows[2:]:
                out.append("<tr>" + "".join(f"<td>{_inline_format(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody>")
        else:
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{_inline_format(c)}</td>" for c in r) + "</tr>")
        out.append("</table>")
        table_buf = []

    for line in lines:
        if line.strip().startswith("```"):
            _flush_list()
            _flush_quote()
            _flush_table()
            if not in_code:
                in_code = True
                out.append("<pre><code>")
            else:
                in_code = False
                out.append("</code></pre>")
            continue
        if in_code:
            out.append(html_lib.escape(line))
            continue

        if line.strip().startswith("$$") and not line.strip().endswith("$$"):
            _flush_list()
            _flush_quote()
            _flush_table()
            in_math = True
            math_buf = [line]
            continue
        if in_math:
            math_buf.append(line)
            if line.strip().endswith("$$"):
                in_math = False
                out.append('<div class="math-block">' + "\n".join(math_buf) + "</div>")
            continue

        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            _flush_list()
            _flush_quote()
            table_buf.append(line.strip())
            continue
        _flush_table()

        m_hdr = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m_hdr:
            _flush_list()
            _flush_quote()
            lvl = len(m_hdr.group(1))
            title = m_hdr.group(2).strip()
            slug = f"c{chapter_idx}-{len(toc)}"
            toc.append((lvl, slug, title))
            out.append(f'<h{lvl} id="{slug}">{_inline_format(title)}</h{lvl}>')
            continue

        if line.strip().startswith(">"):
            _flush_list()
            if not in_quote:
                in_quote = True
                out.append("<blockquote>")
            out.append(f"<p>{_inline_format(line.strip().lstrip('>').strip())}</p>")
            continue
        _flush_quote()

        m_ul = re.match(r"^\s*[-*]\s+(.*)$", line)
        m_ol = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m_ul:
            if in_list != "ul":
                _flush_list()
                in_list = "ul"
                out.append("<ul>")
            out.append(f"<li>{_inline_format(m_ul.group(1))}</li>")
            continue
        if m_ol:
            if in_list != "ol":
                _flush_list()
                in_list = "ol"
                out.append("<ol>")
            out.append(f"<li>{_inline_format(m_ol.group(1))}</li>")
            continue
        _flush_list()

        if not line.strip():
            continue
        if line.strip().startswith("$$") and line.strip().endswith("$$"):
            out.append(f'<div class="math-block">{line.strip()}</div>')
        else:
            out.append(f"<p>{_inline_format(line)}</p>")

    _flush_list()
    _flush_quote()
    _flush_table()
    return "\n".join(out), toc


def format_bib_html(bib_path: Path) -> str:
    if not bib_path.exists():
        return ""
    raw = bib_path.read_text(encoding="utf-8")
    entries = re.split(r"\n(?=@)", raw.strip())
    items = []
    for e in entries:
        key_m = re.match(r"@\w+\{([^,]+),", e)
        key = key_m.group(1) if key_m else "ref"
        fields = dict(re.findall(r"(\w+)\s*=\s*\{([^}]*)\}", e))
        author = fields.get("author", "")
        title = fields.get("title", "")
        venue = fields.get("booktitle") or fields.get("journal") or ""
        year = fields.get("year", "")
        note = fields.get("note", "")
        parts = [p for p in (author, f"« {title} »" if title else "", venue, year, note) if p]
        cleaned = ", ".join(parts).replace(r"\&", "&")
        items.append(f"<li><strong>[{html_lib.escape(key)}]</strong> {html_lib.escape(cleaned)}</li>")
    return '<h1 id="bib">Bibliographie</h1><ul class="bib">' + "\n".join(items) + "</ul>"


def build_html() -> Path:
    bodies = []
    all_toc: list[tuple[int, str, str]] = []
    missing_figs: list[str] = []

    for idx, fname in enumerate(CHAPTER_ORDER):
        fpath = THESIS_DIR / fname
        if not fpath.exists():
            continue
        md = fpath.read_text(encoding="utf-8")
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
            fig_rel = m.group(1)
            if not (THESIS_DIR / fig_rel).exists():
                missing_figs.append(fig_rel)
        html_part, toc = md_to_html(md, idx)
        bodies.append(f'<section class="chapter">{html_part}</section>')
        all_toc.extend(toc)

    if missing_figs:
        print(f"[build_dossier] Attention, figures manquantes : {missing_figs}")

    toc_html = ['<nav class="toc"><h2>Table des matières</h2><ul>']
    for lvl, slug, title in all_toc:
        if lvl <= 2:
            cls = f"toc-l{lvl}"
            toc_html.append(f'<li class="{cls}"><a href="#{slug}">{_inline_format(title)}</a></li>')
    toc_html.append('<li class="toc-l1"><a href="#bib">Bibliographie</a></li></ul></nav>')

    bib_html = format_bib_html(THESIS_DIR / "references.bib")

    doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Dossier Scientifique Eyetrack — Estimation du Regard RGB Temps Réel</title>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$']] }} }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body {{ font-family: "Georgia", "Times New Roman", serif; line-height: 1.58; color: #1b1f23; max-width: 900px; margin: 2.5rem auto; padding: 0 1.5rem; }}
  h1, h2, h3 {{ font-family: "Segoe UI", Arial, sans-serif; color: #0f2c59; margin-top: 1.8rem; }}
  h1 {{ border-bottom: 2px solid #0f2c59; padding-bottom: 0.35rem; page-break-before: always; }}
  .chapter:first-of-type h1 {{ page-break-before: auto; }}
  code {{ font-family: Consolas, monospace; background: #f3f5f8; padding: 0.12rem 0.32rem; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #f6f8fa; padding: 0.9rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 0.88em; }}
  blockquote {{ border-left: 4px solid #2b6cb0; margin: 1rem 0; padding: 0.4rem 1rem; background: #f7fafc; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.65rem; text-align: left; }}
  th {{ background: #eef3f9; font-family: "Segoe UI", Arial, sans-serif; }}
  figure {{ text-align: center; margin: 1.4rem 0; }}
  figure img {{ max-width: 100%; height: auto; border: 1px solid #e1e4e8; border-radius: 4px; }}
  figcaption {{ font-size: 0.88rem; color: #57606a; margin-top: 0.4rem; font-style: italic; }}
  .toc {{ background: #f8fafc; border: 1px solid #dce3ec; padding: 1rem 1.4rem; border-radius: 6px; }}
  .toc ul {{ list-style: none; padding-left: 0; }}
  .toc-l1 {{ font-weight: bold; margin-top: 0.4rem; }}
  .toc-l2 {{ padding-left: 1.2rem; font-size: 0.95rem; }}
  .math-block {{ overflow-x: auto; margin: 0.8rem 0; }}
  .bib li {{ margin-bottom: 0.45rem; font-size: 0.92rem; }}
  @media print {{ body {{ max-width: none; margin: 1.5cm; }} .toc {{ page-break-after: always; }} }}
</style>
</head>
<body>
<header>
  <p style="text-transform:uppercase;letter-spacing:1px;color:#4a5568;font-family:sans-serif;font-size:0.85rem;">Dossier de Recherche Expérimentale &amp; Monographie Technique — Projet Eyetrack</p>
</header>
{''.join(toc_html)}
{''.join(bodies)}
<section class="chapter">{bib_html}</section>
</body>
</html>"""

    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"[build_dossier] HTML généré : {OUT_HTML}")
    return OUT_HTML


def try_export_pdf(html_path: Path, pdf_path: Path) -> bool:
    for exe in ("msedge", "chrome", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        found = shutil.which(exe) or (exe if Path(exe).exists() else None)
        if found:
            cmd = [
                found,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=8000",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and pdf_path.exists():
                print(f"[build_dossier] PDF généré : {pdf_path}")
                return True
    print("[build_dossier] Aucun navigateur headless détecté pour l'export PDF direct (ouvrez le HTML et imprimez en PDF).")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="Tente un export PDF via Edge/Chrome headless")
    args = ap.parse_args()
    html_p = build_html()
    if args.pdf:
        try_export_pdf(html_p, OUT_PDF)


if __name__ == "__main__":
    main()
