#!/usr/bin/env python3
"""Serve a synthetic knowledge-note preview using the production note schema,
checks, deterministic Markdown renderer, and frontmatter builder.

No external APIs, production podcast feeds, or private data are used.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, Response

from podcast_engine.knowledge import frontmatter
from podcast_engine.knowledge.notes_v3 import checks, render, schema


EPISODE = {
    "episode_key": "demo-knowledge-note",
    "podcast": "Example Strength & Nutrition Podcast",
    "podcast_id": "example-strength-nutrition",
    "title": "Protein Timing, Hypertrophy Research & Exercise Selection",
    "published": "Thu, 01 Oct 2026 08:00:00 +0000",
    "link": "https://example.com/podcast/protein-timing-hypertrophy",
    "status": {"compiler": {"state": "completed"}},
}

TRANSCRIPT = (
    "Welcome back to Example Strength & Nutrition Podcast. "
    "Today we're discussing protein distribution, a twelve-week hypertrophy trial, and Romanian deadlift technique. "
    "In the synthetic 2025 trial, forty-two resistance-trained adults were randomized for twelve weeks. "
    "Both groups consumed 1.6 grams per kilogram of protein per day. "
    "One group consumed protein within thirty minutes after training, while the other consumed the same amount later in the day. "
    "The speakers said there was no meaningful difference in muscle gain when total daily protein was matched. "
    "The host described this as a reminder that total daily intake matters more than a narrow post-workout window. "
    "For most lifters, the coach suggested spreading protein across three to five meals rather than chasing a single perfect feeding window. "
    "For Romanian deadlifts, the coach said to keep the bar close to the legs, push the hips back, and stop the descent when hamstring tension is high but before the lower back position changes. "
    "The coach recommended two to four sets of six to ten repetitions for this example, while noting that the exact prescription depends on the lifter and program. "
    "The speaker cited Demo et al. 2025 as the study being discussed."
)

NOTE = {
    "tldr": [
        "When total daily protein was matched, the synthetic trial found no meaningful muscle-gain difference between immediate and delayed post-workout protein timing.",
        "The practical default was to prioritize daily protein intake and distribute it across several meals rather than chase a narrow anabolic window.",
        "For Romanian deadlifts, the coaching emphasis was bar proximity, hip movement, hamstring tension, and stopping before spinal position changes.",
    ],
    "sections": [
        {
            "title": "Protein intake and timing",
            "bottom_line": "Total daily protein was presented as more important than a narrow post-workout timing window.",
            "bullets": [
                {
                    "text": "The synthetic trial included 42 resistance-trained adults, lasted 12 weeks, and matched daily protein at 1.6 g/kg.",
                    "basis": "research",
                    "hedged": False,
                    "anchor": "forty-two resistance-trained adults were randomized for twelve weeks. Both groups consumed 1.6 grams per kilogram of protein per day",
                },
                {
                    "text": "The speakers reported no meaningful difference in muscle gain between immediate and delayed post-workout protein when total daily intake was matched.",
                    "basis": "research",
                    "hedged": False,
                    "anchor": "there was no meaningful difference in muscle gain when total daily protein was matched",
                },
                {
                    "text": "For most lifters, spreading protein across 3-5 meals was offered as a practical default.",
                    "basis": "coaching_experience",
                    "hedged": False,
                    "anchor": "the coach suggested spreading protein across three to five meals",
                },
            ],
            "protocols": [
                {
                    "name": "Practical protein distribution",
                    "what": "Prioritize total daily protein and distribute intake across several meals.",
                    "dose_parameters": "3-5 meals across the day; the synthetic trial used 1.6 g/kg/day total protein.",
                    "when_for_whom": "General example for resistance-trained lifters.",
                    "caveat": "The speakers did not present a single mandatory post-workout feeding window.",
                    "basis": "coaching_experience",
                    "hedged": False,
                    "anchor": "spreading protein across three to five meals rather than chasing a single perfect feeding window",
                }
            ],
        },
        {
            "title": "Romanian deadlift execution",
            "bottom_line": "Use hamstring tension and maintained position to determine the bottom of the repetition.",
            "bullets": [
                {
                    "text": "Keep the bar close to the legs, push the hips back, and stop the descent before lower-back position changes.",
                    "basis": "coaching_experience",
                    "hedged": False,
                    "anchor": "keep the bar close to the legs, push the hips back, and stop the descent when hamstring tension is high but before the lower back position changes",
                }
            ],
            "protocols": [
                {
                    "name": "Romanian deadlift example",
                    "what": "Controlled hip hinge with the bar kept close to the legs.",
                    "dose_parameters": "2-4 sets of 6-10 reps.",
                    "when_for_whom": "Example working-set range, not a universal prescription.",
                    "caveat": "Exact loading and volume depend on the lifter and program.",
                    "basis": "coaching_experience",
                    "hedged": False,
                    "anchor": "recommended two to four sets of six to ten repetitions for this example, while noting that the exact prescription depends on the lifter and program",
                }
            ],
        },
    ],
    "research_discussed": [
        {
            "authors_year": "Demo et al. 2025 (synthetic)",
            "design": "randomized parallel trial",
            "sample": "42 resistance-trained adults",
            "duration": "12 weeks",
            "result": "No meaningful difference in muscle gain when total daily protein was matched between immediate and delayed post-workout timing.",
            "host_comment": "Total daily intake was emphasized over a narrow timing window.",
            "basis": "research",
            "hedged": False,
            "anchor": "synthetic 2025 trial, forty-two resistance-trained adults were randomized for twelve weeks",
        }
    ],
    "numbers": [
        {"topic": "Daily protein in synthetic trial", "value": "1.6 g/kg/day", "basis": "research"},
        {"topic": "Protein distribution example", "value": "3-5 meals/day", "basis": "coaching_experience"},
        {"topic": "Romanian deadlift example", "value": "2-4 sets of 6-10 reps", "basis": "coaching_experience"},
    ],
    "takeaways": [
        "Treat total daily protein as the first-order target before optimizing a narrow post-workout window.",
        "Use exercise technique constraints and individual programming context rather than forcing a fixed range of motion or volume.",
    ],
    "sources_mentioned": [
        {
            "reference_as_heard": "Demo et al. 2025 (synthetic study)",
            "anchor": "The speaker cited Demo et al. 2025 as the study being discussed",
        }
    ],
    "also_discussed": [],
    "follow_up": [],
    "topics": ["protein timing", "hypertrophy", "Romanian deadlift technique"],
    "people": [],
    "existing_tags": ["protein", "hypertrophy", "romanian-deadlift"],
    "new_tag_candidates": [{"tag": "protein-timing", "category": "nutrition"}],
}

METADATA = {
    "topics": NOTE["topics"],
    "people": NOTE["people"],
    "tags": NOTE["existing_tags"] + [candidate["tag"] for candidate in NOTE["new_tag_candidates"]],
}


def build_demo_markdown() -> str:
    errors = schema.validate_note(NOTE)
    if errors:
        raise RuntimeError("Synthetic note violates production schema: " + "; ".join(errors))
    checked, report = checks.apply_checks(NOTE, TRANSCRIPT)
    if report.anchored_dropped:
        raise RuntimeError("Synthetic note unexpectedly lost anchored evidence")
    body = render.render_body(checked)
    return frontmatter.render(EPISODE, METADATA, "2026-10-01T18:00:00Z", body)


def _inline(value: str) -> str:
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    return value


def _body_to_html(body: str) -> str:
    lines = body.splitlines()
    out: list[str] = []
    in_list = False
    in_quote = False
    i = 0

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def close_quote() -> None:
        nonlocal in_quote
        if in_quote:
            out.append("</div>")
            in_quote = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            close_list(); close_quote()
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("> "):
            close_list()
            if not in_quote:
                out.append('<div class="protocol">')
                in_quote = True
            out.append(f"<p>{_inline(line[2:])}</p>")
        elif line.startswith("- "):
            close_quote()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            close_list(); close_quote()
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([cell.strip() for cell in lines[i].strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in headers) + "</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>")
            out.append("</tbody></table>")
            continue
        elif not line.strip():
            close_list(); close_quote()
        elif line.startswith("*") and line.endswith("*"):
            close_list(); close_quote()
            out.append(f'<p class="bottom-line">{_inline(line)}</p>')
        else:
            close_list(); close_quote()
            out.append(f"<p>{_inline(line)}</p>")
        i += 1

    close_list(); close_quote()
    return "\n".join(out)


def create_demo_app() -> Flask:
    app = Flask(__name__)
    markdown = build_demo_markdown()
    _frontmatter, body = markdown.split("---\n", 2)[1:]
    # body starts after the closing frontmatter delimiter.
    body = body.lstrip("\n")
    metadata = frontmatter.build_frontmatter(EPISODE, METADATA, "2026-10-01T18:00:00Z")

    @app.get("/")
    def home():
        chips = "".join(
            f'<span class="chip">{html.escape(tag)}</span>'
            for tag in metadata.get("tags", [])
        )
        page = f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic Knowledge Note</title>
<style>
:root{{color:#172033;background:#f4f6f9;font:16px/1.58 Inter,system-ui,-apple-system,sans-serif}}
body{{margin:0}} .shell{{max-width:980px;margin:auto;padding:2.4rem 1.25rem 5rem}}
.banner{{background:#172033;color:white;border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem}}
.banner b{{display:block;font-size:.95rem}} .banner span{{opacity:.82;font-size:.9rem}}
article{{background:white;border:1px solid #d8dee8;border-radius:14px;padding:2rem 2.2rem;box-shadow:0 2px 10px #1720330c}}
.kicker{{color:#64748b;font-size:.9rem;font-weight:650;letter-spacing:.04em;text-transform:uppercase}}
h1{{font-size:2rem;line-height:1.18;margin:.35rem 0 .6rem}} h2{{font-size:1.35rem;margin:2rem 0 .6rem;padding-top:.25rem;border-top:1px solid #edf0f5}}
.meta{{display:flex;flex-wrap:wrap;gap:.45rem;margin:.8rem 0 1.2rem}} .chip{{background:#eef2f7;border-radius:999px;padding:.22rem .62rem;font-size:.84rem}}
.source-line{{color:#64748b;font-size:.9rem;margin-bottom:1.5rem}} .legend{{background:#f7f9fc;border:1px solid #e6ebf2;border-radius:10px;padding:.8rem 1rem;margin:1rem 0 1.4rem}} .legend-title{{font-weight:700;margin-bottom:.35rem}} .legend-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.25rem 1rem;font-size:.9rem;color:#42516a}} .legend-note{{font-size:.82rem;color:#64748b;margin-top:.45rem}} p{{margin:.45rem 0}} ul{{padding-left:1.4rem}} li{{margin:.45rem 0}}
.bottom-line{{color:#42516a;font-style:italic;background:#f7f9fc;padding:.7rem .85rem;border-radius:8px}}
.protocol{{border-left:4px solid #3b6ea8;background:#f6f9fd;padding:.6rem .9rem;margin:1rem 0;border-radius:0 8px 8px 0}}
.protocol p{{margin:.2rem 0}} table{{width:100%;border-collapse:collapse;margin:1rem 0}} th,td{{text-align:left;padding:.55rem .65rem;border-bottom:1px solid #e4e8ef;vertical-align:top}} th{{font-size:.88rem;color:#536176}}
strong{{color:#0f172a}} code{{background:#eef2f7;padding:.1rem .25rem;border-radius:4px}}
.raw{{display:inline-block;margin-top:1.4rem;color:#315f9d;text-decoration:none}}
@media(max-width:680px){{.shell{{padding:.8rem .55rem 3rem}}article{{padding:1.15rem}}h1{{font-size:1.55rem}}}}
</style>
<main class="shell">
<section class="banner"><b>Synthetic portfolio demo</b><span>Rendered from the production knowledge-note-v3 schema, evidence checks and deterministic Markdown renderer. No real podcast data.</span></section>
<article>
<div class="kicker">{html.escape(metadata["podcast"])}</div>
<h1>{html.escape(metadata["title"])}</h1>
<div class="meta">{chips}</div>
<div class="source-line">Compiled transcript · transcript reviewed · {html.escape(metadata["published"])} · English</div>
<div class="legend">
  <div class="legend-title">Evidence basis</div>
  <div class="legend-grid">
    <span><b>▰▰▰ Study</b> — research cited</span>
    <span><b>▰▰▱ Coaches</b> — multi-client coaching experience</span>
    <span><b>▰▱▱ One person</b> — personal experience</span>
    <span><b>▱▱▱ Opinion</b> — speculation / hypothesis</span>
  </div>
  <div class="legend-note">“· unsure” means the speaker hedged the claim. The meter records stated basis, not independent truth or study-quality verification.</div>
</div>
{_body_to_html(body)}
<a class="raw" href="/raw.md">View exact generated Markdown →</a>
</article>
</main>"""
        return Response(page, mimetype="text/html")

    @app.get("/raw.md")
    def raw_markdown():
        return Response(markdown, mimetype="text/markdown")

    return app


def main() -> None:
    markdown = build_demo_markdown()
    output = ROOT / "examples" / "demo-knowledge-note.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Synthetic knowledge note written to: {output}")
    print("Knowledge-note demo: http://127.0.0.1:8766")
    print("No cloud credentials, model calls, or production podcast data are used.")
    create_demo_app().run(host="127.0.0.1", port=8766, debug=False)


if __name__ == "__main__":
    main()
