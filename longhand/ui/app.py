"""FastAPI app — one HTML page over a WebSocket (§2, hours 10-12).

`GET /` serves a single self-contained page. `WS /ws` replays a demo session
through `ReplaySession` and streams init/pending/document messages; the page
renders the document live, shimmers in-flight segments, exposes a verbatim/clean
toggle, and shows a forced-cut seam inspector.

No network, no mic — the demo fixture is synthetic and deterministic. Spec §10:
"the chart wins, not the CSS." The page is deliberately lean.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..bench.fixtures import (
    _DEMO_DRIFT_TERMS,
    _VOCAB,
    _demo_clips,
    build_drift_fixture,
)
from .session import ReplaySession

# A lower ceiling than the real 108s so a ~46s run-on clip trips ONE forced cut
# (for the seam inspector) while the ~13s structured clips still cut on silence.
DEMO_MAX_SEGMENT_S = 30.0
DEFAULT_LATENCY_S = 0.18  # per-segment pause so the shimmer is visible in the demo
_RUNON_AT = 12  # insert the run-on monologue after this many structured clips


def build_demo_drift():
    """Structured drift session + one run-on monologue that forces a cut (§6)."""
    base = _demo_clips(24, 34)  # ~13s clips, drift terms, varied pauses
    runon = " ".join(_VOCAB[i % len(_VOCAB)] for i in range(120))  # ~46s, no pauses
    clips = base[:_RUNON_AT] + [runon] + base[_RUNON_AT:]

    gaps: list[float] = []
    for i in range(len(clips)):
        if i in (_RUNON_AT - 1, _RUNON_AT):  # section breaks around the monologue
            gaps.append(4.0)
        else:
            gaps.append([0.8, 1.5, 4.0][i % 3])
    return build_drift_fixture(clips, gaps=gaps, drift_terms=_DEMO_DRIFT_TERMS)


def create_app() -> FastAPI:
    app = FastAPI(title="Longhand")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _PAGE

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            latency = float(websocket.query_params.get("latency", DEFAULT_LATENCY_S))
        except (TypeError, ValueError):
            latency = DEFAULT_LATENCY_S

        session = ReplaySession(
            build_demo_drift(),
            max_segment_s=DEMO_MAX_SEGMENT_S,
            latency_s=latency,
            sleep=asyncio.sleep,
        )
        try:
            async for msg in session.stream():
                await websocket.send_json(msg)
        except WebSocketDisconnect:
            return

    return app


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Longhand — live document</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --ink:#e7e9ee; --dim:#8b93a3;
          --accent:#6ea8fe; --warn:#ffcf6e; --gap:#ff8b8b; --seam:#c79bff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:16px 24px; border-bottom:1px solid #262a33;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; letter-spacing:.3px; }
  h1 span { color:var(--accent); }
  .stats { display:flex; gap:18px; margin-left:auto; flex-wrap:wrap; }
  .stat { text-align:right; }
  .stat b { display:block; font-size:16px; }
  .stat small { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
  .toggle { display:flex; align-items:center; gap:8px; color:var(--dim); font-size:13px; }
  main { max-width:820px; margin:0 auto; padding:28px 24px 80px; }
  .section { padding:6px 0; }
  .section + .section { border-top:1px dashed #2c313c; margin-top:22px; padding-top:22px; }
  p.para { margin:0 0 14px; }
  .blk { transition:background .3s; }
  .blk.forced { text-decoration:underline dotted var(--seam); }
  .gap { color:var(--gap); font-style:italic; }
  .shimmer { display:inline-block; height:1em; width:200px; border-radius:4px; vertical-align:middle;
             background:linear-gradient(90deg,#20242e 25%,#2c323f 50%,#20242e 75%);
             background-size:400px 100%; animation:sh 1.1s linear infinite; }
  @keyframes sh { from{background-position:-200px 0} to{background-position:200px 0} }
  .pending-line { color:var(--dim); font-size:13px; margin-top:10px; }
  .seam { margin:8px 0; padding:8px 10px; border-left:3px solid var(--seam);
          background:var(--panel); border-radius:4px; font-size:13px; }
  .seam b { color:var(--seam); }
  .seam .row { color:var(--dim); margin-top:4px; white-space:pre-wrap; }
  .seam .kept { color:var(--ink); }
  .drift-note { color:var(--dim); font-size:12px; margin-top:4px; }
  code { background:#0b0d11; padding:1px 5px; border-radius:4px; color:var(--warn); }
</style>
</head>
<body>
<header>
  <h1>Long<span>hand</span> · live document</h1>
  <label class="toggle"><input type="checkbox" id="verbatim"/> show verbatim (raw ASR)</label>
  <div class="stats" id="stats"></div>
</header>
<main>
  <div id="doc"></div>
  <div class="pending-line" id="pending"></div>
</main>
<script>
const docEl = document.getElementById('doc');
const pendEl = document.getElementById('pending');
const statsEl = document.getElementById('stats');
const verbatimEl = document.getElementById('verbatim');
let lastDoc = null, showVerbatim = false;

verbatimEl.addEventListener('change', () => { showVerbatim = verbatimEl.checked; render(); });

function stat(label, val) {
  const d = document.createElement('div'); d.className = 'stat';
  d.innerHTML = `<b>${val}</b><small>${label}</small>`; return d;
}
function renderStats(s) {
  statsEl.innerHTML = '';
  statsEl.append(stat('segments', `${s.segments_done}/${s.segments_total}`));
  statsEl.append(stat('sections', s.sections));
  statsEl.append(stat('paragraphs', s.paragraphs));
  statsEl.append(stat('forced cuts', s.forced_cuts));
  if (s.gaps) statsEl.append(stat('gaps', s.gaps));
  if (s.terminology_consistency !== null)
    statsEl.append(stat('term. consistency', (s.terminology_consistency*100).toFixed(0)+'%'));
  if (s.wer !== null) statsEl.append(stat('WER', (s.wer*100).toFixed(1)+'%'));
}
function seamNode(seam) {
  const d = document.createElement('div'); d.className = 'seam';
  if (seam.aligned) {
    d.innerHTML = `<b>⟡ forced-cut seam</b> — overlap of ${seam.overlap_len} words `
      + `detected and de-duplicated.<div class="row">raw: ${seam.raw_concat}</div>`
      + `<div class="row kept">kept: ${seam.spliced}</div>`;
  } else {
    d.innerHTML = `<b>⚠ forced-cut seam</b> — no confident overlap; marked, not silently merged.`
      + `<div class="row kept">${seam.spliced}</div>`;
  }
  return d;
}
function render() {
  if (!lastDoc) return;
  docEl.innerHTML = '';
  for (const section of lastDoc.sections) {
    const sec = document.createElement('div'); sec.className = 'section';
    for (const para of section.paragraphs) {
      const p = document.createElement('p'); p.className = 'para';
      for (const b of para.blocks) {
        if (b.seam) sec.appendChild(seamNode(b.seam));
        const span = document.createElement('span');
        span.className = 'blk' + (b.forced ? ' forced' : '') + (b.is_gap ? ' gap' : '');
        if (b.is_gap) {
          span.textContent = '[audio unavailable' + (b.error ? ' — ' + b.error : '') + '] ';
        } else {
          const text = showVerbatim ? b.verbatim : b.clean;
          span.textContent = text + ' ';
          if (showVerbatim && b.verbatim !== b.clean) span.title = 'cleaned: ' + b.clean;
        }
        p.appendChild(span);
      }
      sec.appendChild(p);
    }
    docEl.appendChild(sec);
  }
}

const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
const sock = new WebSocket(url);
sock.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.type === 'init') {
    const terms = m.drift_terms.map(t => `<code>${t.drift}</code>→${t.canonical}`).join(', ');
    pendEl.innerHTML = `session: ${m.total_s.toFixed(0)}s, ${m.n_segments} segments · `
      + `drift terms carried across segments: ${terms}`;
  } else if (m.type === 'pending') {
    pendEl.innerHTML = `transcribing segment ${m.seq} `
      + `(${m.duration_s.toFixed(1)}s${m.forced ? ', <b style="color:var(--seam)">forced cut</b>' : ''}) `
      + `&nbsp;<span class="shimmer"></span>`;
  } else if (m.type === 'document') {
    lastDoc = m; renderStats(m.stats); render();
    if (m.final) pendEl.textContent = '✓ document complete — every pause became structure, every term stayed consistent.';
  }
};
sock.onclose = () => { if (lastDoc && !lastDoc.final) pendEl.textContent = 'connection closed.'; };
</script>
</body>
</html>
"""
