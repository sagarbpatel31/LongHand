# Longhand — demo & recording guide

Everything you need to run the demo and record it. Two modes:

- **replay** — a synthetic session through the real pipeline. No mic, no API, no
  credits. Deterministic. This is your safe demo and the only place the **seam
  inspector** is guaranteed to appear.
- **live** — your microphone through the real pipeline and the real API. Spends
  a small amount of credit per pause. This is the "wow" beat.

---

## 1. One-time setup

1. Grant microphone permission (macOS): **System Settings → Privacy & Security →
   Microphone** → enable your terminal (Terminal / iTerm) **or** your browser.
2. Start the server from the project root:
   ```bash
   cd /Users/sagarpatel/Desktop/LongHand
   .venv/bin/python -m longhand
   ```
   It prints `Longhand UI → http://127.0.0.1:8000`. Leave it running.
3. Open **http://127.0.0.1:8000** in Chrome / Safari / Firefox.

The page opens in **replay** mode automatically.

---

## 2. Replay mode (warm up — zero risk)

1. Watch the document build: segments **shimmer** while in flight, then land.
2. Tick **show verbatim (raw ASR)** — rare terms flip between the drifted form
   (`koobernetes`) and the cleaned form (`kubernetes`). That is the
   verbatim/clean toggle and the terminology-consistency feature.
3. Point at the **stat chips**: sections, paragraphs, forced cuts, **term.
   consistency 100%**, **WER**.
4. Point at the **seam inspector** box — the one forced cut, showing the
   overlapping words that were de-duplicated (`raw:` vs `kept:`).

If the mic misbehaves during recording, this mode alone is a complete demo.

---

## 3. Live mode (real mic + real API)

1. Click **🎤 live**. First click downloads the Silero VAD model (~2–3s), then
   shows `listening…`.
2. **Speak in ~6–10 second chunks, then pause.** Each pause closes a segment and
   fires one API call.
3. Click **■ stop** when done. No more calls after that.

### How pauses become structure (say it, then wait)

The cut policy uses silence length. Practical cheat sheet:

| Silence you leave | What happens |
|---|---|
| < 0.6 s | keeps going, same segment |
| ~0.6–1.2 s | segment cut, **same paragraph** (sentence break) |
| ~1.2–3.0 s | **new paragraph** |
| ≥ 3.0 s | **new section** (a horizontal rule appears) |

Also: a segment needs **at least ~6 s of speech** before a pause will cut it — so
don't speak in tiny fragments; give each chunk a full sentence or two.

> The **seam inspector** only triggers on a *forced* cut, and live mode forces a
> cut only after **108 s of non-stop talking** — impractical to perform. Show the
> seam in **replay** mode instead (it's built into the demo session).

---

## 4. What to say

Pick one script. Read at a normal pace. `[pause]` = ~1 s, `[PAUSE]` = ~2 s (new
paragraph), `[LONG PAUSE]` = ~3 s+ (new section). Repeat the **bold** terms —
that is what makes terminology carryover visible.

### Script A — Kubernetes control plane (~45 s)

> **Kubernetes** reconciles the desired state of the cluster through a continuous
> control loop, comparing what is declared against what is actually running.
> `[pause]`
> The **operator** watches its resources and drives them toward the manifest we
> committed, retrying until they converge.
> `[PAUSE]`
> Meanwhile the **scheduler** places pods onto nodes that still have spare
> capacity, respecting affinity rules and resource requests.
> `[pause]`
> And the **ingress** routes external traffic to the backing services, so the
> **scheduler** and the **ingress** never have to know about each other.
> `[LONG PAUSE]`
> When a node fails, **Kubernetes** notices the missing heartbeat, marks the pods
> as lost, and the **operator** reschedules them somewhere healthy.

Repeated terms to carry: **Kubernetes**, **operator**, **scheduler**, **ingress**.

### Script B — Distributed consensus / Raft (~45 s)

> **Raft** keeps a replicated log consistent across every node in the **cluster**,
> so that each machine applies the same commands in the same order.
> `[pause]`
> The **leader** appends new entries and the followers acknowledge them once the
> entries are safely on disk.
> `[PAUSE]`
> When the **leader** stops sending heartbeats, an election begins, and a
> candidate requests votes for a new **term**.
> `[pause]`
> **Raft** guarantees that a committed entry is never lost as long as a majority
> of the **cluster** stays alive, which is why the **term** number keeps
> monotonically increasing.
> `[LONG PAUSE]`
> In practice most of the difficulty is not the happy path but the corner cases
> around a **leader** crashing mid-replication.

Repeated terms to carry: **Raft**, **cluster**, **leader**, **term**.

### Script C — Transformers / attention (~45 s)

> A **transformer** processes every token in parallel, using **attention** to let
> each position look at every other position in the sequence.
> `[pause]`
> **Attention** computes a weighted sum of value vectors, where the weights come
> from the similarity between queries and keys.
> `[PAUSE]`
> Stacking many layers lets the **transformer** build increasingly abstract
> representations, from surface syntax up to something like meaning.
> `[pause]`
> Because **attention** is quadratic in the sequence length, most of the
> engineering effort goes into making the **transformer** cheaper on long inputs.
> `[LONG PAUSE]`
> The same architecture now underpins language, vision, and audio models, which is
> why **attention** shows up almost everywhere.

Repeated terms to carry: **transformer**, **attention**.

### Full 90-second run (hits every beat in order)

1. **Open (structure + liveness):** read the first two sentences of any script,
   pausing ~1 s between them, then a ~3 s **LONG PAUSE** so a new section opens.
2. **Terminology carryover:** read the middle, repeating the bold term 3–4 times
   across different segments. Toggle **show verbatim** afterward to show that the
   raw transcripts were normalised to one spelling.
3. **Stats:** stop, then point at **term. consistency** and the section/paragraph
   counts.
4. **Seam inspector:** switch to **replay** mode and point at the forced-cut box.
5. **Benchmark:** cut to a terminal and run `python -m longhand.bench` — the A/B/C
   chart (WER, terminology consistency, paragraph F1) is the headline.

---

## 5. Record it

- macOS screen capture: **⌘⇧5** (or QuickTime → New Screen Recording). Record
  system audio if you want your voice in the clip.
- Suggested ~60–90 s cut:
  1. Replay building + verbatim toggle + stat chips + seam inspector (~30 s).
  2. `python -m longhand.bench` output — the A/B/C chart (~15 s).
  3. Live mic: speak → document builds → stop (~30 s).
- Drop a screenshot or GIF into the README (there's a License/results slot ready).

---

## Gotchas

- Live fails → a red `⚠` line on the page tells you why (no key, no mic, no
  model). Replay always works.
- Speak in full sentences; chunks under ~6 s won't cut into their own segment.
- Credits are small — keep live takes short.
- The verbatim/clean difference is dramatic in **replay** (seeded drift). Live
  ASR is usually already consistent, so carryover mostly matters over long, noisy
  sessions — mention that rather than expecting wild live corrections.
