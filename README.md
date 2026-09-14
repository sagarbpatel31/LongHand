# Longhand

**Dictate for twenty minutes over an API that cuts you off at two.**

AssemblyAI's [Dictation API](https://www.assemblyai.com/) is fast and accurate,
but it caps every call at **120 seconds**. Longhand lets you speak for as long as
you like and get back one coherent, formatted document — seamlessly stitched,
terminologically consistent, and paragraphed from the rhythm of your own pauses.

It does this by segmenting on **natural silence**, transcribing segments in
parallel, and assembling the pieces with the timing data most pipelines throw
away.

---

## The two ideas it rests on

1. **Cut at silence, not at the cap.** A cut placed inside a pause has no word
   straddling it, so there is nothing to deduplicate at the seam. Naive chopping
   at the 110-second ceiling slices through whatever word happens to be there and
   loses it — at *every* boundary. Overlap-splicing exists only as a fallback for
   the rare forced cut when a speaker never pauses.

2. **Silence structure is document structure.** The pause durations the VAD gives
   us for free are not noise to discard — they *are* the document. A short pause
   is a sentence break; a longer one starts a paragraph; a long one starts a
   section. Longhand never throws timing data away.

---

## Results

A ~10-minute synthetic session with exact ground truth, three conditions, all
offline and deterministic (`python -m longhand.bench`):

- **A** — naive 110s chop (the baseline): cuts land mid-word.
- **B** — segment on silence (VAD cut policy): cuts land in gaps.
- **C** — B + glossary carryover across segments + a consistency pass.

```
Session: 636.6s, 1428 words, 0.45s/word avg

Condition                 segs      WER  boundary loss  forced%
---------------------------------------------------------------
A: naive 110s chop           6     2.7%              5     0.0%
B: VAD, no carryover        42     2.3%              0     0.0%
C: VAD + carryover          42     0.0%              0     0.0%

Terminology consistency (canonical / total rare-term occurrences)
---------------------------------------------------------------
A: naive 110s chop          68.6%
B: VAD, no carryover        68.6%
C: VAD + carryover         100.0%

Paragraph-boundary F1 (structure recovered from pauses)
---------------------------------------------------------------
A: naive 110s chop           0.00
B: VAD, no carryover         1.00
C: VAD + carryover           1.00
```

**Cutting at silence recovers every boundary word** (A loses 5, B/C lose 0).
**Carryover keeps terminology consistent** — a rare term first heard as
`koobernetes` is pinned as a keyterm for later segments and normalised across the
document, lifting consistency from 69% to 100% and driving WER to zero.
**Pauses reconstruct paragraph structure** with perfect F1, where the naive chop
recovers none.

---

## Architecture

```
 mic / audio ──► RingBuffer ──► Silero VAD ──► CutPolicy
 (int16 PCM)     (retains        (speech /      (cut at silence;
                  audio until      silence)       forced cut at 108s
                  confirmed)                      with 2s overlap)
                                                        │
                                                 SegmentClosed(seq, audio,
                                                   t_start, t_end, lead_pause,
                                                   forced)
                                                        │
                                                        ▼
                              SegmentScheduler  ── N parallel in-flight ──►  Dictation API
                              (cap, retries,                                 (≤110s / call)
                               backpressure)    ◄── out-of-order results ───
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                    ▼                   ▼
                            OrderedAssembler        Glossary          (per-segment
                            (seq order;             (keyterms +        config enrich)
                             failures become        stt_prompt fed
                             visible gaps)          into later segments)
                                    │
                                    ▼
                            DocumentBuilder
                            (pause → paragraphs/sections,
                             forced-cut splice, consistency pass)
                                    │
                                    ▼
                            UI  (live document, in-flight shimmer,
                                 seam inspector, verbatim/clean toggle)
```

| Module | Responsibility |
|---|---|
| `capture/mic.py` | sounddevice `RawInputStream` → int16 frames (live only) |
| `capture/ring.py` | rolling buffer; retains audio until a segment is confirmed |
| `segment/vad.py` | Silero VAD → speech/silence decisions (+ `FakeVad` for tests) |
| `segment/policy.py` | cut rules: min/max length, clean cuts, forced cuts |
| `stt/client.py` | async Dictation wrapper — an **interface**, so it can be faked |
| `stt/fake.py` | `FakeDictationClient`: latency, out-of-order, 5xx, timeouts, `llm_error` |
| `stt/scheduler.py` | concurrency cap, retries, backpressure, carryover hooks |
| `assemble/order.py` | out-of-order arrival → ordered document; failures → visible gaps |
| `assemble/splice.py` | overlap alignment + dedup for forced cuts |
| `assemble/glossary.py` | running term registry; consistency normalization |
| `assemble/structure.py` | pause durations → paragraphs and sections |
| `ui/session.py` | `ReplaySession`: drives the whole pipeline, emits UI messages |
| `ui/app.py` | FastAPI + one HTML page over a WebSocket |
| `bench/` | synthetic fixtures, fixture ASR client, naive-vs-Longhand comparison |

---

## Run it

```bash
pip install -e .

pytest                    # full offline suite — never touches the network
python -m longhand.bench  # the A vs B vs C benchmark above
python -m longhand        # the live-document UI at http://127.0.0.1:8000
pytest -m live            # the one real-API smoke test (needs a key; opt-in)
```

The UI replays a synthetic session through the *real* pipeline, so you can watch
the document assemble live — segments shimmer while in flight, pauses fold into
paragraphs and sections, a verbatim/clean toggle reveals the terminology
normalization, and the seam inspector shows the one forced cut being spliced.
No key, no mic, no network required.

### Live dictation from your mic

Open the UI and click **🎤 live**. Real audio flows through the genuine
concurrent pipeline — Silero VAD segments on your pauses, segments transcribe in
parallel against the real API, and the document assembles in front of you.
Click **■ stop** when you're done. This path needs three things and degrades to a
single visible error message if any is missing:

- `ASSEMBLYAI_API_KEY` in `.env` (never commit it):

  ```
  ASSEMBLYAI_API_KEY=your_raw_key_here
  ```

- a working microphone (PortAudio / `sounddevice`);
- the Silero VAD model (`onnxruntime`, fetched on first use).

The API client and VAD are validated against the real service
(`pytest -m live` sends one short WAV and asserts the response shape); the
mic-to-document round trip is inherently manual — it needs you to speak.

---

## How it works

**Segmenter.** Silero VAD (small ONNX model, CPU-fast) marks speech vs silence
per 32 ms frame. `CutPolicy` closes a segment when silence runs long enough after
enough content — the cut lands *in the gap*, so no word is split. If a speaker
runs past 108 seconds without pausing, the policy forces a cut and prepends a
2-second overlap to the next segment so the boundary region is transcribed twice.

**Scheduler.** Segments transcribe in parallel under a concurrency cap, with
retries on retryable failures (5xx/timeout) from retained in-memory audio — a
chunked request body can't be replayed, so each attempt re-sends the full bytes.
Backpressure widens the pause threshold when the queue backs up. A permanently
failed segment never vanishes: it becomes a **visible gap marker** that still
advances the document.

**Assembler.** Results arrive out of order and are re-sequenced by `seq`. Pause
durations become sentence / paragraph / section breaks. Forced-cut seams are
spliced by aligning the overlap and keeping the higher-confidence word on
disagreement — or, if no alignment is confident, marked visibly rather than
silently duplicated. A glossary learns terms from early segments and feeds them
as keyterms / prompt context into later ones; a final consistency pass normalises
any residual variants to the majority form.

---

## Hard API facts baked in

Verified against the Dictation API and treated as ground truth:

- Endpoint `POST /v1/transcribe/live`; `Authorization: <raw key>` — **no `Bearer`
  prefix**. An **invalid key returns 404**, not 401.
- `multipart/form-data` with parts in order **`config` (JSON) then `audio`** — wrong
  order or missing config is a 400.
- **WAV or raw 16-bit PCM only** (else 415); raw PCM needs `sample_rate` + `channels`.
- **120 s hard max per call** — Longhand ceilings content at 108 s (10 s margin).
- Send only one of `stt_prompt`/`prompt`; only one of
  `keyterms_prompt`/`keyterms`/`word_boost`; keyterms cap 100 terms **and** 8000 chars.
- A **`200` with `llm_error` and a null `llm_response` is a success** — fall back to
  the verbatim `text`.

Every one of these is a guard or mapping in `stt/client.py`, unit-tested without
a network.

---

## Testing

- **Tests never touch the network.** Only tests marked `@pytest.mark.live` may, and
  they are excluded from the default run.
- All development runs against `FakeDictationClient` / fixture clients, which can
  simulate latency, out-of-order returns, 5xx, timeouts, and `llm_error`.
- The splicer and keyterm packer carry property tests — their invariants are easy
  to state and easy to break.

**140 offline tests, 0 network.** See `LONGHAND_SPEC.md` for the full design.

---

## License

MIT — see [`LICENSE`](LICENSE).
