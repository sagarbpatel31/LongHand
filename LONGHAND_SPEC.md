# Longhand — long-form dictation over a 120-second API

**One-liner:** AssemblyAI's Dictation API caps at two minutes. Longhand lets you
dictate for twenty and get back one coherent, formatted document — seamlessly
stitched, terminologically consistent, and paragraphed from the rhythm of your
own speech.

---

## 0. The two ideas the project rests on

### Idea 1 — cut at silence, not at the cap

The obvious approach is to chop every 120 seconds and concatenate. That severs
words mid-utterance at every boundary, and both adjacent segments mangle the
seam.

Instead, segment on **voice activity**. A cut placed inside a natural pause has
no word straddling it, so there is nothing to dedup. Boundary stitching stops
being the core problem and becomes a fallback for one rare case (a speaker who
runs past 110 seconds without pausing).

This is the single decision that makes the project shippable in 15 hours. Say it
out loud in the demo.

### Idea 2 — silence structure *is* document structure

VAD gives you pause durations for free. Naive chunking discards them. Longhand
uses them:

| Inter-segment pause | Inferred structure |
|---|---|
| < 1.2s | same paragraph, sentence break |
| 1.2s – 3s | new paragraph |
| > 3s | candidate section break |

Tune these against your own speech, not these numbers. The result is a formatted
document produced from how the person actually spoke — which no naive
implementation can do, because it threw the timing away.

---

## 1. API facts (verified — do not re-derive)

- `POST https://dictation.assemblyai.com/v1/transcribe/live`
- `Authorization: <RAW_KEY>` — **no `Bearer` prefix**
- **Invalid key returns `404`, not `401`.** Treat any 404 as auth failure.
- `multipart/form-data`, parts **in order**: `config` (JSON) then `audio`.
  Audio before config, or no config → `400`.
- **WAV or raw 16-bit PCM only.** MP3/M4A/OGG/WebM → `415`
- **Max 120 seconds per call**
- Raw PCM requires `sample_rate` + `channels` in config
- Config: `sample_rate`, `channels`, `language_codes`, `stt_prompt` (≤6000
  chars; also accepted as `prompt` — send one, not both), `keyterms_prompt`
  (≤100 terms **and** ≤8000 chars; aliases `keyterms`/`word_boost` — send only
  one of the three), `llm_instruction` (≤2048 chars)
- Response: `text` (verbatim, never LLM-touched), `words[]` of
  `{text, confidence}`, `confidence`, `llm_response`, `llm_error`,
  `audio_duration_ms`, `session_id`, `request_time_ms`, `sync_time_ms`
- **Rewrite is best-effort.** `llm_error: "timeout"` still returns `200` with a
  good transcription. `llm_response: null` → fall back to `text`. Never treat a
  non-null `llm_error` as a failed request.
- HTTP timeout: **90s**. Rewrite has a 5s internal deadline.
- **A chunked body cannot be replayed.** Keep each segment's audio in memory
  until its response is confirmed, or you cannot retry.
- Don't let a chunked upload go silent mid-body — it gets timed out, not held.
- `warm()` pays DNS/TCP/TLS up front. Call it at session start *and* keep a warm
  spare connection during the session.
- Python SDK: `DictationTranscriber`, `transcribe_live(iterator, config)`,
  `open_live(config)` + `session.write(bytes)`. `AsyncDictationTranscriber` is
  the asyncio counterpart — **use the async one**, you need concurrency.

---

## 2. Architecture

```
mic (sounddevice, 16kHz mono int16)
   │
   ▼
RingBuffer ──────────────────────┐  (keeps audio for retry + forced-cut overlap)
   │                             │
   ▼                             │
VadSegmenter                     │
   │  emits SegmentClosed(seq, audio, t_start, t_end, lead_pause)
   ▼                             │
SegmentScheduler ◄───────────────┘
   │  ≤4 requests in flight, each streaming while its segment records
   ▼
AssemblyAI  ×N in parallel
   │
   ▼
OrderedAssembler ──► Glossary ──► feeds stt_prompt/keyterms of later segments
   │
   ▼
DocumentBuilder (pause→structure, seam splice, consistency pass)
   │
   ▼
UI (live document, in-flight shimmer, seam inspector)
```

### Modules

| Module | Responsibility |
|---|---|
| `capture/mic.py` | sounddevice RawInputStream → int16 frames |
| `capture/ring.py` | rolling buffer, retains audio until segment confirmed |
| `segment/vad.py` | Silero VAD → speech/silence decisions |
| `segment/policy.py` | cut rules, min/max length, forced cuts |
| `stt/client.py` | async Dictation wrapper — **interface, so it can be faked** |
| `stt/scheduler.py` | concurrency cap, retries, backpressure |
| `assemble/order.py` | out-of-order arrival → ordered document |
| `assemble/splice.py` | overlap alignment for forced cuts |
| `assemble/glossary.py` | running term registry, consistency normalization |
| `assemble/structure.py` | pause durations → paragraphs and sections |
| `ui/` | FastAPI + one HTML page over WebSocket |
| `bench/` | fixture replay, naive-vs-Longhand comparison |

---

## 3. The segmenter

Use **Silero VAD** (small ONNX model, CPU-fast) over WebRTC VAD — noticeably
better on real speech, and worth the extra dependency.

### Cut policy

```
MIN_SEGMENT      =   6s   # don't fragment into tiny requests
TARGET_PAUSE     = 600ms  # silence long enough to cut cleanly
MAX_SEGMENT      = 110s   # hard ceiling, 10s margin under the API's 120s
FORCED_OVERLAP   =   2s   # only used when MAX_SEGMENT is hit
```

Close a segment when:
1. silence ≥ `TARGET_PAUSE` **and** segment ≥ `MIN_SEGMENT` → **clean cut**, or
2. segment length hits `MAX_SEGMENT` → **forced cut**, retain the last
   `FORCED_OVERLAP` of audio to prepend to the next segment

Record `lead_pause` (silence duration *before* this segment began) on every
segment. That's what `structure.py` consumes later.

**Instrument the forced-cut rate.** If it exceeds ~5% of segments on natural
speech, your `TARGET_PAUSE` is too strict. Log it and put the number in the
README.

---

## 4. Streaming and concurrency

### Per segment

Open the request on **speech onset**, not on segment close. Config goes out
immediately, audio streams as it's captured, and the body closes when the VAD
says cut. The user waits only for the tail to process — typically well under a
second.

### Scheduling

- Cap at **4 concurrent** in-flight requests.
- Each segment carries a monotonic `seq`. Responses arrive out of order; the
  assembler holds gaps and emits in order.
- **Retry:** on 5xx or timeout, resend from the retained in-memory audio (not
  chunked this time — you have the whole thing, send it with a Content-Length).
  Max 2 retries, exponential backoff.
- **Backpressure:** if 4 are in flight and a new segment closes, queue it. If the
  queue exceeds 8, widen `TARGET_PAUSE` adaptively to produce fewer, longer
  segments.
- A permanently failed segment leaves a visible gap marker in the document. Never
  silently drop audio.

---

## 5. Terminology carryover

Per-segment transcription has no memory. A name transcribed correctly in segment
1 can drift in segment 7. Three mechanisms, in increasing strength:

### 5.1 `stt_prompt` chaining
Each segment's `stt_prompt` carries the tail (~200 chars) of the most recent
*available* transcript, plus a one-line situational description.

### 5.2 Running glossary
Accumulate rare/proper-noun terms across the session with occurrence counts. Feed
the top-N (under the 100-term / 8000-char cap) as `keyterms_prompt` on every
subsequent segment. Self-reinforcing: once a term lands correctly, it's pinned
for the rest of the session.

**Graceful degradation:** under parallel load, segment N+1 may launch before
segment N's transcript exists. Use the most recent *available* transcript rather
than blocking. In practice responses return in ~1s and segments are 6–30s apart,
so the current transcript is almost always there; only under load does context
get slightly staler. State this tradeoff in the README — it's a good design
answer if a judge asks.

### 5.3 Post-hoc consistency pass
After assembly, cluster rare terms by edit distance + metaphone. If "Kubernetes"
appears 8 times and "Coobernetes" once, normalize the outlier to the majority
form. This catches what keyterms missed.

**Report this as a metric:** terminology consistency rate, with and without
carryover. It's a second chart, and it's cheap to produce.

---

## 6. Seam handling

### Clean cuts (the common path)
Nothing to splice. Concatenate, and let `structure.py` decide whether the seam is
a sentence break, a paragraph break, or a section break based on `lead_pause`.

### Forced cuts (the rare path)
Both segments transcribed the 2s overlap region. To splice:

1. Take the last ~15 words of segment N and the first ~15 of segment N+1.
2. Normalize (lowercase, strip punctuation) and find the best alignment —
   longest common subsequence over the token sequences.
3. Splice at the alignment midpoint; where the two disagree on a word, keep the
   side with higher per-word `confidence`.
4. If no alignment scores above threshold, fall back to naive concatenation and
   **mark the seam visibly** in the UI. An honest visible seam beats a silent
   duplication.

---

## 7. The rewrite

Per-segment `llm_response` handles what it's designed for and what is genuinely
segment-local: filler removal, self-corrections, punctuation, capitalization.
Leave `llm_instruction` unset for the default cleanup unless you have a reason.

Document-level structure does **not** come from the rewrite — it comes from
`structure.py` and the pause signal. Keep that separation clean; it's a large
part of why the architecture works.

Keep the verbatim `text` for every segment alongside the cleaned version. A
verbatim/clean toggle in the UI is a five-minute feature and a good demo beat.

---

## 8. Benchmark — build it by hour 7, not hour 12

### Fixtures without recording 10 minutes of ground truth

Recording and hand-transcribing a long session is expensive. Instead:

1. Record ~20 short clips (10–25s) and hand-type ground truth for each.
2. Concatenate them programmatically with **known, controlled silence gaps**
   (0.4s / 1.5s / 4s) into synthetic 5- and 10-minute sessions.
3. You now have exact ground truth for a long session, plus known correct
   paragraph boundaries to score `structure.py` against.

Store as `bench/fixtures/`. Every test replays these. **Never hit a live mic in a
test run** — deterministic tests, no credits burned.

### Conditions

| ID | Approach |
|---|---|
| A | naive 120s chop + concatenate (the baseline) |
| B | VAD segmentation, no carryover |
| C | VAD + glossary carryover + consistency pass (full system) |

### Metrics

- **WER on the full document** — A loses words at every boundary; that gap is
  your headline
- **boundary word-loss count** — words destroyed at seams, per session
- **terminology consistency rate** — B vs C
- **paragraph-boundary F1** against the known gaps in the synthetic fixture
- time-to-first-text, and steady-state lag behind the speaker
- forced-cut rate

---

## 9. Testing plan

1. **`FakeDictationClient`** (hour 1) — implements the same interface, and
   "transcribes" by slicing fixture ground truth at the requested time offsets.
   Can simulate latency, out-of-order returns, 5xx, timeouts, and
   `llm_error`. Every piece of segmentation, scheduling, assembly, and splicing
   is then testable with zero network.
2. **VAD unit tests** on synthetic audio — tone/silence patterns with known cut
   points.
3. **Splice property test:** for any split point with overlap, stitched output
   equals the original. Generate split points randomly.
4. **Ordering test:** responses returned in reverse order still assemble
   correctly.
5. **Forced-cut test:** 130 seconds of continuous speech with no pause.
6. **Failure injection:** one segment fails permanently → document survives with
   a visible gap marker, and no other segment is lost.
7. **Retry test:** a chunked request fails mid-body → retried from the retained
   buffer, not lost. (This is the bug that will otherwise eat your evening.)
8. **Backpressure test:** 20 segments closing faster than the API returns.
9. **Memory bound:** 20 minutes at 16kHz mono int16 ≈ 38MB. Assert the buffer
   doesn't grow unbounded after segments are confirmed.
10. **One live smoke test**, `@pytest.mark.live`, excluded from the default run.

---

## 10. Hour-by-hour (15h)

| Hours | Work |
|---|---|
| 0–1 | Spike: one real call, confirm PCM + chunked upload, measure latency. Write `FakeDictationClient`. |
| 1–3 | Mic capture, ring buffer, Silero VAD segmenter + unit tests. |
| 3–5 | Async scheduler: parallel in-flight, ordered assembly, retries, backpressure. |
| 5–7 | **Benchmark harness + synthetic fixtures.** Get A vs B WER on the board. |
| 7–9 | Glossary carryover, `stt_prompt` chaining, consistency pass. Condition C. |
| 9–10 | Forced-cut overlap splice + property tests. |
| 10–12 | UI: live document, in-flight shimmer, seam inspector, verbatim toggle, pause→structure. |
| 12–14 | Demo recording, README, charts, architecture diagram. |
| 14–15 | Buffer. There will be something. |

**Hour-5 checkpoint:** if a 3-minute fixture doesn't transcribe end-to-end
through the scheduler, drop the glossary work and ship clean stitching alone.
Stitching plus the pause-structure feature is already a complete, defensible
project.

---

## 11. Demo script (2 minutes)

1. **The problem.** Naive 120s chop on a 5-minute fixture. Scroll to a boundary,
   show the mangled words. (~20s)
2. **Live dictation.** Speak for 45 seconds with deliberate pauses. Text lands
   about a second after each pause; the document paragraphs itself as you go.
   (~45s)
3. **Seam inspector on.** Show where the cuts actually fell — inside silences,
   never inside words. (~15s)
4. **Consistency.** A term that drifts without carryover, pinned with it. (~20s)
5. **The chart.** A vs C on full-document WER and boundary word-loss. (~20s)

Pre-record the 10-minute run. Do the short one live.

---

## 12. Things that will bite you

- **`config` after `audio`, or missing** → `400`. Order is load-bearing.
- Sending both `prompt` and `stt_prompt`, or two of
  `keyterms_prompt`/`keyterms`/`word_boost` → `400`.
- **A chunked body cannot be replayed.** Retain segment audio until the response
  confirms, or retries are impossible.
- Silence mid-body on a chunked upload → timed out, not held open.
- Forgetting the 10s safety margin and sending 121 seconds.
- Treating `llm_error` as a request failure.
- Async cancellation leaking sockets when a segment is abandoned — close
  sessions in `finally`.
- A speaker who pauses constantly → dozens of tiny segments. `MIN_SEGMENT`
  exists for this; test with a hesitant-speech fixture.
- Burning the morning on UI. The chart wins, not the CSS.
