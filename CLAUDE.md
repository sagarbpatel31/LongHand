# Longhand

Long-form dictation over AssemblyAI's Dictation API, which caps at 120 seconds
per call. We segment on natural pauses, transcribe segments in parallel, and
assemble one coherent document.

Full design: see `LONGHAND_SPEC.md`. Read it before making architectural
decisions. This file is the operating rules.

**This is a hackathon build with a hard deadline.** Prefer working and tested
over complete. When a choice is between more features and more confidence in
what exists, choose confidence.

---

## The two ideas the project rests on

1. **Cut at silence, not at the cap.** A cut placed inside a pause has no word
   straddling it, so there is nothing to deduplicate. Overlap-splicing exists
   only as a fallback for forced cuts.
2. **Silence structure is document structure.** Pause durations between segments
   determine sentence / paragraph / section breaks. Never discard timing data.

---

## Hard API facts — treat as verified, do not re-derive or "fix"

- `POST https://dictation.assemblyai.com/v1/transcribe/live`
- Header: `Authorization: <RAW_KEY>` — **no `Bearer` prefix**
- **Invalid key returns `404`, not `401`.** Any 404 must surface as an auth error.
- `multipart/form-data`, parts **in this order**: `config` (JSON), then `audio`.
  Wrong order or missing config → `400`.
- **WAV or raw 16-bit PCM only.** Anything else → `415`.
- **120 second maximum per call.** We use 110s as the ceiling.
- Raw PCM requires `sample_rate` and `channels` in config.
- Send only ONE of `stt_prompt` / `prompt`. Send only ONE of
  `keyterms_prompt` / `keyterms` / `word_boost`. Two of either → `400`.
- `keyterms_prompt` cap: 100 terms AND 8000 chars — both, simultaneously.
- Response fields: `text` (verbatim), `words[]` of `{text, confidence}`,
  `confidence`, `llm_response`, `llm_error`, `audio_duration_ms`, `session_id`,
  `request_time_ms`, `sync_time_ms`.
- **`llm_error` is not a request failure.** A `200` with
  `llm_error: "timeout"` and `llm_response: null` is a success — fall back to
  `text`.
- **A chunked request body cannot be replayed.** Retain each segment's audio in
  memory until its response is confirmed, or retries are impossible.
- HTTP client timeout: 90s.

---

## Rules for this repo

### Credits and the network
- We have a fixed, small credit budget.
- **Tests never touch the network.** Only tests marked `@pytest.mark.live` may,
  and they are excluded from the default run.
- **Never run a live test or a real API call without asking me first.**
- All development and debugging uses `FakeDictationClient` against fixtures.

### Testing
- `stt/client.py` defines an interface. `FakeDictationClient` implements it and
  must be able to simulate: latency, out-of-order returns, 5xx, timeouts, and
  `llm_error`.
- Write the test before or alongside the code, not after.
- Run the suite before telling me something works. If you haven't run it, say so.
- Prefer property tests for the splicer and the keyterm packer — they have
  invariants that are easy to state and easy to break.

### Debugging
- When something fails, find the actual cause before changing code. No
  speculative fixes, no "try this and see."
- Add a failing test that reproduces the bug first, then fix it.
- Structured logging with `seq` on every segment-related log line. Without it,
  concurrency bugs are unreadable.

### Style
- Python 3.11+, asyncio. Use `AsyncDictationTranscriber`, not the sync one.
- Type hints on module boundaries.
- Small modules matching the layout in the spec.
- No new dependency without telling me what it's for.

---

## Never do these

- Silently drop audio. A failed segment leaves a visible gap marker in the
  document; it does not vanish.
- Discard pause/timing data anywhere in the pipeline.
- Concatenate across a forced-cut seam without either splicing or visibly
  marking it.
- Send more than 110 seconds of audio in one request.
- Commit the API key. It lives in `.env`, which is gitignored.

---

## Commands

```bash
pytest                    # default suite, no network
pytest -m live            # real API — ask me first
python -m longhand.bench  # benchmark: naive chop vs Longhand
python -m longhand        # run the app
```

---

## Current status

Hours 0–10 done (STT + scheduler + assembler + segmenter + benchmark + assembly
intelligence + forced-cut splice). **122 offline tests pass, 0 network**; 1 live
smoke deselected.

Hour 0–1 — STT layer:
- Skeleton: `pyproject.toml` (hatchling), `longhand/` package, `.env` via
  `settings.py`, `pytest` with a `live` marker excluded by default.
- `stt/client.py`: `DictationClient` interface + `AssemblyAIDictationClient`.
  Every hard API fact is a guard/mapping — 110s ceiling (pre-network), PCM/WAV
  only, raw `Authorization` (no Bearer), part order config→audio (hand-built
  multipart), 404→`AuthError`, `llm_error` is success (`best_text` falls back),
  alias exclusivity, keyterm caps, 5xx/timeout `retryable`.
- `stt/fake.py`: `FakeDictationClient` — latency, out-of-order, 5xx, timeout,
  `llm_error`, transient-then-success. The offline stand-in for everything.

Hours 3–5 — async core (Part A, zero new deps):
- `segment/types.py`: `SegmentClosed` (leaf, breaks import cycles).
- `assemble/order.py`: `OrderedAssembler` + `AssembledSegment` — out-of-order →
  seq-ordered; permanent failure → visible gap that never blocks/drops (§9.4/§9.6).
- `stt/scheduler.py`: `SegmentScheduler` (+ `transcribe_segments` helper) —
  4-concurrent cap, retry retryable-only (max 2, injectable exp backoff),
  backpressure signal (queue>8), gap on permanent failure, finally-cleanup
  (§9.4/§9.6/§9.7/§9.8/§12).

Hours 1–3 — segmenter (Part B):
- `capture/ring.py`: bounded `RingBuffer`, confirm/retain/slice, memory-bounded
  (§9.9).
- `segment/vad.py`: `Vad` interface + `FakeVad` + `frames_from_spans`;
  `SileroVad` (onnxruntime via `silero-vad-notorch`, lazy import).
- `segment/policy.py`: `CutPolicy` — clean cut at silence, forced cut at 108s
  with 2s overlap, `lead_pause`, forced-cut rate (§9.2/§9.5 + property test).
- `capture/mic.py`: `MicCapture` sounddevice adapter (live-only, not unit-tested).
- `tests/test_pipeline_offline.py`: hour-5 gate — frames→policy→scheduler→
  assembler, all synthetic, passes.

Decisions locked:
- Buffered upload (not streaming-on-onset): retry re-sends retained bytes; §9.7
  satisfied via retained-buffer retry. Streaming deferred.
- Policy content ceiling 108s < client 110s (room for the 2s overlap).

Hours 5–7 — benchmark (Part done):
- `bench/fixtures.py`: synthetic sessions (TimedWord timeline + int16 audio +
  frame_script + exact ground truth) from clips and known gaps; `demo_fixture`
  (~10 min).
- `bench/fixture_client.py`: `FixtureDictationClient` — idealized ASR that slices
  ground truth by segment time-range; straddling words become mangled fragments
  (spec §0). `words_lost` counts boundary losses.
- `bench/wer.py`: token WER (Levenshtein). `bench/harness.py` + `__main__.py`:
  A (naive 110s chop) vs B (VAD) through the real scheduler → table.
  `python -m longhand.bench`. On 10-min demo: A loses 4 boundary words, B loses 0.

Hours 7–9 — assembly intelligence (done):
- `assemble/structure.py`: `build_document` turns `lead_pause` into
  sentence/paragraph/section blocks (bands 1.2/3.0); gaps render a visible marker.
  Idea #2 ("silence IS structure").
- `assemble/glossary.py`: `Glossary` (observe/keyterms/stt_prompt, no-dep term
  extraction), `consistency_pass` (phonetic key + `wer.edit_distance` clustering,
  strict-majority normalization). No new deps.
- `stt/scheduler.py`: added `prepare_config` + `on_result` hooks (keyword-only,
  backward-compatible) so a glossary can enrich per-segment config and learn from
  results. `transcribe_segments` gained `max_in_flight`/`prepare_config`/`on_result`.
- Bench condition C: `DriftTerm`/`DriftFixture`/`build_drift_fixture`,
  `DriftDictationClient` (tagged terms drift unless pinned), `bench/metrics.py`
  (terminology consistency, paragraph-boundary F1), `compare_drift` → [A,B,C].
  C runs serial (`max_in_flight=1`) + consistency pass for determinism.

`python -m longhand.bench` (10-min demo) shows three charts:
- WER: A 2.7% > B 2.3% > C 0.0%
- terminology consistency: A/B 69% → C 100%
- paragraph-boundary F1: A 0.00 → B/C 1.00

Hour 9–10 — forced-cut splice (done):
- `assemble/splice.py`: `splice_words`/`splice_texts` align the tail of a forced
  segment against the head of the next (overlap match), keep the overlap once
  (higher-confidence word on disagreement), drop the duplication; fall back to
  naive concat + visible `SEAM_MARKER` when no alignment clears threshold.
- Property test: any split point + overlap reconstructs the original (§9.3).
  Integration: real 130s forced cut → overlapping transcripts → splice → exact.

Deferred / not yet run:
- The real spike call — `test_live_smoke.py` ready; run `pytest -m live` manually.
- Live mic + real Silero validation (speak → segments emit → transcript). Manual.
- Next up (§10): 10–12 UI (FastAPI + one HTML page over WebSocket: live doc,
  in-flight shimmer, seam inspector, verbatim toggle, pause→structure). Spec warns
  "the chart wins, not the CSS." Then 12–14 README + charts + demo recording.

Deps: runtime `httpx`, `python-dotenv`, `numpy`, `onnxruntime`,
`silero-vad-notorch`, `sounddevice` (last three lazy/live-only). Dev `pytest`,
`pytest-asyncio`, `hypothesis`.
