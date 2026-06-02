# TigerForm ⛳

**An AI golf-swing monitor that compares your mechanics to Tiger Woods' swing and gives corrective feedback.**

TigerForm takes a swing video, extracts body pose with computer vision, derives
scale-invariant biomechanical features (turn, X-factor, arm extension, tempo,
posture, head stability…), scores how closely the swing matches a Tiger Woods
reference, and generates prioritized coaching feedback.

> CS 395/396 Final Project — Harrison Gillespie & Ian Evensen.

---

## What it does

1. **Pose estimation** — MediaPipe BlazePose extracts 33 body landmarks per frame.
2. **Swing segmentation** — locates the 8 standard swing events (address → top →
   impact → finish) from the lead-hand height profile.
3. **Club approximation** — estimates the shaft from the arms/hands (MediaPipe
   doesn't see the club; ball tracking is intentionally out of scope).
4. **Biomechanical features** — view-robust joint angles + tempo + X-factor +
   head stability, normalized so golfers of different size are comparable.
5. **Comparison** — z-score deviations vs. a **Tiger reference template**, a
   0–100 **similarity score**, DTW sequence matching, and a **Tiger-vs-amateur
   discriminator** whose feature importances explain *which* mechanics differ.
6. **Feedback** — a rule table maps deviations to drills; the Claude API phrases
   them into natural coaching (deterministic template fallback when offline).
7. **App** — a Streamlit UI for uploading a swing and viewing the annotated
   overlay, score, charts, and coaching.

### Design choices (vs. the original proposal)

- **Normalized, scale-invariant biomechanics** instead of matching Tiger's
  absolute joint angles — a golfer of any size can be compared fairly.
- **Similarity + discriminator** instead of a literal "good/bad" classifier
  (labeled "flawed Tiger swings" don't exist). The discriminator measures match
  to Tiger's *signature mechanics*, not an objective quality verdict.
- **Body + approximate club**; ball-flight tracking dropped as a stretch goal.

---

## Quick start

```bash
cd tigerform
py -3.12 -m venv .venv
# bash / Git Bash:
source .venv/Scripts/activate
# PowerShell:
# .\.venv\Scripts\Activate.ps1
# or cmd.exe:
# .\.venv\Scripts\activate.bat
python -m pip install -U pip
python -m pip install -r requirements.txt

# Build the Tiger reference + discriminator (uses synthetic swings if you have
# no real clips — works out of the box):
python scripts/build_reference.py
python scripts/train_discriminator.py

# Try the whole pipeline on a synthetic swing (no video needed):
python scripts/demo.py --kind amateur

# Or analyze a real swing video:
python -m tigerform.pipeline data/raw/your_swing.mp4

# Launch the web app:
streamlit run app/streamlit_app.py
```

Use Python 3.11 or 3.12 for this project. Python 3.13 is not supported by the
current NumPy/SciPy stack pinned here.

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to get
Claude-phrased coaching (otherwise a deterministic template is used).

### Using real data

- Put Tiger reference clips (front-on / down-the-line, ≥3) in `data/reference/`
  and re-run `build_reference.py`.
- Put amateur clips in `data/amateur/` and re-run `train_discriminator.py`.
- `scripts/download_golfdb.py` explains how to get **GolfDB** (event-labeled
  swings) for the optional learned-segmentation upgrade and real negative-class
  swings.

> Footage is never committed — only derived features. Use clips under
> educational fair use.

---

## Architecture

```
src/tigerform/
  config.py      paths, landmark indices, swing events, feature registry
  geometry.py    angle / rotation / smoothing helpers
  ingest.py      video -> frames + metadata + framing warnings
  pose.py        MediaPipe -> smoothed landmark time series (PoseSequence)
  club.py        approximate club shaft from arms/hands
  events.py      heuristic swing-event segmentation
  normalize.py   torso-length normalization for positional features
  features.py    biomechanical feature vector + DTW time series
  synthetic.py   parametric swing generator (Tiger/amateur) for demo + tests
  analyze.py     pose -> club + events + features (light orchestration)
  reference.py   Tiger reference template (mean/std + curves), save/load
  model.py       Tiger-vs-amateur discriminator (sklearn), importances
  compare.py     z-scores, similarity score, DTW, ranked deviations
  feedback.py    coaching rules + Claude phrasing + offline fallback
  viz.py         annotated overlay video + comparison charts
  pipeline.py    end-to-end orchestration + CLI
scripts/         build_reference, train_discriminator, evaluate, demo, download_golfdb
app/             streamlit_app.py
tests/           geometry / features / events / compare / pipeline
```

`PoseSequence` is the central data object; everything downstream consumes it, so
swapping synthetic swings for real MediaPipe output (or a learned segmenter)
requires no changes to comparison/feedback/app code.

---

## Evaluation

```bash
python scripts/evaluate.py
```

Reports, on a held-out synthetic set:
- **Discriminator**: precision / recall / F1 / ROC-AUC.
- **Score separation**: Tiger-like swings should score clearly above amateur-like.
- **Event segmentation**: PCE-style accuracy (±3 frames) vs. known event frames.

`scripts/train_discriminator.py` also prints cross-validated metrics on the
training data.

---

## Limitations

- Single-camera 2D→3D is approximate; use consistent front-on / down-the-line
  framing and good lighting.
- The club shaft is approximated from the hands, not detected; no ball tracking.
- The bundled reference/discriminator are trained on a **synthetic** swing model
  for demonstration. Swap in real Tiger + amateur clips for meaningful real-world
  scores — the code path is identical.
- "Tiger-likeness" reflects similarity to Tiger's signature mechanics, not an
  absolute measure of a "good" swing.

## Testing

```bash
pytest
```

All tests run on synthetic swings — no video files, MediaPipe, or trained
artifacts required.
