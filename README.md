# FlameAnalyzer - inference backend

Headless, cloud-ready flame analysis:
**detection -> segmentation -> flame colour (CIELAB, five clustering
algorithms) -> material matching -> fire class -> JSON**.

There is no GUI, no browser UI and no tunnel. The ML pipeline is
transport-agnostic; a thin FastAPI layer (`app/api.py`) and a CLI
(`app/cli.py`) sit on top of it.

Optionally, the API can attach a **secondary AI material analysis**
(`ai_material_analysis`, powered by Gemini) alongside the deterministic
result. It is strictly additive: it receives only the extracted mean flame
RGB/LAB values and the canonical material vocabulary - never the image - and
any Gemini failure degrades to `{"available": false}` without affecting the
deterministic result. See [AI Material Analysis](#ai-material-analysis-secondary-gemini-powered).

A **video** is supported by the same building blocks through two further
endpoints, with no second classifier and no duplicated vocabulary:
`POST /material-identification` re-runs the deterministic matcher on the
aggregated flame colour the client measured across its frames, and
`POST /video-material-analysis` sends every frame's evidence to Gemini in
**one** request that returns **one** consolidated, video-level assessment. See
[Video material analysis](#video-material-analysis-one-consolidated-assessment-per-video).

```
HTTP request                                     CLI
    |                                               |
    v                                               v
app/api.py  -> app.imaging.decode_image_bytes   -> FlameAnalyzer.analyze_image()
                     |                                     |
                     +-------------- app/schemas.py <------+
                                      (JSON-safe result)
```

## Project layout

```
flame-analyzer/
├── app/
│   ├── __init__.py           # public exports
│   ├── analyzer.py           # FlameAnalyzer: model loading + pipeline orchestration
│   ├── api.py                # FastAPI app: POST /analyze, POST /material-identification, POST /video-material-analysis, GET /health, GET /, POST/GET /activity
│   ├── cli.py                # command line runner (prints JSON)
│   ├── color_analysis.py     # flame pixels -> LAB -> 5 clustering algorithms
│   ├── config.py             # Settings: all inference parameters in one place
│   ├── detection.py          # OBJ_best.pt fire detection
│   ├── errors.py             # AnalysisError hierarchy + error codes
│   ├── fire_classes.py       # material -> fire class + extinguishing agents
│   ├── flame_evidence.py     # measured flame evidence shared by the image + video prompts
│   ├── gemini_analysis.py    # secondary Gemini material analysis (optional, additive)
│   ├── imaging.py            # image validation / bytes -> BGR decoding
│   ├── mask.py               # mask -> base64 PNG codec (MASK_ENCODING)
│   ├── material_matching.py  # LAB matching against flame_dataset.json
│   ├── schemas.py            # response dataclasses + JSON-safe conversion
│   ├── segmentation.py       # SEG_best.pt flame segmentation + bbox fallback
│   ├── video_material.py     # video evidence aggregation + the two video endpoints
│   └── visualization.py      # optional debug renderers (never called by the API)
├── models/
│   ├── OBJ_best.pt           # custom fire detection model (do not replace)
│   └── SEG_best.pt           # custom flame segmentation model (do not replace)
├── data/
│   └── flame_dataset.json    # material database (reference flame colours + suppression data)
├── frontend/
│   └── index.html            # minimal reference UI: deterministic + AI material cards
├── tests/
│   ├── test_pipeline.py      # pipeline tests + real-weight integration test
│   ├── test_api.py           # FastAPI TestClient tests (stubbed analyzer)
│   ├── test_gemini_analysis.py  # secondary AI analysis tests (mocked Gemini client)
│   ├── test_material_identification.py  # deterministic video material endpoint tests
│   └── test_video_analysis.py          # consolidated video AI endpoint tests
├── main.py                   # CLI entry point
├── test_pipeline.py          # same CLI, kept as the standalone smoke-test script
├── requirements.txt
├── .env.example              # safe placeholder configuration
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

For a CUDA GPU, install the matching `torch` wheel before the rest.

## Usage

Analyse a local image and print the JSON result:

```bash
python main.py fire_1.jpg
python test_pipeline.py fire_1.jpg --pretty --verbose
```

Run the test suite (295 tests: pipeline, API, CLI, Gemini analysis with a
mocked client, the deterministic video endpoint, the consolidated video AI
endpoint, plus a real-weight integration test that is skipped when the
`.pt` files are absent):

```bash
pip install pytest httpx
python -m pytest tests -q
```

Use it as a library (this is exactly what the API layer does):

```python
from app import FlameAnalyzer, decode_image_bytes

analyzer = FlameAnalyzer()               # models + database loaded once
result = analyzer.analyze_image(decode_image_bytes(uploaded_bytes))
```

## Local API

The FastAPI app in `app/api.py` is a thin wrapper: it decodes the upload,
calls the same `FlameAnalyzer.analyze_image()` as the CLI, and returns the same
JSON. No pipeline logic lives in the API layer.

```bash
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

```bash
# PowerShell
curl.exe -X POST http://localhost:8000/analyze -F "image=@5.png"
```

```bash
# bash
curl -X POST http://localhost:8000/analyze -F "image=@5.png"
```

Interactive docs: <http://localhost:8000/docs> (Swagger UI) and
<http://localhost:8000/redoc>. Both are served locally and expose no
third-party calls.

### Endpoints

| Method | Path       | Purpose                                              |
| ------ | ---------- | ---------------------------------------------------- |
| `GET`  | `/`        | Service name, status, version                        |
| `GET`  | `/health`  | `200` when the models are loaded, `503` otherwise   |
| `POST` | `/analyze` | `multipart/form-data` with a single `image` field    |
| `POST` | `/material-identification` | Deterministic material match for one aggregated flame colour (JSON body) |
| `POST` | `/video-material-analysis` | **One** Gemini request over all frame evidence → **one** consolidated assessment (JSON body) |
| `POST` | `/activity` | EC2 inactivity watchdog heartbeat (touches `/var/run/flame-analyzer-last-activity`) |
| `GET`  | `/activity` | Same `{"status": "active"}` response, for testing    |

`POST /activity` takes no body and no authentication. It updates the
modification time of `/var/run/flame-analyzer-last-activity` (override with
`FLAME_ACTIVITY_FILE`) and returns `200 {"status": "active"}`. The update is
best-effort: if the file cannot be written the endpoint still returns 200, so
a transient filesystem problem never breaks `/analyze` or the frontend.

`POST /analyze`:

- `image` is **required**. Accepts any format OpenCV can decode (JPEG, PNG,
  BMP, WebP); there is no need to convert files first.
- Max upload size is `FLAME_MAX_IMAGE_MB` (default `10` MiB), enforced from
  the request headers and again while reading the body.
- `200` is returned both for a successful analysis **and** for a successfully
  processed image in which no fire was found (`success: false` with
  `NO_FIRE_DETECTED` / `NO_FLAME_PIXELS`). A 4xx/5xx always means the request
  could not be analysed.

| Status | Code                                     | Meaning                                        |
| ------ | ---------------------------------------- | ---------------------------------------------- |
| `200`  | –                                        | analysed (`success: true`) or no fire found     |
| `400`  | `INVALID_IMAGE`                          | the bytes are not a decodable image            |
| `413`  | `IMAGE_TOO_LARGE`                        | upload exceeds `FLAME_MAX_IMAGE_MB`            |
| `422`  | `VALIDATION_ERROR`, `MISSING_IMAGE`      | the `image` field is missing or empty          |
| `500`  | `INFERENCE_ERROR`                        | unexpected failure (no stack trace is leaked)  |
| `503`  | `SERVICE_UNAVAILABLE`, `MISSING_MODEL`, `MISSING_DATABASE` | models or database not loaded |

Every error body uses one shape:

```json
{
  "success": false,
  "error": {"code": "INVALID_IMAGE", "message": "The uploaded file is not a valid image."}
}
```

### Operational notes

- **One model instance per worker process.** `FlameAnalyzer` is constructed in
  the FastAPI lifespan handler and reused for every request. Running
  `--workers 4` therefore loads four copies of the models - that is the
  intended trade-off for using multiple cores.
- **Concurrent requests are serialised per worker** by a lock, because the
  pipeline is not re-entrant. The event loop stays responsive; only inference
  is queued.
- **Uploads are never written to disk.** Bytes are decoded in memory and
  discarded after the response.
- **CORS is off by default.** Set `FLAME_CORS_ORIGINS` to a comma-separated
  list of browser origins to enable it. A `*` entry is supported but is never
  combined with credentials.
- **No authentication.** The app is intended for local testing and for
  deployment behind a gateway that handles access control.
- **Config:** see `.env.example`. `FLAME_CORS_ORIGINS` and
  `FLAME_MAX_IMAGE_MB` are the only new variables; every inference setting is
  read from the same central `app/config.py` used by the CLI.

## Response shape

The example below is a real result for `5.png` (966x681), with long
repetitive fields abbreviated as `"..."`. Re-running
`python main.py 5.png --pretty` reproduces it.

```json
{
  "success": true,
  "fire_detection": {
    "detected": true,
    "confidence": 0.7899,
    "bounding_box": {"x1": 208, "y1": 144, "x2": 530, "y2": 452},
    "bbox": {"x1": 208, "y1": 144, "x2": 530, "y2": 452}
  },
  "segmentation": {
    "available": true,
    "fallback_used": false,
    "fallback": false,
    "flame_pixel_count": 44820,
    "mask_area_ratio": 0.068131,
    "confidence": 0.9578,
    "mask_width": 966,
    "mask_height": 681,
    "mask_encoding": "png_base64",
    "mask": "iVBORw0KGgoAAAANSUhEUg...",
    "bbox_fallback_reason": null
  },
  "flame_analysis": {
    "kmeans": {
      "rgb": [255, 203, 89], "lab": [80.0, 12.0, 44.0],
      "method": "kmeans", "cluster_count": 4,
      "samples_used": 2000, "pixels_sampled": true,
      "dominant_cluster": 1,
      "dominant_color": {"rgb": [254, 241, 42], "lab": [93.0, -6.0, 65.0]},
      "centroids": [
        {"index": 0, "size": 318, "weight": 0.159,
         "rgb": [255, 149, 60], "lab": [72.0, 20.0, 50.0]},
        {"index": 1, "size": 903, "weight": 0.4515,
         "rgb": [254, 241, 42], "lab": [93.0, -6.0, 65.0]}
      ],
      "noise_count": 0,
      "representative": "unweighted mean of the K-Means centroids",
      "fallback": false
    },
    "gmm": {
      "rgb": [255, 221, 98], "method": "gmm", "cluster_count": 4,
      "dominant_cluster": 3, "noise_count": 0,
      "representative": "mixture-weight weighted mean of the GMM component means"
    },
    "bayesian_gmm": {
      "rgb": [255, 221, 98], "method": "bayesian_gmm", "cluster_count": 4,
      "dominant_cluster": 3, "noise_count": 0,
      "representative": "mixture-weight weighted mean of the variational Bayesian GMM component means"
    },
    "dbscan": {
      "rgb": [255, 222, 99], "method": "dbscan", "cluster_count": 1,
      "dominant_cluster": 0, "noise_count": 16,
      "representative": "unweighted mean of the DBSCAN cluster means (noise excluded)"
    },
    "agglomerative": {
      "rgb": [255, 196, 89], "method": "agglomerative", "cluster_count": 4,
      "dominant_cluster": 1, "noise_count": 0,
      "representative": "unweighted mean of the Ward agglomerative cluster means"
    },
    "mean_color": {"rgb": [255, 221, 98], "lab": [89.09, -0.36, 63.31]},
    "flame_pixel_count": 44820,
    "samples_used": 2000,
    "pixels_sampled": true,
    "n_clusters": 4,
    "algorithms": ["K-Means", "GMM", "Bayesian GMM", "DBSCAN", "Agglomerative"],
    "skipped_reason": null
  },
  "material_analysis": {
    "primary_material": "Natural Fibers",
    "similarity": 0.9268,
    "alternatives": [
      {"material": "Spray Products", "similarity": 0.8469},
      {"material": "Electrical Components", "similarity": 0.7224}
    ],
    "database_notes": "...",
    "score_basis": "mean LAB distance to flame_dataset.json reference colours"
  },
  "suppression_information": {
    "source": "flame_dataset.json",
    "material": "Natural Fibers",
    "methods": ["Water", "CO2", "Foam"],
    "database_notes": "..."
  },
  "fire_class": {
    "class": "Class A",
    "description": "Ordinary combustibles",
    "confidence": 0.9268,
    "material": "Natural Fibers",
    "basis": "material 'Natural Fibers' -> Class A via app/fire_classes.py (documented material -> fire class mapping)",
    "mapping_source": "app/fire_classes.py (documented material -> fire class mapping)",
    "notes": "Derived from the matched material via the documented material -> fire class mapping in app/fire_classes.py, which is itself derived from the extinguishers recorded in flame_dataset.json. The fire class is not predicted by the detection model, and the confidence is the material match's distance-based similarity, not a calibrated probability."
  },
  "extinguishing_agents": [
    {"name": "Water", "compound": null, "type": "cooling",
     "source": "flame_dataset.json", "fire_class": "Class A",
     "compound_basis": "name verbatim from flame_dataset.json; the dataset records no chemical identity, so no compound formula is asserted"},
    {"name": "CO2", "compound": null, "type": "oxygen_displacement",
     "source": "flame_dataset.json", "fire_class": "Class A",
     "compound_basis": "name verbatim from flame_dataset.json; the dataset records no chemical identity, so no compound formula is asserted"},
    {"name": "Foam", "compound": null, "type": "blanketing",
     "source": "flame_dataset.json", "fire_class": "Class A",
     "compound_basis": "name verbatim from flame_dataset.json; the dataset records no chemical identity, so no compound formula is asserted"}
  ],
  "timing": {"total_ms": 2395.26, "detection_ms": 1870.71, "segmentation_ms": 245.38,
              "color_ms": 276.21, "material_ms": 1.38, "ai_material_ms": 412.7},
  "ai_material_analysis": {
    "available": true,
    "primary_material": "Wood Materials",
    "matches": [
      {"rank": 1, "material": "Wood Materials", "confidence_percent": 78.0,
       "reason": "Yellow-orange flame with a blue base matches the measured colour."},
      {"rank": 2, "material": "Paper Products(Wood material)", "confidence_percent": 71.0,
       "reason": "Orange to bright yellow flame is compatible with the measured tone."}
    ],
    "overall_confidence_level": "medium",
    "uncertain": false,
    "reasoning_summary": "The measured yellow-orange flame is most compatible with organic, carbon-based materials; flame colour alone is not a reliable identification."
  },
  "error": null
}
```

`ai_material_analysis` is present in every successful response. When Gemini is
disabled it is `{"available": false, "error": "Gemini material analysis is
disabled (GEMINI_ENABLED is not enabled)."}` and `timing` carries no
`ai_material_ms` key.

The whole body is about 11 KB, of which the mask accounts for roughly 4.4 KB.

`similarity` is a **distance-based score in [0, 1]**, not a calibrated
probability - it is named accordingly and must not be presented as a confidence.
`fire_class.confidence` is that same material similarity carried through the
class mapping, not an independent classification score.

### Field notes

- **Mask vs. bounding box.** `fire_detection.bounding_box` (alias `bbox`) is the
  *detection rectangle*. `segmentation.mask` is the *actual segmented flame
  region*, returned as a base64 PNG whose decoded size is exactly
  `mask_width` x `mask_height` - the analysed image's own dimensions - so a
  client can overlay it directly. The two are independent.
- **Mask encoding.** `mask` is a bare base64 string, not a data URI; prefix it
  with `data:image/png;base64,`. `mask_encoding` names the format and is
  currently always `png_base64`. A binary mask compresses very well: a 966x681
  mask of 44,820 pixels is about 4.4 KB encoded, far smaller than sending the
  raw array. Set `FLAME_EMIT_MASK=0` to omit it.
- **`bbox_fallback_reason`.** Non-null means the segmentation model produced no
  usable mask and the detection box was rasterised instead, so `mask` is a
  rectangle rather than real segmentation output. `fallback_used` / `fallback`
  report the same condition.
- **`flame_analysis.algorithms`.** Display names of the five retained
  algorithms, in report order. Every algorithm that ran also appears as a key
  in `flame_analysis`; a mask too small to cluster sets `skipped_reason`
  instead. The `method` field inside each result is the machine name
  (`kmeans`, `bayesian_gmm`, ...).
- **`representative`.** How that algorithm turned its clusters into the single
  reported colour - the mean of the centroids for K-Means, the mixture-weighted
  mean of component means for the GMMs, the mean of the cluster means for
  Ward-linkage Agglomerative, and so on. It is reported so the number is
  reproducible rather than opaque.
- **`dbscan.noise_count`.** Points DBSCAN labelled noise (16 in the example
  above). The other algorithms report `0`; noise is a DBSCAN concept.
- **`centroids` / `dominant_color`.** Every cluster of that algorithm with its
  size and share of the samples, so the dominant-cluster extraction is
  inspectable rather than a black box.
- **`compound` is always `null`.** `flame_dataset.json` records agent names
  (`CO2`, `Water`, `Foam`, `Sand`, `Dry powder`, `Class D powder`,
  `Water spray`), not chemical identities, so no formulas are invented.
- **Compatibility aliases.** `bbox` mirrors `bounding_box` and `fallback`
  mirrors `fallback_used`, so consumers expecting either spelling keep working.
  `suppression_information` is likewise retained for existing consumers.

### Errors

Failures return HTTP-friendly JSON and never a stack trace:

```json
{
  "success": false,
  "error": {"code": "NO_FIRE_DETECTED",
            "message": "No fire region was detected in the supplied image."}
}
```

| Code | Meaning |
| --- | --- |
| `INVALID_IMAGE` | Empty, corrupt or undecodable upload |
| `NO_FIRE_DETECTED` | Nothing above the detection confidence threshold |
| `NO_FLAME_PIXELS` | Fire region found, but the mask was empty |
| `MISSING_MODEL` | `OBJ_best.pt` / `SEG_best.pt` missing or unloadable |
| `MISSING_DATABASE` | `flame_dataset.json` missing or malformed |
| `INFERENCE_ERROR` | Unexpected failure (details logged internally only) |

## AI Material Analysis (secondary, Gemini-powered)

`ai_material_analysis` is a **secondary, LLM-based interpretation** of the
numerically extracted flame-colour evidence. It is independent from the
deterministic matcher and is never allowed to alter the fire class, the
extinguishing agents or the deterministic `material_analysis`.

- **The original image is not sent to Gemini.** No image, no segmentation mask
  and no base64 data ever leave the backend. Gemini receives only the final
  whole-mask flame colour (`flame_analysis.mean_color`: mean RGB and mean LAB,
  exactly as `FlameAnalyzer` extracted them) plus the canonical material
  vocabulary and notes loaded from the same `flame_dataset.json`.
- **The vocabulary is fixed.** Gemini may only choose from the 18 canonical
  dataset categories; every response is validated against the vocabulary, the
  rank sequence (1-5) and the confidence range `[0, 100]`. Unknown materials,
  duplicates, wrong ranks, wrong candidate counts and malformed JSON are all
  rejected into a clean `{"available": false, "error": ...}` state.
- **The scores are heuristic.** `confidence_percent` values are the model's
  relative confidence allocation. They are **not calibrated probabilities** and
  are never derived from token probabilities.
- **Failure never fails the request.** If Gemini is disabled
  (`GEMINI_ENABLED=0`, the default), the API key is missing, the free-tier
  quota is exhausted, the request times out (`GEMINI_TIMEOUT_S`, default 30 s)
  or the payload fails validation, the deterministic FlameAnalyzer result is
  returned untouched and `ai_material_analysis.available` is `false`.
- **Timing** is reported separately as `timing.ai_material_ms`, only when the
  AI analysis actually ran.

The CLI is unaffected: it prints the deterministic `FlameAnalyzer` payload;
the AI analysis is attached by the API layer.

## Video material analysis (one consolidated assessment per video)

A video frontend analyses a handful of sampled frames with `POST /analyze` and
collects the measured evidence itself. Two endpoints consume that evidence.
Both reuse the existing pieces only - `app/material_matching.match_material`,
the canonical `data/flame_dataset.json`, `app/fire_classes.py` and the Gemini
client boundary - so there is **no second classifier, no duplicated material
vocabulary and no new dataset**.

### `POST /material-identification` (deterministic)

The client aggregates its frames into one flame colour and sends it:

```json
{"rgb": [255, 221, 98], "lab": [89.09, -0.36, 63.31]}
```

That colour goes straight into the **same** `match_material()` used by
`/analyze`, against the **same** dataset, and the ranked match, alternatives,
fire class and extinguishing agents come back in the structure `/analyze`
returns. There is no Gemini call, and **no YOLO detection or segmentation
runs here** - the client already has the evidence. (The example below is the
real response to that exact body; long fields are abbreviated.)

```json
{
  "success": true,
  "material_analysis": {
    "primary_material": "Natural Fibers",
    "similarity": 0.921,
    "alternatives": [
      {"material": "Paper Products(Wood material)", "similarity": 0.9151},
      {"material": "Wax Materials", "similarity": 0.9151},
      {"material": "Organic Waste", "similarity": 0.8956}
    ],
    "database_notes": "Yellow flame, cotton burns fast, wool self-extinguishes, ...",
    "score_basis": "mean LAB distance to flame_dataset.json reference colours"
  },
  "primary_material": "Natural Fibers",
  "similarity": 0.921,
  "alternatives": [{"material": "Paper Products(Wood material)", "similarity": 0.9151}],
  "suppression_information": {"source": "flame_dataset.json", "material": "Natural Fibers",
                              "methods": ["Water", "CO2", "Foam"], "database_notes": "..."},
  "fire_class": {"class": "Class A", "description": "Ordinary combustibles",
                 "confidence": 0.921, "material": "Natural Fibers",
                 "basis": "material 'Natural Fibers' -> Class A via app/fire_classes.py",
                 "mapping_source": "app/fire_classes.py", "notes": "..."},
  "extinguishing_agents": [
    {"name": "Water", "compound": null, "type": "cooling",
     "source": "flame_dataset.json", "fire_class": "Class A", "compound_basis": "..."},
    {"name": "CO2", "compound": null, "type": "oxygen_displacement",
     "source": "flame_dataset.json", "fire_class": "Class A", "compound_basis": "..."},
    {"name": "Foam", "compound": null, "type": "blanketing",
     "source": "flame_dataset.json", "fire_class": "Class A", "compound_basis": "..."}
  ]
}
```

`primary_material`, `similarity` and `alternatives` are mirrors of the nested
`material_analysis` fields, following the project's existing alias convention
(`bbox` mirrors `bounding_box`). **The video fire class comes from this
deterministic match, never from Gemini** - that is what makes it trustworthy.

### `POST /video-material-analysis` (one Gemini request, one result)

The client sends the whole frame collection:

```json
{
  "frames": [
    {"frame_index": 0, "timestamp_seconds": 0.0,  "rgb": [255, 180, 60], "lab": [70.0, 25.0, 60.0],
     "detection_confidence": 0.91, "segmentation_confidence": 0.95, "flame_area_ratio": 0.18},
    {"frame_index": 1, "timestamp_seconds": 4.2,  "rgb": [166, 80, 27],  "lab": [43.8, 31.4, 43.0],
     "detection_confidence": 0.78, "segmentation_confidence": 0.91, "flame_area_ratio": 0.14}
  ]
}
```

Only `frame_index`, `rgb` and `lab` are required; the other measurements are
omitted rather than invented when absent. A video costs one Gemini call per
frame the client analysed with `/analyze`, plus **one** here for the entire
frame collection - never one per frame of this endpoint.

- **ONE request, ONE answer.** All frame evidence - plus collection-level mean
  RGB/LAB, per-channel LAB standard deviation, mean flame area, mean confidences
  and the observed time span - is assembled into a single text prompt. Gemini is
  **never** called per frame, per-frame answers are **never** averaged, and
  neither the first nor the highest-confidence frame's verdict is selected. The
  model is instructed to judge the collection as one fire event and to let the
  consistency, quality and amount of evidence across all frames drive its
  confidence. A list of per-frame assessments is rejected as malformed.
- **Numbers only.** `extra="forbid"` on the request bodies rejects any unknown
  field, so `image_base64`, `mask` or `filename` is a clean `422` rather than a
  silently dropped multi-megabyte payload. No image, mask or base64 ever reaches
  the prompt.
- **The same validation contract as the image path.** The result must name
  canonical dataset materials only, use ranks 1-5, return exactly 5 candidates
  with `confidence_percent` in `[0, 100]`, and set `primary_material` consistently
  with `uncertain`. Failures degrade to `{"available": false, "error": ...}`.
- **Confidence is never inflated.** `display_threshold_percent` (`45.0`) is the
  frontend AI-card visibility rule and is reported with the result so it lives in
  one place. It is **never** applied to the value: a `12.0` comes back as `12.0`.

```json
{
  "available": true,
  "primary_material": "Natural Fibers",
  "matches": [
    {"rank": 1, "material": "Natural Fibers", "confidence_percent": 72.0,
     "reason": "Consistent yellow-orange colour across all frames supports an organic fuel."},
    {"rank": 2, "material": "Paper Products(Wood material)", "confidence_percent": 64.0,
     "reason": "Also plausible; the frames do not separate paper from fibre."}
  ],
  "overall_confidence_level": "medium",
  "uncertain": false,
  "evidence_quality": "moderate",
  "reasoning_summary": "All sampled frames of this video show the same yellow-orange flame ...",
  "frames_supplied": 4,
  "frames_analyzed": 4,
  "display_threshold_percent": 45.0
}
```

`frames_supplied` vs. `frames_analyzed` reports how many frames carried usable
colour evidence. The response is Gemini-only: it carries no per-frame verdicts
and no fire class - the video fire class is the deterministic
`/material-identification` answer.

| Status | Code                | Meaning                                                       |
| ------ | ------------------- | ------------------------------------------------------------- |
| `200`  | –                   | analysed, **or** `available: false` when Gemini was unusable   |
| `422`  | `VALIDATION_ERROR`  | malformed body, out-of-range value, or an unknown field        |

Gemini failures are **not** HTTP errors. Disabled (`GEMINI_ENABLED=0`), a
missing `GEMINI_API_KEY`, exhausted quota, a timeout (`GEMINI_TIMEOUT_S`,
default 30 s), a missing `flame_dataset.json` or a payload that fails validation
all return `200` with `available: false` and a short `error` string - no
stack trace, no key, and the deterministic result is untouched.

### Reference UI

`frontend/index.html` is a minimal standalone page (no build step). Serve it
from the API origin, or open it behind any static server pointed at the API,
and it renders two separate cards: **Material Identification** (Powered by
FlameAnalyzer, the deterministic result) and **AI Material Analysis** (Powered
by Gemini, with the top match, confidence, top-5 list, overall confidence
level, reasoning summary and the "heuristic AI scores, not calibrated
probabilities" disclaimer). When Gemini is unavailable the AI card shows
"Currently unavailable" and the deterministic card keeps working.

## Configuration

Every parameter lives in `app/config.py` and can be overridden with environment
variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLAME_DET_MODEL` | `models/OBJ_best.pt` | detection weights |
| `FLAME_SEG_MODEL` | `models/SEG_best.pt` | segmentation weights |
| `FLAME_MODEL_DIR` | `models/` | directory holding both weights; the two variables above win |
| `FLAME_DATASET` | `data/flame_dataset.json` | material database |
| `FLAME_IMGSZ` | `640` | YOLO inference size |
| `FLAME_DET_CONF` | `0.4` | detection confidence threshold |
| `FLAME_SEG_CONF` | `0.4` | segmentation confidence threshold |
| `FLAME_DET_CLASS` | `0` | class index treated as fire |
| `FLAME_SEG_RETINA` | `1` | upscale masks to original resolution |
| `FLAME_BBOX_FALLBACK` | `1` | allow bounding-box mask fallback |
| `FLAME_N_CLUSTERS` | `2` | cluster count for `fixed` k, and the search lower bound |
| `FLAME_K_SELECTION` | `silhouette` | `silhouette` (search) or `fixed` (always `N_CLUSTERS`) |
| `FLAME_K_MAX` | `6` | upper bound searched by silhouette selection |
| `FLAME_K_CAP` | `4` | hard cap on the silhouette winner (`min(best_k, cap)`) |
| `FLAME_SILHOUETTE_SAMPLES` | `300` | samples used for the silhouette score |
| `FLAME_MAX_PIXELS` | `2000` | flame pixels sampled for clustering |
| `FLAME_MIN_PIXELS` | `10` | below this, only the mean colour is used |
| `FLAME_KMEANS_NINIT` | `10` | K-Means restarts for the final fit |
| `FLAME_GMM_MAX_ITER` | `100` | GMM EM iterations |
| `FLAME_BGGMM_MAX_ITER` | `50` | variational Bayesian GMM iterations |
| `FLAME_BGGMM_REG_COVAR` | `0.001` | Bayesian GMM variance floor (see note below) |
| `FLAME_DBSCAN_EPS` | `0.5` | DBSCAN radius in **standardised** units |
| `FLAME_DBSCAN_MIN_SAMPLES` | `3` | DBSCAN `min_samples` floor |
| `FLAME_DBSCAN_MIN_SAMPLES_DIV` | `50` | adaptive rule: `max(3, n // 50)` |
| `FLAME_EMIT_MASK` | `1` | include the base64 PNG mask in the response |
| `FLAME_MASK_PNG_COMPRESSION` | `6` | zlib level for the PNG encoder (0-9) |
| `FLAME_SIM_SCALE` | `200.0` | LAB distance that maps to similarity 0.0 |
| `FLAME_MAX_ALTERNATIVES` | `3` | alternative materials returned |
| `FLAME_DEVICE` | auto | e.g. `0` or `cpu`; unset means CUDA when available |
| `FLAME_SEED` | `42` | RNG seed for sampling and clustering |
| `GEMINI_ENABLED` | `0` | master switch for the secondary AI material analysis |
| `GEMINI_API_KEY` | – | Gemini API key; never hardcoded, logged or sent to the frontend |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | model used for the secondary analysis |
| `GEMINI_TIMEOUT_S` | `30` | hard timeout for one Gemini request |

Both video endpoints reuse these settings unchanged: the deterministic one needs
only `FLAME_DATASET`, `FLAME_SIM_SCALE` and `FLAME_MAX_ALTERNATIVES`, and the
video AI one uses the same `GEMINI_*` variables as `/analyze`. No video-specific
environment variable was added; `MAX_VIDEO_FRAMES` and
`VIDEO_DISPLAY_THRESHOLD_PERCENT` are fixed constants in
`app/video_material.py`.

### Parameters that differ from the original script

The clustering algorithms, their hyper-parameters and the k-selection procedure
are restored from `main.py`, with three deliberate exceptions. All three are
overridable, so nothing is locked in:

| Setting | Here | Original | Why |
| --- | --- | --- | --- |
| `FLAME_MAX_PIXELS` | `2000` | `1000` | Latency cap. `2000` gives silhouette search and DBSCAN more to work with; the real-image pipeline still runs in ~300 ms warm. |
| `FLAME_KMEANS_NINIT` | `10` | `3` | More restarts make the K-Means centroid more stable. Only affects the final fit; the k search already uses the original `n_init=3`. |
| `FLAME_GMM_MAX_ITER` | `100` | `50` | GMM occasionally had not converged at 50 on flame LAB tones. |

Two settings are **new**, not restorations:

- `FLAME_BGGMM_REG_COVAR` (`1e-3`) is a variance floor for
  `BayesianGMM`. scikit-learn's default (`1e-6`) raises on ill-defined
  covariances, which happens on tightly separated flame tones. The fit is
  retried with `1e2` and then `1e4` before it is given up on, and the
  `representative` field says which value was used. Real-image output is
  unchanged by it.
- `FLAME_K_SELECTION` lets you fall back to a fixed `k`, which the original
  pipeline had no equivalent for.

## Behaviour notes

* **LAB convention** - colours use the `scikit-image` convention (L\* in 0-100,
  D65), which is what `flame_dataset.json` was written in. OpenCV's
  `COLOR_BGR2Lab` scales L\* to 0-255 and must not be mixed with the database.
* **Clustering** - flame pixels are sampled, converted to LAB, standardised
  with `StandardScaler`, then run through **five** algorithms: K-Means, GMM,
  Bayesian GMM, DBSCAN and Agglomerative. `k` is chosen by silhouette score over
  `[n_clusters, k_max]` and capped at `k_cap`; set `FLAME_K_SELECTION=fixed` for
  a fixed `k`. Every algorithm reports its own representative colour, dominant
  cluster and full centroid list. **MeanShift is not used** and is no longer
  imported anywhere.
* **Determinism** - `FLAME_SEED` seeds both the pixel sampling and every
  estimator, so repeated runs on the same image give the same result.
* **Fire class is derived, not predicted.** `app/fire_classes.py` maps the
  matched material to a class - Classes A, B, C or D, or `Unclassified` when
  there is no evidence - with the full 18-material table spelled out in
  `MATERIAL_FIRE_CLASS` and documented precedence rules. The detection model
  predicts fire presence only; no claim is made that it predicts a class.
* **Material matching** - deterministic LAB nearest-reference matching. The
  highest-confidence detection and the highest-confidence mask are used, as in
  the original pipeline.
* **AI material analysis** - optional secondary Gemini opinion
  (`app/gemini_analysis.py`). It receives only the extracted mean RGB/LAB flame
  colour and the canonical vocabulary from `flame_dataset.json` - never the
  image - and is validated against that vocabulary. Its confidence percentages
  are heuristic scores, not calibrated probabilities, and any Gemini failure
  degrades to `ai_material_analysis: {available: false}` without affecting the
  deterministic result.
* **Video material analysis** (`app/video_material.py`) adds no new classifier.
  The deterministic half re-runs `match_material()` on the client's aggregated
  colour, so the video fire class and agents come from the same documented
  mapping as an image. The AI half builds **one** prompt from all frame evidence
  (mean colours, LAB spread, mean confidences, time span) and takes **one**
  consolidated answer - never per-frame calls, never an average of per-frame
  verdicts, never the first or highest-confidence frame. Frame counts up to
  `MAX_VIDEO_FRAMES` (120) and unknown request fields, including any image or
  base64 field, are rejected before Gemini is ever contacted.
* **Suppression data** - copied verbatim from `flame_dataset.json`; nothing is
  generated or invented. Agents are reported as dataset names with a mechanism
  category, and `compound` stays `null` because the dataset has no chemical
  identities to report.
* **Masks are returned** as a compact base64 PNG at the source resolution. No
  raw pixel arrays are ever serialised. Debug renderers live in
  `app/visualization.py` and are never called automatically.
* **Models are loaded once** and kept in memory; a single `FlameAnalyzer`
  instance expects sequential calls (Ultralytics predictors hold mutable
  state), so use one instance per worker/container.
