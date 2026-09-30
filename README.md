# FlameAnalyzer - inference backend

Headless, cloud-ready flame analysis: **detection -> segmentation -> flame
colour (CIELAB) -> material matching -> JSON**.

There is no GUI, no browser UI, no tunnel and no LLM call anywhere in the
request path. The ML pipeline is transport-agnostic; a thin FastAPI layer
(`app/api.py`) and a CLI (`app/cli.py`) sit on top of it.

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
│   ├── api.py                # FastAPI app: POST /analyze, GET /health, GET /
│   ├── cli.py                # command line runner (prints JSON)
│   ├── color_analysis.py     # flame pixels -> LAB -> K-Means / GMM representative colour
│   ├── config.py             # Settings: all inference parameters in one place
│   ├── detection.py          # OBJ_best.pt fire detection
│   ├── errors.py             # AnalysisError hierarchy + error codes
│   ├── imaging.py            # image validation / bytes -> BGR decoding
│   ├── material_matching.py  # LAB matching against flame_dataset.json
│   ├── schemas.py            # response dataclasses + JSON-safe conversion
│   ├── segmentation.py       # SEG_best.pt flame segmentation + bbox fallback
│   └── visualization.py      # optional debug renderers (never called by the API)
├── models/
│   ├── OBJ_best.pt           # custom fire detection model (do not replace)
│   └── SEG_best.pt           # custom flame segmentation model (do not replace)
├── data/
│   └── flame_dataset.json    # material database (reference flame colours + suppression data)
├── tests/
│   ├── test_pipeline.py      # pipeline tests (stub models; runs without the weights)
│   └── test_api.py           # FastAPI TestClient tests (stubbed analyzer)
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

Run the test suite (50 tests: pipeline + API):

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

```json
{
  "success": true,
  "fire_detection": {
    "detected": true,
    "confidence": 0.5149,
    "bounding_box": {"x1": 321, "y1": 99, "x2": 349, "y2": 145}
  },
  "segmentation": {
    "available": false,
    "fallback_used": true,
    "flame_pixel_count": 1288,
    "mask_area_ratio": 0.006941,
    "confidence": null
  },
  "flame_analysis": {
    "kmeans": {"rgb": [224, 193, 190], "lab": [80.62, 10.34, 5.85], "cluster_count": 2,
               "samples_used": 1288, "pixels_sampled": false},
    "gmm": {"rgb": [237, 217, 214], "lab": [88.24, 6.28, 4.2], "cluster_count": 2,
            "samples_used": 1288, "pixels_sampled": false},
    "mean_color": {"rgb": [237, 217, 214], "lab": [88.24, 6.28, 4.2]},
    "flame_pixel_count": 1288,
    "samples_used": 1288,
    "pixels_sampled": false
  },
  "material_analysis": {
    "primary_material": "Alcohol-Based Products",
    "similarity": 0.8846,
    "alternatives": [
      {"material": "Spray Products", "similarity": 0.8469},
      {"material": "Electrical Components", "similarity": 0.7224}
    ],
    "database_notes": "...",
    "score_basis": "mean LAB distance to flame_dataset.json reference colours"
  },
  "suppression_information": {
    "source": "flame_dataset.json",
    "material": "Alcohol-Based Products",
    "methods": ["CO2", "Water spray", "Foam"],
    "database_notes": "..."
  },
  "timing": {"total_ms": 83.2, "detection_ms": 41.0, "segmentation_ms": 22.0,
             "color_ms": 19.0, "material_ms": 0.9},
  "error": null
}
```

`similarity` is a **distance-based score in [0, 1]**, not a calibrated
probability - it is named accordingly and must not be presented as a confidence.

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

## Configuration

Every parameter lives in `app/config.py` and can be overridden with environment
variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLAME_DET_MODEL` | `models/OBJ_best.pt` | detection weights |
| `FLAME_SEG_MODEL` | `models/SEG_best.pt` | segmentation weights |
| `FLAME_DATASET` | `data/flame_dataset.json` | material database |
| `FLAME_IMGSZ` | `640` | YOLO inference size |
| `FLAME_DET_CONF` | `0.4` | detection confidence threshold |
| `FLAME_SEG_CONF` | `0.4` | segmentation confidence threshold |
| `FLAME_DET_CLASS` | `0` | class index treated as fire |
| `FLAME_SEG_RETINA` | `1` | upscale masks to original resolution |
| `FLAME_BBOX_FALLBACK` | `1` | allow bounding-box mask fallback |
| `FLAME_N_CLUSTERS` | `2` | fixed cluster count |
| `FLAME_MAX_PIXELS` | `2000` | flame pixels sampled for clustering |
| `FLAME_MIN_PIXELS` | `10` | below this, only the mean colour is used |
| `FLAME_SIM_SCALE` | `200.0` | LAB distance that maps to similarity 0.0 |
| `FLAME_MAX_ALTERNATIVES` | `3` | alternative materials returned |
| `FLAME_DEVICE` | auto | e.g. `0` or `cpu`; unset means CUDA when available |
| `FLAME_SEED` | `42` | RNG seed for sampling and clustering |

## Behaviour notes

* **LAB convention** - colours use the `scikit-image` convention (L\* in 0-100,
  D65), which is what `flame_dataset.json` was written in. OpenCV's
  `COLOR_BGR2Lab` scales L\* to 0-255 and must not be mixed with the database.
* **Clustering** - a fixed `k=2` for K-Means and GMM plus a mean-colour
  fallback. No silhouette search, no DBSCAN/MeanShift/Agglomerative/VB-GMM.
* **Material matching** - deterministic LAB nearest-reference matching. The
  highest-confidence detection and the highest-confidence mask are used, as in
  the original pipeline.
* **Suppression data** - copied verbatim from `flame_dataset.json`; nothing is
  generated or invented.
* **No image payloads** - masks and flame pixels are never returned. Debug
  renderers live in `app/visualization.py` and are never called automatically.
* **Models are loaded once** and kept in memory; a single `FlameAnalyzer`
  instance expects sequential calls (Ultralytics predictors hold mutable
  state), so use one instance per worker/container.
