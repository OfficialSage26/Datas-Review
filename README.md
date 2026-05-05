# AI Video Fraud Detection System

This project reviews creator reward submissions for suspicious engagement patterns. It loads the existing datasets from `Datas/`, applies review rules from `Guides/`, engineers fraud signals, trains a rule-based baseline and a machine learning classifier, and exposes predictions through Python, FastAPI, and Streamlit.

The system is for moderation, campaign review, and budget protection. It does not create botting, fake engagement, autoclicking, autoswiping, or bypass tooling.

## Folder Structure

```text
Datas/
  video_engagement_fraud_dataset.csv
  video_engagement_fraud_dataset.xlsx
  video_engagement_graph_timeseries.csv
  video_engagement_fraud_dataset_generated_10000.csv
  video_engagement_graph_timeseries_generated_10000.csv
Guides/
  How to Check Botted Videos--Complete Guide.pdf
  How to Check Video Analytics .pdf
  Whop Videos Review Guidelines.pdf
models/
  fraud_model.pkl
  training_metrics.json
src/
  data_loader.py
  feature_engineering.py
  synthetic_data_generator.py
  train_model.py
  predict.py
  risk_scoring.py
  explanation_generator.py
  api.py
  screenshot_analyzer.py
  utils.py
app/
  streamlit_app.py
streamlit_app.py
tests/
  test_prediction.py
```

## Data

The original CSV has 300 labeled submissions with labels `clean`, `suspicious`, and `botted`. The graph time-series file has 3,900 rows and links to the main dataset by `submission_id`.

Because 300 rows is small for a stable model, the project generates 10,000 additional synthetic training submissions and matching graph histories. Generated data is saved separately and does not overwrite original files. Synthetic rows are useful for prototyping, not production validation.

## Train The Model

```powershell
python -m pip install -r requirements.txt
python -m src.synthetic_data_generator
python -m src.train_model
```

Training saves:

```text
models/fraud_model.pkl
models/training_metrics.json
```

The training script prints accuracy, precision, recall, F1 score, confusion matrix, rule-based baseline metrics, and feature importance.

## Run A Prediction

```powershell
python -m src.predict
```

The output includes decision recommendation, fraud risk score, risk level, predicted class, calculated ratios, graph analysis, suspicious signals, likely fraud types, reviewer reasoning, missing evidence, creator-facing reason, and final verdict.

## Run The API

```powershell
uvicorn src.api:app --reload
```

Endpoints:

- `GET /`
- `POST /predict`
- `POST /predict-batch`
- `POST /retrain`

Example request:

```json
{
  "platform": "TikTok",
  "views": 102146,
  "likes": 105,
  "comments": 0,
  "shares": 75,
  "graph_pattern": "flat_then_spike",
  "traffic_source": null,
  "watch_time": null,
  "audience_location": null
}
```

## Run The Dashboard

```powershell
streamlit run streamlit_app.py
```

The dashboard supports screenshot upload, manual metric entry, CSV upload, batch predictions, risk scores, graph analysis, suspicious signals, creator-facing reasons, and feature importance.

The screenshot review tab is strict and moderator-facing. Its first output is always one of:

```text
Decision: Approved
Decision: Reject: Send Full Analytics
```

Upload a Whop/TikTok-style analytics screenshot to extract visible metrics and graph shape, then run the extracted evidence through the model. If anything is missing, unclear, suspicious, or not fully proven, the screenshot review returns `Reject: Send Full Analytics`.

For Streamlit Community Cloud, use:

```text
Repository: OfficialSage26/Datas-Review
Branch: main
Main file path: streamlit_app.py
```

## Analyze A Dashboard Screenshot

`src/screenshot_analyzer.py` can extract visible metrics with OCR, classify the visible graph shape with computer vision, and send the extracted fields into the same fraud prediction model.

```powershell
python -m src.screenshot_analyzer "C:\path\to\dashboard_screenshot.png" TikTok
```

OCR requires the local Tesseract executable to be installed and available to `pytesseract`. If OCR or OpenCV fails, the module returns missing fields safely instead of crashing.

## Adding Real Reviewed Submissions Later

Add new reviewed rows to the original schema or a new CSV with the same key fields. Keep labels as:

- `clean`
- `suspicious`
- `botted`

Important fields include `submission_id`, `views`, `likes`, `comments`, `shares`, `graph_pattern`, `label`, and any available analytics such as retention, traffic sources, geography, and account history. Retrain after adding reviewed examples:

```powershell
python -m src.train_model
```
