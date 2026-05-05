# Run Instructions

These commands assume you are in:

```powershell
c:\Users\Admin\Desktop\Ashly\Personal\Datas Review
```

## 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 2. Generate the extra 10,000 training rows

```powershell
python -m src.synthetic_data_generator
```

This creates:

```text
Datas/video_engagement_fraud_dataset_generated_10000.csv
Datas/video_engagement_graph_timeseries_generated_10000.csv
```

## 3. Train the model

```powershell
python -m src.train_model
```

This creates:

```text
models/fraud_model.pkl
models/training_metrics.json
```

## 4. Run tests

```powershell
pytest
```

## 5. Run one sample prediction

```powershell
python -m src.predict
```

## 6. Start the FastAPI server

```powershell
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

## 7. Start the Streamlit dashboard

```powershell
streamlit run app/streamlit_app.py
```

Streamlit will print the local dashboard URL.

## 8. Analyze a dashboard screenshot

```powershell
python -m src.screenshot_analyzer "C:\path\to\dashboard_screenshot.png" TikTok
```

This uses OCR for visible metrics and computer vision for graph shape, then runs the extracted values through the fraud model. If OCR is not configured, the command still returns a safe result with missing fields.

## Notes

- The original files in `Datas/` are not overwritten.
- Generated data is synthetic and only supplements model training.
- If OCR is not installed on your computer, `src/screenshot_analyzer.py` will still return a safe result with missing fields.
