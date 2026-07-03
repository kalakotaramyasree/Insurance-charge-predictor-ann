# Deploying the Insurance Charges ANN

## What's in this package

| File | Purpose |
|---|---|
| `train.py` | Reproduces both your notebooks in one script; outputs the 3 artifacts below |
| `app.py` | Streamlit web app (interactive UI) |
| `api.py` | Flask REST API (for programmatic/JSON access) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build for either app |

You only need **one** of `app.py` / `api.py` depending on whether you want a UI or an API — you can also run both.

## Step 1 — Generate the model artifacts

Your notebooks already save these three files when run start to finish:
- `preprocessor.pkl`
- `y_scaler.pkl`
- `insurance_ann_model.keras`

Easiest path: **just re-run your two notebooks** and copy those 3 output files into this folder.

Alternative: run the provided script directly on your `insurance.csv`:
```bash
pip install -r requirements.txt
python train.py --data insurance.csv
```

Either way, end up with `preprocessor.pkl`, `y_scaler.pkl`, and `insurance_ann_model.keras` sitting next to `app.py`/`api.py`.

## Step 2 — Run locally to verify

**Streamlit (web UI):**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

**Flask (REST API):**
```bash
pip install -r requirements.txt
python api.py
```
Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":30,"sex":"male","bmi":25.0,"children":0,"smoker":"no","region":"northeast"}'
```

## Step 3 — Deploy publicly

Pick whichever fits your comfort level:

### Option A: Streamlit Community Cloud (free, easiest, UI only)
1. Push this folder (including the 3 artifact files) to a public GitHub repo.
2. Go to https://share.streamlit.io → "New app" → point it at your repo, main file `app.py`.
3. It installs `requirements.txt` automatically and gives you a public URL.

### Option B: Render / Railway (free tier, works for both app.py and api.py)
1. Push the folder to GitHub.
2. On Render.com: "New Web Service" → connect repo.
   - Build command: `pip install -r requirements.txt`
   - Start command (Streamlit): `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Start command (Flask): `gunicorn api:app`
3. Deploy — you get a public HTTPS URL.

### Option C: Docker (any cloud host — AWS/GCP/Azure/DigitalOcean)
```bash
docker build -t insurance-ann .
docker run -p 8501:8501 insurance-ann
```
Push the image to a registry and deploy on any container host (Cloud Run, ECS, App Runner, etc.).

### Option D: Hugging Face Spaces (free, popular for ML demos)
1. Create a new Space, SDK = Streamlit.
2. Upload `app.py`, `requirements.txt`, and the 3 artifact files.
3. Space builds and serves automatically.

## Notes / gotchas

- **Model file size**: Keras ANN files are usually small (KBs–few MB) for a model this size, so free-tier hosts are fine.
- **TensorFlow install size**: `tensorflow` is a heavy dependency (~500MB). If deploying to a size-constrained free tier, swap it for `tensorflow-cpu` in `requirements.txt`.
- **Don't re-fit the preprocessor at inference time** — always load the saved `preprocessor.pkl`/`y_scaler.pkl` so incoming requests are transformed exactly like training data (this is what `app.py`/`api.py` already do).
- **Column names/order**: the app builds a DataFrame with columns `age, sex, bmi, children, smoker, region` — matches what `ColumnTransformer` was fitted on, so order doesn't matter, only names do.
