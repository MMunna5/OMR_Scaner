# OMR Scanner (Shaheed Noor Memorial School & College answer sheet)

Scan an OMR sheet from a photo, PDF, or live camera capture, automatically count answers,
read roll numbers, and export the results to Excel.

## Files
- `app.py` – Flask server (API + page)
- `omr.py` – OMR engine (OpenCV): sheet align, bubble read, roll read
- `static/index.html` – user interface
- `requirements.txt`, `render.yaml` – Render deploy config

## Run locally (optional)
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

## Deploy to Render
1. This project is available on GitHub: [MMunna5/OMR_Scaner](https://github.com/MMunna5/OMR_Scaner).
2. Log in to Render.com with GitHub → **New + → Web Service** → select the `OMR_Scaner` repository.
3. Settings:
   - Runtime: **Python 3**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --workers 1 --threads 2 --timeout 300 --bind 0.0.0.0:$PORT`
   - Instance type: **Free**
   - Environment variable: `PYTHON_VERSION` = `3.11.9`
   (Alternatively, select **New + → Blueprint** to configure the service automatically from `render.yaml`.)
4. Select **Create Web Service**. The build usually takes 3-5 minutes. When it finishes, you will receive a URL such as `https://omr-scanner-xxxx.onrender.com`.
5. To deploy code changes, push or commit the updated files to GitHub. Render will redeploy automatically.

## Notes
- On the free plan, the server sleeps after 15 minutes of inactivity; the first request may take about one minute.
- The camera works only over an `https` connection. The Render URL uses HTTPS.
- The scanner is calibrated only for this specific sheet (top black boxes, bottom tick marks, 50-question layout, and six-digit roll number).
- You can choose any whole-number question count from 1 to 50. Counts above 50 are not supported because the sheet layout is calibrated for a maximum of 50 questions.
- A portrait answer sheet without the calibrated anchor marks is also supported through the fallback layout detector. The current fallback profile supports the common 30-question, two-block format; its roll number is left for manual correction.
- Portrait fallback processing uses the detected page aspect ratio and dynamically fitted bubble rows/columns rather than a fixed image size, so rotated or differently cropped camera photos can be normalized before reading.
- The standard final sheet measures the four answer bubbles in each answer block independently after page rectification. Question-number text columns are excluded, and the six roll-number columns are fitted separately from their ten digit rows.
- Answer marking accepts either a dark bubble or a dark four-corner square fill when at least 25% of the inner mark area is filled. The 50-question geometry is fitted once per uploaded page and reused for all rows, including partially filled marks.
- For best results, keep the entire sheet in frame, use good lighting, hold it straight, and avoid shadows. Keep PDFs to about 30 pages per upload.
- Both ballpoint and pencil marks are supported. Verify faint marks using **View overlay**.
