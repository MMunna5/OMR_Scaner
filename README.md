# OMR Scanner (Shaheed Noor Memorial School & College answer sheet)

Scan an OMR sheet from a photo, PDF, or live camera capture, automatically count answers,
read roll numbers, and export the results to Excel.

## Files
- `app.py` – Flask server (API + page)
- `omr.py` – OMR engine (OpenCV): sheet alignment, bubble reading, roll reading, and quarter-turn photo correction
- `omr_patch.py` – compatibility handling for the alternate 30-question portrait layout and multi-answer keys
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
- The school template is aligned from the top red registration boxes and bottom red bars. If those anchors cannot be detected, the scanner stops instead of guessing a different layout and silently returning incorrect answers.
- After alignment, the Q1–25 and Q26–50 answer sections are treated as separate four-corner quadrilaterals. Every row and option position is interpolated from its own box; Hough-circle detection is not used, preventing detected circle centres from drifting vertically.
- The six-column roll-number box is independently mapped from its four corners into six columns × ten digit rows. Each column is read automatically; a six-digit result is shown as auto-detected, while blank or multiple marks are flagged rather than guessed.
- Answer and roll marks are measured inside rectangular cell regions, excluding the printed red outline. A mark is accepted when at least 15% of the inner cell contains dark ink in the red channel, which suppresses the red printed Bengali letters and bubble borders.
- The answer key supports multiple correct options. A student answer is correct when exactly one option is marked and it is one of the configured correct options; multiple student marks are wrong. Blank answers are displayed as blank but included in the wrong count. Selecting all four key options makes any single student option correct.
- The calibrated 50-question layout now tries all four 90-degree camera orientations before alignment, so sideways phone photos can be normalized automatically.
- Overlay highlights the student's actual selected bubbles, including every mark in a double-marked question, and supports multiple accepted answers in the key.
- Regression tests run with GitHub Actions on pushes and pull requests (`python -m unittest discover -s tests -v`).
- For best results, keep the entire sheet in frame, use good lighting, hold it straight, and avoid shadows. Keep PDFs to about 30 pages per upload.
- Both ballpoint and pencil marks are supported. Verify faint marks using **View overlay**.
