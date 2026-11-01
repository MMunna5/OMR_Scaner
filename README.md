# OMR Scanner (Shaheed Noor Memorial School & College sheet)

Photo / PDF / live camera diye OMR sheet scan kore auto count, roll number read ar Excel export.

## Files
- `app.py` – Flask server (API + page)
- `omr.py` – OMR engine (OpenCV): sheet align, bubble read, roll read
- `static/index.html` – user interface
- `requirements.txt`, `render.yaml` – Render deploy config

## Local-e chalano (optional test)
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

## Render-e deploy (step by step)
1. Ei project already GitHub-e push kora ache: [MMunna5/OMR_Scaner](https://github.com/MMunna5/OMR_Scaner).
2. Render.com-e GitHub diye login koro → **New + → Web Service** → `OMR_Scaner` repo select koro.
3. Settings:
   - Runtime: **Python 3**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --workers 1 --threads 2 --timeout 300 --bind 0.0.0.0:$PORT`
   - Instance type: **Free**
   - Environment variable: `PYTHON_VERSION` = `3.11.9`
   (Ba **New + → Blueprint** chaple `render.yaml` theke automatic hoye jabe.)
4. **Create Web Service** → 3-5 minute build hobe. Shesh hole `https://omr-scanner-xxxx.onrender.com` link pabe. Ei link-i shobaike dao.
5. Code bodle: GitHub-e file replace/commit korlei Render auto redeploy kore.

## Notes
- Free plan-e 15 min kew na use korle server ghumiye jay; prothom open-e ~1 minute lage.
- Camera shudhu `https` link-e chole (Render-er link https, tai thik ache).
- Eta shudhu ei specific sheet-er jonno calibrated (top-er kalo box, niche-r tick mark, 50 ta question block, 6 digit roll).
- Bhalo result-er jonno: pura sheet frame-e, bhalo alo, sidha kore, shadow chhara photo. PDF-e ekbare ~30 page porjonto rakho.
- Ballpoint/pencil dui-i kaj kore; halka mark hole "Overlay dekho" diye verify koro.
