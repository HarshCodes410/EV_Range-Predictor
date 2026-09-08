# EV Range Predictor — Deployment Guide

You already know GitHub, so this whole path is: **push files → get a public link**. No servers, no installs beyond Python for local testing.

## Files in this folder
- `app.py` — the Streamlit app (sliders + prediction, rebuilt from your notebook's pipeline)
- `requirements.txt` — Python packages Streamlit Cloud will auto-install
- `ev_data.xls` — **you must add this yourself** (your training data). Put it in the same folder.

## Step 1 — Test locally (optional but recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app at `http://localhost:8501`. Confirm sliders move and the prediction updates.

## Step 2 — Push to GitHub
1. Create a new GitHub repo (e.g. `ev-range-predictor`).
2. Add these 3 files to it: `app.py`, `requirements.txt`, and your `ev_data.xls`.
3. Commit and push — exactly like any other repo you've made.

## Step 3 — Deploy on Streamlit Community Cloud (free)
1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Pick your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.

In about a minute you'll get a public URL like:
`https://ev-range-predictor-yourname.streamlit.app`

That's the link you paste into your PPT — it works on any PC, no installation needed by the viewer.

## Step 4 — Updating the app later
Whenever you want to change the code:
1. Edit `app.py` locally (or directly in GitHub's web editor).
2. Commit and push.
3. Streamlit Cloud detects the push and **auto-redeploys** — same link, updated app, usually live within a minute. Your old link never breaks.

## Notes
- The model retrains once per app restart (Streamlit caches it), not on every slider move — so it stays responsive.
- The sidebar slider ranges are pulled directly from your dataset's 1st–99th percentiles, so bounds stay realistic.
- If you add new columns to your dataset later, you'll need to add a matching slider/dropdown in `app.py` — I'm happy to help with that whenever you're ready.
