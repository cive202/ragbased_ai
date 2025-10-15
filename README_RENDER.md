Render deployment steps

1. Push this repository to GitHub (or connect Git provider) so Render can access it.
   Example commands to push:

```powershell
git add .
git commit -m "Prepare app for Render deployment"
git push origin master
```

If `embeddings.joblib` is large, do NOT push it directly. Use Git LFS or upload it to object storage and set `EMBEDDINGS_URL`.

2. On Render dashboard, create a new "Web Service".
   - Connect the Git repo and select the branch (e.g., master).
   - Runtime: Python (uses `runtime.txt` if present).
   - Build Command: pip install -r requirements_prod.txt
   - Start Command: gunicorn main:app --bind 0.0.0.0:$PORT --workers 2

3. Add environment variables in Render settings:
   - COHERE_API_KEY: <your-cohere-api-key>
   - (Optional) FLASK_DEBUG: true (for debugging) or leave unset/false in production

4. Files included for Render:
   - `Procfile` — start command for other hosts
   - `runtime.txt` — Python version (3.11.4)
   - `requirements_prod.txt` — minimal requirements for production

5. Notes and troubleshooting
   - Ensure `embeddings.joblib` exists in the repository or create it before deploy. If it's large, consider storing it in object storage (S3) and change `main.py` to load from a URL.
   - To avoid committing a large `embeddings.joblib`, upload it to a public/private object store (S3, DigitalOcean Spaces, or Render Static Files). Then set the `EMBEDDINGS_URL` environment variable in Render to the file URL. The app will download and load it at startup.
    Example: set `EMBEDDINGS_URL` to `https://your-bucket.s3.amazonaws.com/embeddings.joblib`
   - If the app fails with missing packages, update `requirements_prod.txt` and redeploy.
  - To host embeddings externally: upload `embeddings.joblib` to S3/Spaces and set `EMBEDDINGS_URL` in Render settings.
   - Health check endpoint: `/health`

6. Local testing
   - Set COHERE_API_KEY locally and run:

```powershell
$env:COHERE_API_KEY = 'your_key_here'; python main.py
```

Or using gunicorn locally:

```powershell
pip install -r requirements_prod.txt; gunicorn main:app --bind 0.0.0.0:5000
```
