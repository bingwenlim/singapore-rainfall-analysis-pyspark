"""Project configuration — copy this file to config.py and fill in your values.

    cp config.example.py config.py

config.py is gitignored, so your real project/bucket IDs stay out of the repo.
Scripts and the notebook import from config.py, so anyone can run them against
their own GCP project and bucket by editing only their copy.
"""

# GCP project that owns the bucket and runs Spark jobs.
PROJECT = "YOUR_PROJECT"

# Region for the bucket and compute. Keep them in the same region.
REGION = "asia-southeast1"

# GCS bucket name (no gs:// prefix). Must be globally unique.
BUCKET = "YOUR_BUCKET"

# Spark Connect endpoint for interactive cloud analysis (e.g. an SSH tunnel).
CONNECT_URL = "sc://localhost:15002"
