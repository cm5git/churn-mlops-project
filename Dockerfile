# Small official Python image. Same minor version as local development and CI.
FROM python:3.10-slim

# Don't write .pyc files; send logs straight to the terminal instead of buffering.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install dependencies first. Docker caches each step, so this slow step is
#    only re-run when requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy only what the API needs at runtime: the app and the trained model.
COPY app.py .
COPY model/ model/

# 3. Run as an ordinary user rather than root.
RUN useradd --create-home appuser
USER appuser

# Documentation only: the port the API listens on.
EXPOSE 8000

# Listen on 0.0.0.0 so the port can be reached from outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]