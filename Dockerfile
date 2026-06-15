FROM python:3.12-slim

LABEL org.opencontainers.image.title="Marketing Attribution & Budget Optimization"
LABEL org.opencontainers.image.description="MMM + multi-touch attribution + budget optimization pipeline"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
