FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/*.py ./
COPY engine/ui.html ./
COPY data/paper_b/basis_99_v3.json ./all_99_corrected.json
COPY data/paper_b/poincare_data_v3.json ./poincare_data_v3.json

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
