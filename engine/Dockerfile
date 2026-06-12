FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY wazn.py .
COPY hyperbolic.py .
COPY all_99_corrected.json .
COPY poincare_data.json .
COPY poincare_data_v3.json .
COPY name_vecs_v3.npz .
COPY ui.html .
COPY privacy.html .
COPY schema.json .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
