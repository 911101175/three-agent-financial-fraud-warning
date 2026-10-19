FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py index.html rule_model.py workpaper_generator.py three_agents_three_level.py ./
COPY 汇总数据.csv ./汇总数据.csv
COPY outputs/rule_model/threshold_table.csv ./outputs/rule_model/threshold_table.csv
COPY outputs/rule_model/ml_features.csv ./outputs/rule_model/ml_features.csv
COPY outputs/rule_model/sample_id_map_internal.csv ./outputs/rule_model/sample_id_map_internal.csv
COPY outputs/ml_model/best_model.pkl outputs/ml_model/model_metadata.json ./outputs/ml_model/

EXPOSE 5001
CMD ["python", "app.py"]
