FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agi_agent/ /app/agi_agent/

EXPOSE 8090

ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app

CMD ["python", "-m", "agi_agent.webui.api_server"]
