FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY earth_ops/ earth_ops/

RUN useradd --create-home operator && mkdir -p /app/data && chown operator /app/data
USER operator

# Products, outputs and KPIs live in /app/data: mount it as a volume
ENTRYPOINT ["python", "-m", "earth_ops"]
CMD ["--help"]
