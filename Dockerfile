FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 5000

# No shell variable expansion required: app.py reads PORT/HOST from the
# environment via pydantic settings, so this works with or without a shell.
CMD ["python", "app.py"]
