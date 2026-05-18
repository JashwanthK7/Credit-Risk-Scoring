# 1. Use an official, lightweight Python image as the base operating system
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system dependencies (LightGBM/XGBoost sometimes need these on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file into the container
COPY requirements.txt .

# 5. Install Python dependencies (adding the API packages we just used)
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn \
    pydantic

# 6. Copy the rest of your application code and the saved model artifact
COPY config/ ./config/
COPY src/ ./src/
COPY artifacts/best_model.pkl ./artifacts/
COPY serve_api.py .

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to start the server when the container boots
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]