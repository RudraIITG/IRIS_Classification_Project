# Base image with Python
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (app.py, templates, static, etc.)
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the application
CMD ["python", "start.py"]
