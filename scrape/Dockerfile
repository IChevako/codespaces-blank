FROM python:3.11

# Install system libraries required by Python and Playwright
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libxshmfence-dev \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Python libraries
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g playwright

# Install Python libraries
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies
RUN pip install python-multipart

# Copy application code
COPY . /app

# Make sure to include the service account JSON in the directory where Docker build is run and reference it correctly
COPY robotwriter-3c7744b0d983.json /app/robotwriter-3c7744b0d983.json

# Set up Playwright
RUN npx playwright install

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

