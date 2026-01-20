# fast and light python image
FROM python:3.11-slim
WORKDIR /app
# Copy the requirements
COPY requirements.txt .
# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt
# Copy the code
COPY . .
# Launch ChromaDB server
CMD ["chroma", "run", "--host", "0.0.0.0", "--port", "8000"]