# Build stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Copy only pyproject.toml and poetry.lock (if it exists)
COPY pyproject.toml poetry.lock* ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the project files
COPY . .

# Runtime stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS runtime

# Set the working directory in the container
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

# Copy the project files from builder stage
COPY --from=builder /app /app

# Make port 80 available to the world outside this container
# Adjust if your application uses a different port
EXPOSE 80

# Define environment variable
ENV NAME Ikepono

# Run the application
CMD ["python", "src/main.py"]