# Build stage
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS builder

# Set the working directory in the container
WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    curl \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy only requirements
COPY pyproject.toml poetry.lock ./

# Configure Poetry to use the virtual environment
RUN poetry config virtualenvs.create false

# Install project dependencies with retries
RUN for i in 1 2 3 4 5; do poetry install --no-interaction --no-ansi && break || sleep 5; done

# Copy the rest of the project files
COPY . .

# Install project dependencies with retries
RUN for i in 1 2 3 4 5; do poetry install --no-interaction --no-ansi && break || sleep 5; done


# Runtime stage
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy project files from builder stage
COPY --from=builder /app /app

ENTRYPOINT ["/bin/bash"]