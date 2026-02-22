# Makefile for FraudOps (Windows/PowerShell)

# Variables
VENV = fraud_env
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip
DOCKER = docker-compose
PREFECT = $(VENV)/Scripts/prefect

# CRITICAL: This tells Make that these are commands, not folder names
.PHONY: setup infra-up infra-down train stream predict web flow flow-ui clean help

# Default target
help:
	@echo "----------------------------------------------------------------"
	@echo "🛡️  FRAUDOPS PROJECT MANAGER"
	@echo "----------------------------------------------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "  setup       : Create venv and install dependencies"
	@echo "  infra-up    : Start Kafka and Zookeeper (Docker)"
	@echo "  infra-down  : Stop Docker containers"
	@echo "  train       : Train the PyTorch model"
	@echo "  stream      : Start generating fake transactions"
	@echo "  predict     : Start the terminal-based predictor"
	@echo "  web         : Launch the Streamlit Web Dashboard"
	@echo "  clean       : Remove pycache and artifacts"
	@echo "  flow        : Run the automated training pipeline"
	@echo "  flow-ui     : Start the Prefect Dashboard Server"
	@echo "----------------------------------------------------------------"

# 1. Environment Setup
setup:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PYTHON) -m pip install kafka-python-ng mlflow pyyaml scikit-learn streamlit pandas prefect

# 2. Infrastructure (Docker)
infra-up:
	$(DOCKER) up -d

infra-down:
	$(DOCKER) down

# 3. Pipeline Commands
train:
	$(PYTHON) main.py train

stream:
	$(PYTHON) main.py stream

predict:
	$(PYTHON) main.py predict

web:
	$(PYTHON) main.py web

# 4. Orchestration (Prefect)
# Note: We use '-m src.orchestrate' to fix the ModuleNotFoundError
flow:
	$(PYTHON) -m src.orchestrate

flow-ui:
	$(PREFECT) server start

# 5. Utilities
clean:
	Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Force -Recurse
	Get-ChildItem -Path . -Include *.pyc -Recurse | Remove-Item -Force -Recurse