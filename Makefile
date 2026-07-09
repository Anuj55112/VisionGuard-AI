.PHONY: setup lint format test run-api run-ui export-onnx clean help

PYTHON = python3
PIP = pip

help:
	@echo "VisionGuard AI Make Command List:"
	@echo "  setup       Install dependencies"
	@echo "  lint        Run linting and static checks (ruff, mypy)"
	@echo "  format      Auto-format code using ruff"
	@echo "  test        Run unit tests with coverage"
	@echo "  run-api     Start the FastAPI backend server"
	@echo "  run-ui      Start the Streamlit frontend"
	@echo "  export-onnx Export models to ONNX format"
	@echo "  clean       Remove cache and build artifacts"

setup:
	$(PYTHON) -m $(PIP) install --upgrade pip
	$(PYTHON) -m $(PIP) install -r requirements.txt

lint:
	ruff check .
	mypy src app tests

format:
	ruff format .

test:
	pytest tests/

run-api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run app/ui.py --server.port 8501

export-onnx:
	$(PYTHON) src/utils/onnx_export.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
