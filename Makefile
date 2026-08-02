.PHONY: setup test cov lint format typecheck security integration run inspect serve docker-build docker-up docker-down fly-deploy clean

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pre-commit install
	@cp -n .env.example .env 2>/dev/null && echo "Created .env — fill in GOOGLE_API_KEY" || echo ".env already exists"

# ── Quality ──────────────────────────────────────────────────────────────────
test:
	pytest --tb=short -q

cov:
	pytest --cov=src --cov-report=term-missing -q

# Usage: make integration IMAGE=path/to/photo.jpg [LAT=12.95] [LNG=77.58]
integration:
ifndef IMAGE
	$(error IMAGE is required — e.g. make integration IMAGE=photo.jpg LAT=12.95 LNG=77.58)
endif
	TEST_IMAGE_PATH=$(IMAGE) \
		$(if $(LAT),TEST_LAT=$(LAT)) \
		$(if $(LNG),TEST_LNG=$(LNG)) \
		pytest -m integration -v -s

lint:
	ruff format --check src/ tests/
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

typecheck:
	mypy src/

security:
	pip-audit --requirement requirements.txt

# ── Run ──────────────────────────────────────────────────────────────────────
# Usage: make run IMAGE=path/to/photo.jpg LAT=12.95 LNG=77.58
run:
ifndef IMAGE
	$(error IMAGE is required — e.g. make run IMAGE=photo.jpg LAT=12.95 LNG=77.58)
endif
	python3 -m src.main --image $(IMAGE) \
		$(if $(LAT),--lat $(LAT)) \
		$(if $(LNG),--lng $(LNG)) \
		$(if $(USER_ID),--user-id $(USER_ID))

inspect:
	streamlit run src/inspector.py

serve:
	uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t lens-os-api .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

fly-deploy:
	fly deploy

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .coverage .pytest_cache htmlcov
