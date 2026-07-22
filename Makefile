.PHONY: setup test cov lint format typecheck run inspect clean

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@cp -n .env.example .env 2>/dev/null && echo "Created .env — fill in GOOGLE_API_KEY" || echo ".env already exists"

# ── Quality ──────────────────────────────────────────────────────────────────
test:
	pytest --tb=short -q

cov:
	pytest --cov=src --cov-report=term-missing -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

typecheck:
	mypy src/ --ignore-missing-imports

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

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .coverage .pytest_cache htmlcov
