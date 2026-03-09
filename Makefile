.PHONY: setup preprocess eda mmm attribution optimize dashboard test verify clean

# ============================================================
# Marketing Attribution & Budget Optimization
# ============================================================

setup:
	pip install -r requirements.txt

preprocess:
	python scripts/preprocess.py

eda:
	jupyter notebook notebooks/01_eda.ipynb

mmm:
	python scripts/mmm_model.py

attribution:
	python scripts/generate_touchpoints.py
	python scripts/multi_touch_attribution.py

optimize:
	python scripts/budget_optimizer.py

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

verify: lint format-check test audit

all: preprocess mmm attribution optimize

clean:
	rm -rf data/processed/*.parquet
	rm -rf data/processed/*.duckdb
	rm -rf reports/images/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} +

# === Quality gates (extended) ===

lint:
	ruff check scripts/ dashboard/ --ignore E501,F401,E402

# === Quality gates (extended) ===

format:
	ruff format scripts/ dashboard/

format-check:
	ruff format --check scripts/ dashboard/

audit:
	python scripts/audit_consistency.py
