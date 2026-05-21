.PHONY: install preprocess train evaluate explain app test clean

install:
	poetry install
	poetry run pre-commit install

preprocess:
	poetry run python -m icu_tft.data.extract
	poetry run python -m icu_tft.features.static
	poetry run python -m icu_tft.features.temporal

train:
	poetry run python -m icu_tft.training.train

evaluate:
	poetry run python -m icu_tft.evaluate.metrics


explain:
	poetry run python -m icu_tft.explain.shap_utils

app:
	poetry run python streamlit run app/main.py

test: 
	poetry run python tests/ -v

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache