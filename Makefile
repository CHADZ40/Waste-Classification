.PHONY: install split baseline transfer finetune compare app test lint

install:
	pip install -e ".[dev,app]"

split:
	python -m waste_classifier.split --config configs/default.yaml

baseline:
	python -m waste_classifier.train --config configs/basic_cnn.yaml
	python -m waste_classifier.evaluate --config configs/basic_cnn.yaml

transfer:
	python -m waste_classifier.train --config configs/transfer.yaml
	python -m waste_classifier.evaluate --config configs/transfer.yaml

finetune:
	python -m waste_classifier.train --config configs/transfer_real_world.yaml
	python -m waste_classifier.evaluate --config configs/transfer_real_world.yaml

compare:
	python -m waste_classifier.compare

app:
	python app/app.py

test:
	pytest

lint:
	ruff check src tests app
