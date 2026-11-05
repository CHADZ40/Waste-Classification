.PHONY: install split baseline transfer finetune final compare app camera test lint

install:
	pip install -e ".[dev,app]"

split:
	python -m waste_classifier.split --config configs/default.yaml
	python -m waste_classifier.near_duplicates --config configs/default.yaml

baseline:
	python -m waste_classifier.train --config configs/basic_cnn.yaml
	python -m waste_classifier.evaluate --config configs/basic_cnn.yaml

transfer:
	python -m waste_classifier.train --config configs/transfer.yaml
	python -m waste_classifier.evaluate --config configs/transfer.yaml

finetune:
	python -m waste_classifier.train --config configs/transfer_real_world.yaml
	python -m waste_classifier.evaluate --config configs/transfer_real_world.yaml

final:
	python -m waste_classifier.train --config configs/transfer_large.yaml
	python -m waste_classifier.evaluate --config configs/transfer_large.yaml

compare:
	python -m waste_classifier.compare

app:
	python app/app.py

camera:
	python app/live_camera.py

test:
	pytest

lint:
	ruff check src tests app
