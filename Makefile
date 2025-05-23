install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

lint:
	pylint --disable=R,C,W *.py

format:
	pyink *.py

all:
	install lint format