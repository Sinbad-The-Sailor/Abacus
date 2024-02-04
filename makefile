all: run

run:
	@python run.py

install:
	@pip install --upgrade pip
	@pip install -r requirements.txt

venv:
	. venv/bin/activate && exec zsh

env:
	@source .env

lint:
	@black src/.

clean:
	echo "cleaing project of *.pyc and __pycache__"
	@find . -type f -name *.pyc -delete
	@find . -type d -name __pycache__ -delete;
