all: lint run

install:
	@pip install --upgrade pip
	@pip install -r requirements.txt

# TODO: run .env file to load environment variables.
run:
	. venv/bin/activate
	python main.py

# TODO: should lint src and working directory.
lint:
	@black .

# TODO: should remove all .pyc and pycache folders.
clean:
	@echo "cleaing project of *.pyc"
