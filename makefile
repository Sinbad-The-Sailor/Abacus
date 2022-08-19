all: run

install:
	@pip install --upgrade pip
	@pip install -r requirements.txt

# TODO: run .env file to load environment variables.
run:
	@clear
	. venv/bin/activate
	python main.py

# TODO: should lint src and working directory.
lint:
	@black *.py

# TODO: should remove all .pyc and pycache folders.
clean:
	@echo "cleaing project of *.pyc"
