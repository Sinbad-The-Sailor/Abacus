FROM python:3.9

WORKDIR /abacus

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python3" "main.py"]
