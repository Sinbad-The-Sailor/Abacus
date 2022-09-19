FROM python:3.9

WORKDIR /abacus

COPY requirements_test.txt requirements_test.txt

RUN pip install -r requirements_test.txt

COPY . .

RUN pip install -e .

CMD ["pytest"]
