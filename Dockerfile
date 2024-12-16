FROM python:3.10.16-alpine3.21

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

EXPOSE 5000

CMD [ "flask", "run"]