FROM python:3.9-slim

WORKDIR /api

COPY ./requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt
RUN pip3 install --force-reinstall tensorflow-io

COPY . .

ENV PORT=8080

EXPOSE 8080

CMD ["python", "./api/main.py"]


