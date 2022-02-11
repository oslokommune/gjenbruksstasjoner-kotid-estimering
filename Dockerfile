# python:3.7-bullseye
FROM python@sha256:48e1422053164310266a9a85e4f3886733a5f1dc025238dba229068806aff4d6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY queue_time_predictions queue_time_predictions
COPY start.sh ./

RUN chmod +x /usr/src/app/start.sh

CMD ["sh", "-c", "/usr/src/app/start.sh"]
