FROM python:3

COPY requirements.txt ./
COPY Ref_train.csv ./
COPY Ref_test.csv ./

ENV PYTHONUNBUFFERED 1
RUN pip install -r requirements.txt
ENTRYPOINT ["usr/local/bin/python"]
CMD ["python", "ml.py"]