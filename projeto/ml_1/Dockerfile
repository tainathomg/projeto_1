FROM python:3

COPY classification.py ./
COPY requirements.txt ./
COPY Ref_train.csv ./
COPY Ref_test.csv ./

RUN pip install -r requirements.txt
CMD ["python", "classification.py"]