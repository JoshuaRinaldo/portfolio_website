FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
RUN pip install --no-cache-dir \
    Pillow==10.2.0 \
    boto3==1.34.34

# Copy function code
COPY photo_processor.py ${LAMBDA_TASK_ROOT}/
RUN chmod 644 ${LAMBDA_TASK_ROOT}/photo_processor.py

CMD [ "photo_processor.handler" ]
