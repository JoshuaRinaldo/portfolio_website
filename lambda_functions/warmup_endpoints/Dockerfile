FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy function code
COPY warmup_endpoints.py ${LAMBDA_TASK_ROOT}
RUN chmod 644 ${LAMBDA_TASK_ROOT}/warmup_endpoints.py

# Set the CMD to your handler
CMD [ "warmup_endpoints.handler" ]