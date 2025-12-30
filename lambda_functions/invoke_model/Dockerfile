FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy function code and set proper permissions
COPY invoke_model.py ${LAMBDA_TASK_ROOT}
RUN chmod 644 ${LAMBDA_TASK_ROOT}/invoke_model.py

# Set the CMD to your handler
CMD [ "invoke_model.handler" ]
