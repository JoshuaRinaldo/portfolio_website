FROM public.ecr.aws/lambda/python:3.12

# Install spacy and small pipeline
RUN pip install -U pip setuptools wheel \
  && pip install -U spacy \
  && python -m spacy download en_core_web_sm

COPY text_counterfactuals.py .
CMD ["text_counterfactuals.handler"]