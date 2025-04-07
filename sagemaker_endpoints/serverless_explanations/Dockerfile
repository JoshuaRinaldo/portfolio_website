FROM amazonlinux:latest

RUN yum -y install python3-pip nginx

# Copy requirements.txt to /opt/program folder
COPY requirements.txt /opt/program/requirements.txt

# Install packages listed in requirements.txt
RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


COPY code /opt/program
WORKDIR /opt/program

# Expose port 8080 for serving
EXPOSE 8080

ENTRYPOINT ["python3"]

# serve is a python script under code/ directory that launches nginx and gunicorn processes
CMD ["serve"]
