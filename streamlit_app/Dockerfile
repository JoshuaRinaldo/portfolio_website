FROM python:3.12
EXPOSE 8080
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN  pip3 install --upgrade pip && pip3 install -r requirements.txt
COPY . .

# Command overriden by docker-compose
CMD ["streamlit" "run" "homepage.py" "--server.port=8080" "--server.address=0.0.0.0"]