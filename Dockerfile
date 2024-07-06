FROM python:3.10.12-slim-buster

COPY ./docker/install.sh .
RUN chmod +x ./install.sh

RUN ./install.sh && python -m venv /opt/venv
RUN apt-get update \
    && apt-get -y install procps

RUN rm ./install.sh

# setup venv as path
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

WORKDIR /opt/app
COPY . .

RUN pip install -r ./requirements.txt

# Main API
CMD ["fastapi", "run", "app.py"]