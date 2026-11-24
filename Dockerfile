FROM ubuntu:24.04
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Need to make a venv because Ubuntu doesn't allow globally installed Python packages anymore (PEP 668)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install this source tree (not the PyPI release), so the image runs the code it was built from.
COPY pyproject.toml README.md LICENSE /usr/src/hvd-src/
COPY src /usr/src/hvd-src/src
RUN python -m pip install /usr/src/hvd-src
COPY ./docker-entrypoint.sh ./entrypoint.sh

ENV DEDUP_DATABASE_DIR=/usr/src/app/db
ENV HYDRUS_API_URL=http://host.docker.internal:45869
CMD ["./entrypoint.sh"]
