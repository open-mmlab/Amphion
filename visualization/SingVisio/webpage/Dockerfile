# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.10

WORKDIR /app

COPY resources ./resources
COPY img ./img
COPY index.html ./index.html
COPY server.py ./server.py
COPY config ./config

RUN pip install numpy scikit-learn flask flask_cors gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8000

ENTRYPOINT ["gunicorn", "-w", "8", "-b", "0.0.0.0:8000", "server:app"]

# docker build -t singvisio .
# docker run -v $(pwd)/data:/app/data -p 8000:8000 singvisio