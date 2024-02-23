# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.10

WORKDIR /app

COPY . .

EXPOSE 8000

ENTRYPOINT [ "python -m http.server 8000" ]

# docker build -t diffsvc .
# docker run -v $(pwd)/data:/app/data -p 8000:8000 diffsvc