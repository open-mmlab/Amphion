FROM python:3.10

WORKDIR /app

COPY . .

EXPOSE 8000

ENTRYPOINT [ "python -m http.server 8000" ]

# docker build -t diffsvc .
# docker run -v $(pwd)/data:/app/data -p 8000:8000 diffsvc