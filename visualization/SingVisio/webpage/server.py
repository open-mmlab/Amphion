# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, send_from_directory, jsonify, abort
from flask_cors import CORS


def select_step_range(start, end, embeddings_section):
    kmeans = KMeans(n_clusters=1).fit(embeddings_section)
    center = kmeans.cluster_centers_[0]

    distances = np.linalg.norm(embeddings_section - center, axis=1)

    selected_step = start + np.argmin(distances)
    return int(selected_step)


def select_steps_v2(
    num_steps,
    time_embeddings,
    total_steps=1000,
):
    interval = total_steps / num_steps
    selected_steps = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(num_steps):
            start = int(i * interval)
            end = int((i + 1) * interval)
            embeddings_section = time_embeddings[start:end]
            futures.append(
                executor.submit(select_step_range, start, end, embeddings_section)
            )

        for future in futures:
            selected_steps.append(future.result())

    return selected_steps


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/process_map", methods=["GET"])
def process():
    start_time = time.time()

    input_path = request.args.get("input_path")
    num_steps = request.args.get("num_steps", "0")

    if not input_path or not num_steps:
        abort(
            400,
            description="Missing query parameters: input_path and num_steps are required.",
        )

    input_path = (
        "data" + input_path.split("data")[1] if "data" in input_path else input_path
    )
    input_path = input_path[1:] if input_path.startswith("/") else input_path

    try:
        num_steps = int(num_steps)
    except ValueError:
        abort(400, description="Invalid parameter: num_steps must be an integer.")

    try:
        time_embeddings = np.load(input_path)
        selected_steps = []
        if num_steps != 0:
            time_embeddings_shape = np.asarray(time_embeddings).shape
            if len(time_embeddings_shape) == 4:
                time_embeddings = time_embeddings.squeeze(1).squeeze(1)
            selected_steps = select_steps_v2(num_steps, time_embeddings)
            selected_steps = sorted(selected_steps, reverse=True)
            selected_steps = [str(step) for step in selected_steps]
    except Exception as e:
        abort(500, description=str(e))

    result = {
        "input_path": input_path,
        "num_steps": num_steps,
        "message": "Processing completed successfully.",
        "time_embeddings": str(time_embeddings.shape),
        "selected_steps": selected_steps,
        "time_cost": time.time() - start_time,
    }

    return jsonify(result)


@app.route("/<path:filename>")
def serve_file(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    else:
        abort(404)


if __name__ == "__main__":
    app.run(debug=True)

# tmux new -s singvisio
# conda activate singvisio
# gunicorn -w 8 -b 0.0.0.0:8080 server:app
