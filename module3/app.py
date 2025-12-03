from flask import Flask, render_template, request, redirect
import os
from processing import (
    compute_gradients, compute_log, detect_edge_keypoints,
    detect_corner_keypoints, extract_boundary,
    detect_aruco_segmentation, run_sam2_segmentation
)

app = Flask(__name__)

DATASETS = {
    "original": "static/images_original/",
    "aruco": "static/images_aruco/"
}

RESULTS = "static/results/"

def clear_results(subfolder):
    path = os.path.join(RESULTS, subfolder)
    if not os.path.exists(path):
        os.makedirs(path)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/module/<int:num>")
def module(num):
    if num == 3:
        return render_template("module3.html")
    return f"<h2>Module {num} content coming soon...</h2> <a href='/'>Back</a>"

@app.route("/process/<task>", methods=["GET"])
def process(task):
    dataset = request.args.get("dataset", "original")
    input_folder = DATASETS[dataset]

    if task == "gradient":
        clear_results("gradient")
        compute_gradients(input_folder, RESULTS + "gradient/")
        images = os.listdir(RESULTS + "gradient/")

    elif task == "log":
        clear_results("log")
        compute_log(input_folder, RESULTS + "log/")
        images = os.listdir(RESULTS + "log/")

    elif task == "edge":
        clear_results("edge")
        detect_edge_keypoints(input_folder, RESULTS + "edge/")
        images = os.listdir(RESULTS + "edge/")

    elif task == "corner":
        clear_results("corner")
        detect_corner_keypoints(input_folder, RESULTS + "corner/")
        images = os.listdir(RESULTS + "corner/")

    elif task == "boundary":
        clear_results("boundary")
        extract_boundary(input_folder, RESULTS + "boundary/")
        images = os.listdir(RESULTS + "boundary/")

    elif task == "aruco":
        clear_results("aruco")
        detect_aruco_segmentation(input_folder, RESULTS + "aruco/")
        images = os.listdir(RESULTS + "aruco/")

    elif task == "sam2":
        clear_results("sam2")
        run_sam2_segmentation(input_folder, RESULTS + "sam2/")
        images = os.listdir(RESULTS + "sam2/")

    else:
        return "Invalid task"

    return render_template("results.html", images=images, task=task)

if __name__ == "__main__":
    app.run(debug=True)

