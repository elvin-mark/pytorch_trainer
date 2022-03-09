from socket import socket
from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
import torch
import torchvision
import pickle
import PIL
import base64
import io

app = Flask(__name__)
app.config["SECTRET_KEY"] = "dashboard_secret"
socketio = SocketIO(app)

sid = None
playground = None
dev = torch.device("cpu")


def get_image_tensor(img_src, extra_info):
    N, H, W = extra_info
    transf_img = torchvision.transforms.Compose([
        torchvision.transforms.Resize((H, W)),
        torchvision.transforms.ToTensor()])

    img_src = img_src.split(",")[1]
    img_bytes = io.BytesIO(base64.b64decode(img_src))
    I = PIL.Image.open(img_bytes).convert("RGB" if N == 3 else "L")
    return transf_img(I)


@app.route("/")
def home():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    global sid
    print("user connected")
    sid = request.sid


@app.route("/send_data", methods=["POST"])
def handle_data():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("data", data, namespace="/", room=sid)
    return "received!"


@app.route("/send_samples", methods=["POST"])
def handle_samples():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("samples", data, namespace="/", room=sid)
    return "received!"


@app.route("/send_landscape", methods=["POST"])
def handle_landscape():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("landscape", data, namespace="/", room=sid)
    return "received!"


@app.route("/send_model", methods=["POST"])
def handle_model():
    if request.method == "POST":
        data = request.get_json()
        with app.test_request_context("/"):
            emit("model", data, namespace="/", room=sid)
    return "received!"


@app.route("/playground_sample", methods=["POST"])
def handle_playground_sample():
    global playground
    if request.method == "POST":
        data = request.get_json()
        if playground is None:
            model_ = torch.load("tmp/tmp_model.ckpt", map_location=dev)
            model_.eval()
            with open("tmp/tmp_extra_info.pkl", "rb") as f:
                extra_info_ = pickle.load(f)
            playground = {"model": model_, "extra_info": extra_info_}

        x = get_image_tensor(
            data["data"], playground["extra_info"]["image_shape"]).to(dev)
        prob = torch.softmax(playground["model"](
            x[None, :, :, :]), dim=1).detach()
        idxs = torch.argsort(prob, axis=1, descending=True)[0].numpy()

        response = {"categories": [playground["extra_info"]["labels"][i]
                                   for i in idxs[:5]], "prob": [prob[0][i].item() for i in idxs[:5]]}
        return response
    return "received"


if __name__ == "__main__":
    socketio.run(app)
