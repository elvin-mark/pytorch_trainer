from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
app.config["SECTRET_KEY"] = "dashboard_secret"
socketio = SocketIO(app)

sid = None


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


if __name__ == "__main__":
    socketio.run(app)
