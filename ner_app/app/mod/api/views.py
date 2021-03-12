from flask import Blueprint, request, jsonify

from ..src.train import gen_data
from ..src.predict import Prediction

api = Blueprint("api", __name__)

prediction = Prediction()


@api.route("/")
def index():
    return jsonify({"api": "Testing restful api"})


@api.route("/predict", methods=["GET"])
def predict():
    x = request.json.get("x")

    result = prediction.glove_predict(x)

    return jsonify({"x": x, "result": " ".join([i for _, i in zip(x.split(), result)])})
