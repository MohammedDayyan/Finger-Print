from flask import Flask, request, jsonify
from ml.inference import predict_fingerprint

app = Flask(__name__)

@app.route("/")
def home():
    return "Azure Flask App Running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON with a 'filepath' field
    data = request.get_json()
    filepath = data.get("filepath")
    if not filepath:
        return jsonify({"error": "No filepath provided"}), 400

    result = predict_fingerprint(filepath)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    # Optional: local testing
    app.run(debug=True)
