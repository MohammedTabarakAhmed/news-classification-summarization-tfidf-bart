from flask import Flask, request, jsonify
from src.classifier import predict_text, load_classifier
from src.summarizer import summarize, get_summarizer

app = Flask(__name__)

# Warm up optional models
try:
    _ = load_classifier()
except Exception:
    # model may not exist yet
    pass

@app.route("/", methods=["GET"])
def index():
    return "News classification & summarization API. Use /classify and /summarize endpoints.", 200

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json() or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "no text provided"}), 400
    try:
        labels = predict_text([text])
        return jsonify({"label": labels[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize_route():
    data = request.get_json() or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "no text provided"}), 400
    try:
        summary = summarize(text)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
