import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from medai.image_io import (
    downscale_for_display,
    img_to_b64_png,
    load_image_bytes,
    superimpose_heatmap,
    to_model_input,
)
from medai.predictor import MultiPredictor

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

predictor = MultiPredictor()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/health')
def health():
    info = predictor.info()
    return jsonify({
        "status": "ok",
        "modalities": {k: vars(v) for k, v in info.items()},
    })


@app.route('/metadata')
def metadata():
    info = predictor.info()
    return jsonify({k: vars(v) for k, v in info.items()})


@app.route('/list_dataset/<category>')
def list_dataset(category):
    try:
        path = os.path.join('clinical_dataset', category)
        return jsonify(sorted(f for f in os.listdir(path) if f.endswith(('.png', '.jpg'))))
    except Exception:
        app.logger.exception("Failed to list dataset category %s", category)
        return jsonify([]), 500


@app.route('/gradcam', methods=['POST'])
def run_gradcam():
    res = {
        "label": "Inconclusive",
        "confidence": 0.0,
        "heatmap": "",
        "overlay": "",
        "medical_data": {
            "severity": "Unknown",
            "recommendation": "Service Ready",
            "detected_type": "Unknown",
            "model_type": "Unavailable",
            "is_pathological": False,
        },
    }

    try:
        if 'image' not in request.files:
            res["error"] = "missing_image"
            return jsonify(res), 400

        file = request.files['image']
        img_bytes = file.read()
        if not img_bytes:
            res["error"] = "empty_image"
            return jsonify(res), 400

        rgb = load_image_bytes(img_bytes)
        modality = request.form.get("modality", "chest").lower()
        info = predictor.info().get(modality)
        input_size = info.input_size if info else 224
        img_batch = to_model_input(rgb, size=input_size)

        payload, heatmap, status = predictor.predict_with_heatmap(img_batch, modality=modality)
        if status != 200:
            payload["modality"] = modality
            return jsonify(payload), status

        display_rgb = downscale_for_display(rgb, max_side=1024)
        overlay, jet = superimpose_heatmap(display_rgb, heatmap)

        payload["heatmap"] = img_to_b64_png(jet)
        payload["overlay"] = img_to_b64_png(overlay)
        payload["modality"] = modality
        return jsonify(payload)

    except Exception:
        app.logger.exception("Grad-CAM processing failed")
        res["error"] = "processing_failed"
        return jsonify(res), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(port=port, debug=debug)
