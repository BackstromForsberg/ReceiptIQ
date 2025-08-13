from flask import Flask, request, jsonify
from vision_ocr.receipt_ocr_scanner import OptimizedReceiptOCRScanner





# Create a Flask app instance
app = Flask(__name__)

# Initialize Optimized OCR scanner
scanner = OptimizedReceiptOCRScanner()

# Define a route
@app.route("/hello")
def hello_world():
    return "Hello, Receipt IQ!"

@app.post("/ocr")
def ocr():
    # Accept raw bytes (image/* or application/pdf)
    data = request.get_data(cache=False, as_text=False)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    # Optional: use a hint for naming in results/metadata
    source_name = request.headers.get("X-Filename", "upload")

    result = scanner.extract_receipt_data_from_bytes(data, source_name=source_name)
    # If you want to persist results on disk, uncomment:
    # scanner.save_results(result, source_name=source_name)

    return jsonify(result), 200

# Define a route for the favicon to remove 404 log
@app.route('/favicon.ico')
def favicon():
    return ('', 204)  # No Content

# Run the app if this file is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)