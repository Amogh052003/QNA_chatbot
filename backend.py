from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer ,HfArgumentParser,TrainingArguments,pipeline,logging

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load your fine-tuned model
MODEL_NAME = "Amogh052003/tiny_Llama-2b"

try:
    print("Loading model and tokenizer")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    # Load model with GPU acceleration if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)  # Move model to the correct device

    print("Model successfully loaded!")

except Exception as e:
    print(f" Error loading model: {str(e)}")
    model = None
    tokenizer = None

# HTML template for home page
HOME_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q&A Bot Backend</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        h1 { color: #2c3e50; }
        p { font-size: 18px; color: #34495e; }
        .status { padding: 10px; border-radius: 5px; display: inline-block; margin-top: 20px; }
        .online { background-color: #2ecc71; color: white; }
        .offline { background-color: #e74c3c; color: white; }
    </style>
</head>
<body>
    <h1>🤖 Q&A Bot Backend</h1>
    <p>This is the backend server for the Q&A Bot.</p>
    <div class="status online">Backend is running!</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """ Serve a simple HTML home page instead of plain JSON """
    return render_template_string(HOME_PAGE)

@app.route("/status", methods=["GET"])
def status():
    """ API to check backend status """
    return jsonify({"status": "running", "message": "Backend is up and running!"})

@app.route("/ask", methods=["POST"])
def ask():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"🔍 Received question: {question}")

    try:
        logging.set_verbosity(logging.CRITICAL)
        prompt = question
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=400)
        answer = pipe(f"[INST] {prompt} [/INST]")
        answer = answer[0]['generated_text']
        print(f"✅ Answer generated: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"❌ Error generating response: {str(e)}")
        return jsonify({"error": "Failed to generate answer"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
