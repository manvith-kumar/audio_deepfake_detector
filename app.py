import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from model.inference import load_model, get_prediction

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    print("--- Attempting to load ML Model ---")
    load_model()
    print("--- Model Loaded Successfully. Ready for predictions. ---")
except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! CRITICAL ERROR: Could not load the model file. !!!")
    print(f"!!! Error: {e} !!!")
    print("!!! Please run the training script first to generate 'model_best.pth'. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = get_prediction(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "File type not allowed. Please use .wav, .mp3, or .flac"}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)