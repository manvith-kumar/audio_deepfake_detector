# Audio DeepFake Detection Web Application 🚀

This project is a fully functional web application that detects deepfaked audio using a trained machine learning model. It was developed as part of an internship project at Valise Technologies.

The application allows a user to upload an audio file through a simple web interface. The backend, built with Flask, processes the audio, extracts features, and uses a trained PyTorch (CNN) model to predict whether the audio is **real** or **fake**.

-----

## \#\# Features

  - **Interactive Frontend:** A clean and responsive user interface built with HTML, CSS, and JavaScript for easy file uploads.
  - **Robust Backend API:** A powerful backend powered by Flask that handles file processing and serves model predictions.
  - **Deep Learning Model:** A Convolutional Neural Network (CNN) trained on log-Mel spectrograms to accurately classify audio authenticity.
  - **Complete ML Pipeline:** The repository includes the full pipeline, from scripts for data preprocessing and feature extraction to model training and evaluation.

-----

## \#\# Technology Stack

  * **Backend:** Python, Flask
  * **Machine Learning:** PyTorch, Librosa, Scikit-learn
  * **Frontend:** HTML5, CSS3, JavaScript
  * **Data Handling:** NumPy, Pandas

-----

## \#\# Project Structure

The repository is organized to separate the machine learning source code from the web application integration code for clarity and scalability.

```
audio_deepfake_detector/
├── checkpoints/              # Contains the trained model file
├── src/                     # Core ML source code for training, eval, etc.
├── model/                   # Python package for model inference in the web app
├── static/                  # CSS and JS files for the frontend
├── templates/               # HTML template for the web page
├── app.py                   # Main Flask web server
├── preprocess_and_extract.py # Script to generate and process data
├── requirements.txt         # Project dependencies
└── .gitignore               # Files and folders to be ignored by Git
```

-----

## \#\# Quick Start: Setup and Usage

Follow these steps to get the application running on your local machine.

### \#\#\# 1. Prerequisites

Make sure you have **Python 3.9+** and **Git** installed on your system.

### \#\#\# 2. Clone the Repository

Open your terminal and clone the repository. The trained model is included and will be downloaded automatically.

```bash
git clone https://github.com/manvith-kumar/audio_deepfake_detector
cd audio_deepfake_detector
```

### \#\#\# 3. Set Up the Environment

Create and activate a Python virtual environment. This keeps your project dependencies isolated.

```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate
```

### \#\#\# 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### \#\#\# 5. Run the Application

You are now ready to launch the web server\!

```bash
flask run
```

Once the server is running, open your web browser and navigate to **`http://127.0.0.1:5000`**. You can now upload an audio file to test the deepfake detector.

-----

## \#\# (Optional) Training the Model from Scratch

If you wish to train a new model yourself, you can use the provided scripts.

### \#\#\# 1. Generate and Preprocess Data

This command will create a small demo dataset in a `data/` folder and prepare it for training.

```bash
python preprocess_and_extract.py --raw-dir data/raw --generate-demo
```

### \#\#\# 2. Run the Training Script

This will train the model using the data from the previous step and save the best-performing version to `checkpoints/model_best.pth`.

```bash
python -m src.train --metadata data/metadata.csv --out-dir checkpoints
```

Once training is complete, you can run the application as described in Step 5 of the Quick Start guide.
