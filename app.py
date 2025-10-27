import string
import nltk
import pickle
import os
from flask import Flask, request, render_template
from nltk.corpus import stopwords

# --- REQUIRED: Your Text Processing Function ---
# This *must* be defined so pickle can load the models
def convertmyTxt(rv):
    np_chars = [c for c in rv if c not in string.punctuation]
    np_joined = ''.join(np_chars)
    return [w for w in np_joined.split() if w.lower() not in stopwords.words('english')]

# --- Point NLTK to local 'nltk_data' folder ---
# (This is the best-practice method for deployment)
project_root = os.path.dirname(__file__)
nltk_data_path = os.path.join(project_root, 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
    print("NLTK stopwords found locally.")
except LookupError:
    print("--- ERROR: 'nltk_data' folder not found. ---")
    print("--- Please follow deployment instructions to add it. ---")


# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load All Trained Models ---
models = {} # Dictionary to hold our models
model_files = {
    'rf': 'rf_model.pkl',
    'svc': 'svc_model.pkl',
    'lr': 'lr_model.pkl'
}

for model_key, file_name in model_files.items():
    try:
        with open(file_name, 'rb') as handle:
            models[model_key] = pickle.load(handle)
        print(f"Model {file_name} loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {file_name} file not found.")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")


# --- Define Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and returns the prediction."""
    if request.method == 'POST':
        try:
            # 1. Get data from the form
            review_text = request.form['review']
            model_choice = request.form['model_choice'] # e.g., 'rf', 'svc', 'lr'

            # 2. Select the chosen model from our dictionary
            selected_model = models.get(model_choice)
            
            if selected_model is None:
                return render_template('index.html',
                                       prediction_text=f"Error: Model '{model_choice}' is not loaded.")

            # 3. Make predictions
            prediction = selected_model.predict([review_text])[0]
            prediction_proba = selected_model.predict_proba([review_text])[0]
            model_classes = selected_model.classes_

            # 4. Format the output
            prob_map = {label: prob for label, prob in zip(model_classes, prediction_proba)}
            fake_prob_percent = round(prob_map.get('CG', 0) * 100, 2)
            real_prob_percent = round(prob_map.get('OR', 0) * 100, 2)
            
            if prediction == 'CG':
                result_text = "This review is likely FAKE."
            else:
                result_text = "This review is likely REAL."

            # 5. Render the page again with the results
            return render_template('index.html',
                                   prediction_text=result_text,
                                   fake_prob=f"Probability of being FAKE (CG): {fake_prob_percent}%",
                                   real_prob=f"Probability of being REAL (OR): {real_prob_percent}%",
                                   submitted_review=review_text,
                                   last_model_choice=model_choice) # Pass back the choice

        except Exception as e:
            error_message = f"An error occurred during prediction: {e}"
            return render_template('index.html', prediction_text=error_message)

    return home() # Fallback


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)