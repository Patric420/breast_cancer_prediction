from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model
with open('model_pickle', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = []
        for key in request.form.keys():
            value = float(request.form[key])
            input_data.append(value)
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-cancerous)"
        return render_template('index.html', prediction_text=f'The tumor is predicted to be {result}')
if __name__ == "__main__":
    app.run(debug=True)
