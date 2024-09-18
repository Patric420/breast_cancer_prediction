from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model
with open('model_pickle', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        input_data = []
        for key in request.form.keys():
            value = float(request.form[key])
            input_data.append(value)
        
        # Convert input data into a numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Based on prediction, return the result
        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-cancerous)"
        
        # Return the result back to the HTML
        return render_template('index.html', prediction_text=f'The tumor is predicted to be {result}')
    
# Run the app
if __name__ == "__main__":
    app.run(debug=True)
