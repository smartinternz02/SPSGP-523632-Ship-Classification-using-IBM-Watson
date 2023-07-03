from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the ML model
model = tf.keras.models.load_model(r'C:\Users\cyril\PycharmProjects\AI_Project\model_Using_Transfer_Learning.h5')

app = Flask(__name__, template_folder=r'C:\Users\cyril\PycharmProjects\AI_Project\templates')


# Home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    # Get the uploaded image file
    image_file = request.files.get('image')

    # Read the image using PIL
    image = Image.open(image_file)

    # Preprocess the image
    # Modify the preprocessing steps based on your ML model's requirements

    # Resize the image to match the input size of the model
    image = image.resize((224, 224))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image
    image_array = image_array / 255.0

    # Expand dimensions to match the input shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)

    # Perform ship classification using the model
    prediction = model.predict(image_array)
    # Modify the post-processing steps based on your ML model's output
    label = ['Cargo', 'Carrier', 'Cruise', 'Military', 'Tankers']
    # Get the predicted class label
    class_label = label[np.argmax(prediction)]

    # Return the classification result
    return render_template('prediction.html', class_label=class_label)


if __name__ == '__main__':
    app.run(debug=True)


