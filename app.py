from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')  # Load your trained model

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Slot Labels (change as per your training)
labels = ['Vacant', 'Bike', 'Car']

@app.route('/', methods=['GET', 'POST'])
def index():
    status = None
    image_path = None

    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filepath)
            image_path = filepath

            # Preprocess the image
            img = image.load_img(filepath, target_size=(150, 150))  # change size based on your model
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            status = labels[np.argmax(prediction)]

    return render_template('index.html', prediction=status, image_url=image_path)

if __name__ == '__main__':
    app.run(debug=True)
