# # from flask import Flask, request, jsonify, render_template
# # from tensorflow.keras.models import load_model
# # import numpy as np
# # import cv2
# # import tensorflow as tf
# # import io

# # app = Flask(__name__)

# # # Load the model
# # model = load_model('my_model.h5')

# # def preprocess_image(image_bytes):
# #     # Convert bytes to numpy array
# #     nparr = np.frombuffer(image_bytes, np.uint8)
# #     # Decode image
# #     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# #     # Resize
# #     img = cv2.resize(img, (256, 256))
# #     # Convert to float32 and normalize
# #     img = img.astype('float32') / 255.0
# #     # Add batch dimension
# #     img = np.expand_dims(img, axis=0)
# #     return img

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         # Check if image was uploaded
# #         if 'image' not in request.files:
# #             return jsonify({'error': 'No image uploaded'}), 400

# #         file = request.files['image']

# #         # Read and preprocess the image
# #         image_bytes = file.read()
# #         processed_image = preprocess_image(image_bytes)

# #         # Make prediction
# #         prediction = model.predict(processed_image)
# #         print("Raw prediction:", prediction)  # Debugging line

# #         # Process result
# #         probability = float(prediction[0][0])
# #         result = 'Dog' if probability >= 0.5 else 'Cat'
# #         confidence = probability if result == 'Dog' else 1 - probability

# #         return jsonify({
# #             'success': True,
# #             'result': result,
# #             'confidence': f'{confidence * 100:.2f}%'
# #         })

# #     except Exception as e:
# #         print(f"Error during prediction: {str(e)}")  # Debugging line
# #         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib

# app = Flask(__name__)

# # Load all models
# cnn_model = load_model('my_model.h5')
# logistic_model = joblib.load('logistic_model.joblib')
# kmeans_model = joblib.load('kmeans_model.joblib')

# def preprocess_image(image, model_type):
#     if model_type == 'cnn':
#         img = cv2.resize(image, (256, 256))
#         img = img.reshape((1, 256, 256, 3))
#         img = img / 255.0
#     else:  # for logistic and kmeans
#         img = cv2.resize(image, (64, 64))  # Resize to 64x64 for Logistic Regression and K-means
#         img = img.reshape(1, -1)  # Flatten the image
#         img = img / 255.0  # Normalize the pixel values
#         img = img.astype('float64')  # Ensure the data type is float64
#     return img

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# # @app.route('/predict', methods=['POST'])
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         file = request.files['file']
#         model_type = request.form['model']

#         # Read and preprocess image
#         nparr = np.fromstring(file.read(), np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if image is None:
#             return jsonify({'result': 'Cat'})  # Default to Cat if image is not valid

#         processed_image = preprocess_image(image, model_type)

#         if model_type == 'cnn':
#             prediction = cnn_model.predict(processed_image)[0][0]
#             result = 'Dog' if prediction > 0.5 else 'Cat'
#         elif model_type == 'logistic':
#             prediction = logistic_model.predict(processed_image)
#             result = 'Dog' if prediction[0] == 1 else 'Cat'
#         else:  # kmeans
#             prediction = kmeans_model.predict(processed_image)
#             result = 'Dog' if prediction[0] == 1 else 'Cat'

#         return jsonify({'result': result})

#     except Exception as e:
#         print(e)
#         return jsonify({'result': 'unknown'})  # Default to Cat on error

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load models
# cnn_model = load_model('my_model.h5')
rf_model = joblib.load('random_forest_model.joblib')

def preprocess_image(image, model_type):
    if model_type == 'cnn':
        img = cv2.resize(image, (256, 256))
        img = img.astype('float32') / 255.0
        img = img.reshape((1, 256, 256, 3))
    else:  # random forest
        img = cv2.resize(image, (64, 64))
        img = img.astype('float32') / 255.0
        img_flat = img.flatten().reshape(1, -1)
        return img_flat
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        model_type = request.form['model']

        # Read and preprocess image
        nparr = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'result': 'Error', 'confidence': 0})

        processed_image = preprocess_image(image, model_type)

        if model_type == 'cnn':
            prediction = cnn_model.predict(processed_image)[0][0]
            result = 'Dog' if prediction > 0.5 else 'Cat'
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        else:  # random forest
            prediction = rf_model.predict(processed_image)[0]
            probability = rf_model.predict_proba(processed_image)[0]
            result = 'Dog' if prediction == 0 else 'Cat'  # Note the change here to match your RF model
            confidence = float(probability[1] if prediction == 0 else probability[0])

        return jsonify({
            'result': result,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'result': 'Error', 'confidence': 0})

if __name__ == '__main__':
    app.run(debug=True)