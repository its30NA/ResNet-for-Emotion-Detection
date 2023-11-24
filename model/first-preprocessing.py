# !pip install tensorflow.python.keras.applications.resnet50 --quiet
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model
from keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import os


def preproc(img_path):
    '''
    This is how the preprocessing of an image looks like.
    '''
    img = image.load_img(img_path, target_size=(48, 48)) # width and height should match the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # model expects a batch of images
    img_array /= 255.0 # scale pixel values if required

    return img_array


def f1_score(y_true, y_pred):
    '''
    taken from old keras source code, this function defines f1_score for our model to run perfectly.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model = load_model("models/ResNet50.h5", custom_objects={'f1_score': f1_score})

def predict(img_path):
    '''
    Checks if there are any images with correct image formats in the given directory.
    Then predicts and prints the detected emotion for each image.
    '''
    for filename in os.listdir(img_path):
        # Check if the file is an image (e.g., .jpg, .png)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_dir = os.path.join(img_path, filename)
            img_array = preproc(img_dir)

            predictions = model.predict(img_array)

            # Process the predictions (e.g., using argmax to get the class index)
            predicted_class_index = np.argmax(predictions, axis=1)
            class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            predicted_class = class_names[predicted_class_index[0]]

            print(f'DDDDDDDDDRUM ROLLLLS:\n\nDetected Emotion:    {predicted_class}\n')

img_path = input('Where is the directory to your images?\nA: raw_data/test_FER2013?    or    B: Somewhere else?    or C: Cancel this process?\n')

if img_path == 'A' or img_path == 'a':
    print('Detecting sets of test data:\n')
    emotion = input(f'Which emotion you want the model to use as test?\n{os.listdir("../raw_data/test_FER2013")}\n\n')
    print(f'Detecting emotions for images in: raw_data/test_FER2013/{emotion}:\n')
    img_path = f'../raw_data/test_FER2013/{emotion}'
    predict(img_path)
elif img_path == 'B' or img_path == 'b':
    img_path = input('Please input the exact path to the directory with images stored in them:\n')
    print(f'Detecting emotions for images in: {img_path}:\n')
    predict(img_path)
else:
    print('See you later :D')
