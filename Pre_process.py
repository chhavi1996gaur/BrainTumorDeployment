import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import cv2
import numpy as np
import os
#from tqdm.notebook import tqdm

def get_label(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
            return key



def make_pred(img_path, model):
    categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    label=[i for i in range(len(categories))]
    label_dict=dict(zip(categories,label))
    img = image.load_img(img_path, target_size = (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.array(img_array)/255
    img_array = np.expand_dims(img_array, axis = 0)
    y_pred = model.predict(img_array)
    y = np.argmax(y_pred)
    label = get_label(y, label_dict)
    return label


def pre_process_img(img_size, Image_Path):
    #Directory_sample = '/kaggle/input/brain-tumor-classification-mri/Training'
    categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    label=[i for i in range(len(categories))]
    label_dict=dict(zip(categories,label))
    data = []
    labels = []
    #img_size = 514
    for category in categories:
        path = os.path.join(Image_Path, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            try:
                color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resize_image = cv2.resize(color, (img_size, img_size))
                data.append(resize_image)
                labels.append(label_dict[category])
            except Exception as e:
                print('Exception', e)
    data = data = np.array(data)/255
    data = np.reshape(data, (data.shape[0], img_size, img_size, 3))
    labels = np.array(labels)
    labels = to_categorical(labels)
    return data, labels