import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
from pathlib import Path
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

dir_path=Path(r'C:\Projects\fashion-recommender-system\data\Footwear\Men\Images\images_with_product_ids')
# img_list=os.listdir(dir_path)
# print("Image List:\n",img_list)

# print("Length:",len(img_list))
filenames = []

for file in os.listdir(dir_path):
    filenames.append(os.path.join(dir_path,file))
    
feature_list = []

for img_file in tqdm(filenames, desc='Extracting features'):
    features = extract_features(os.path.join(dir_path, img_file), model)
    feature_list.append(features)

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

