# import libraries
from flask import Flask, request, render_template

import os
import pickle
import warnings
import numpy as np
import seaborn as sns
sns.set_style("dark")
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# create app and load the trained Model
app = Flask(__name__)

BASE_DIR = 'Dataset'
WORKING_DIR = 'Supporting Material'

with open(os.path.join(BASE_DIR, 'Captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

captions_dict = {}

for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in captions_dict:
        captions_dict[image_id] = []
    captions_dict[image_id].append(caption)

def process_caption(captions_dict):
    for key, captions in captions_dict.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
process_caption(captions_dict)

captions = []
for key in captions_dict:
    for caption in captions_dict[key]:
        captions.append(caption)
        
captions[:25]



tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

print("\n=================")
print("VOCAB_SIZE: ", vocab_size)
print("MAX_LENGTH: ", max_length)
print("=================\n")






def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    return in_text


with open(os.path.join(WORKING_DIR, 'features3.pkl'), 'rb') as f:
    features3 = pickle.load(f)

def get_prediction(image_name):

    image = load_img(image_name, target_size=(224, 224))

    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    image = preprocess_input(image)
    model3 = ResNet50()
    model3 = Model(inputs=model3.inputs, outputs=model3.layers[-2].output)
    feature = model3.predict(image, verbose=0)
    
    model = load_model(WORKING_DIR+'/best_model3.h5')
    y_pred3 = predict_caption(model, feature, tokenizer, max_length)
    print('--------------------Predicted--------------------')
    return y_pred3


# Route to handle HOME
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle PREDICTED RESULT
@app.route('/',methods=['POST'])
def predict():
  
    inputs = [] # declaring input array
    myfile = request.form['myfile']
    
    # Set Path
    process_caption(captions_dict)

    
    # Load Features From Pickle


    
    
    # Load ResNet50 Model

    
    
    
    # Load Model

    


# In[6]:


# Pre-Process Captions




# In[10]:





# In[16]:


    import  tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print(file_path)


# In[17]:


    a = get_prediction(file_path)
    
    
    
    
    
    
    
    
    
    return render_template('index.html', predicted_result = a)

if __name__ == "__main__":
    app.run(debug=True)
