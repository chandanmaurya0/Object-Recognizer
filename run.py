from flask import Flask, render_template,request
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import base64
import io
import re
from scipy.misc import imsave, imread, imresize
from flask import jsonify
from keras.models import load_model
from load import *
app = Flask(__name__)
#global model
global model, graph
#initialize these variables
model, graph = init()
@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/about')
def About():
	return render_template('About.html')




def convertImage(imgData1):
	#imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))


@app.route('/predict',methods=['POST'])
def predict():
	message =request.get_json(force=True)
	imgData=message['image']
	bytesData = bytes(imgData, 'utf-8')
	convertImage(bytesData)
	img = 'output.png'
	img = image.load_img(img, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	#model = ResNet50(weights='imagenet')
	with graph.as_default():
		#perform the prediction
		preds = model.predict(x)
		pred = decode_predictions(preds, top=2)
		response ={
		'prediction':{
		'pred1':pred[0][0][1],
		'confi1':str("{0:.2f}".format(pred[0][0][2]*100)),
		'pred2':pred[0][1][1],
		'confi2':str("{0:.2f}".format(pred[0][1][2]*100))
		}
		}
		return jsonify(response)

if __name__ == "__main__":
	app.run(debug=True)