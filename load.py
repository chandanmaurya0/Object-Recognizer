import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from keras.models import load_model


def init(): 
	model=load_model("ResNet50_model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	#loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return model,graph