# Web flask library url, file upload, bootstrap, csv
import os
from flask import Flask, flash, render_template, url_for, request, redirect
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap 
import csv

# machine learning import lib
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
from keras.models import model_from_json
from tfidf import TFIDF
from random import shuffle
from keras import backend as keras

# setup dir

UPLOAD_FOLDER = '/flask/aplikasi/data'
ALLOWED_EXTENTIONS = set(['csv'])
folder = "data/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

# Create a directory in a known location to save files to.
# uploads_dir = os.path.join(app.instance_path, 'data')
# os.makedirs(uploads_dir)

#setup variabel

xdata = []
ydata = []
clasification = []

#ML function

def preproses(filepath='data/jokpra.csv'):
	global ydata
	global xdata

	f = open(filepath)
	sents = f.read().split('\n')
	shuffle(sents)
	for sent in sents:
		temp = sent.split(';')
		if len(temp) == 2:
			xdata.append(temp[0])
			ydata.append([int(temp[1])])

def getBinaryResult(x):
	return "POSITIF" if x >= 0.5 else "NEGATIF"

def testFromTrained(x):
	model = Sequential()

	# load json and create model
	json_file = open('models/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	# load weights into new self.model
	model.load_weights("models/model_trainb1.h5")
	# print("Loaded model from disk")

	sgd = SGD(lr=0.01)

	model.compile(loss='binary_crossentropy', optimizer=sgd)
	return getBinaryResult(model.predict_proba(np.array(x)))

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENTIONS

#ML Fnction

@app.route('/')
def index():

	return render_template('index.html')

@app.route('/parsing')
def parsing():

	with open('data/test.csv', 'r') as csv_par:
		preproses()
		td = TFIDF([xdata, ydata])
		rowdata = []
		clasification = []
		csv_reader = csv_par.read().split('\n')	
	for row in csv_reader:
		rowdata.append(row)
		clasification.append(testFromTrained([td.transform(row)]))

	keras.clear_session()
	labels, values = np.unique(clasification, return_counts=True)
	lbls, vals = np.unique(clasification, return_counts=True)

	pie_labels = labels
	pie_values = values
	colors = ["#F7464A", "#46BFBD"]

	return render_template('hasil.html', set=zip(values, labels, colors), clasification=zip(csv_reader, clasification), legenda=zip(lbls, vals))

@app.route('/predict', methods=['POST', 'GET'])
def predict():
	preproses()
	td = TFIDF([xdata, ydata])
	clasification = []

# Receives the input query from form
	if request.method == 'POST':
		namequery = request.form['namequery']
		spliter = namequery.split(',')
		
		for row in spliter:
			clasification.append(testFromTrained([td.transform(row)]))
		print (clasification)
		keras.clear_session()

		labels, values = np.unique(clasification, return_counts=True)
		lbls, vals = np.unique(clasification, return_counts=True)

	pie_labels = labels
	pie_values = values
	colors = ["#F7464A", "#46BFBD"]

	return render_template('hasil.html', set=zip(values, labels, colors), clasification=zip(spliter, clasification), legenda=zip(lbls, vals))

@app.route('/coba')
def coba():
	return render_template('upload.html')

@app.route('/upload_file', methods=['POST', 'GET'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('Not file part')
			# return redirect(request.url)
		file = request.files['file']

		if file.filename == '':
			flask('not select file')
			# return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# return redirect(url_for('upload_file', filename=filename))
		print (filename)
		fold = "data/"+filename
		print (fold)
		with open(fold, 'r') as csv_par:
			preproses()
			td = TFIDF([xdata, ydata])
			clasification = []
			csv_reader = csv_par.read().split('\n')

	for row in csv_reader:
		clasification.append(testFromTrained([td.transform(row)]))
	
	keras.clear_session()
	labels, values = np.unique(clasification, return_counts=True)
	lbls, vals = np.unique(clasification, return_counts=True)

	pie_labels = labels
	pie_values = values
	colors = ["#F7464A", "#46BFBD"]

	return render_template('hasil.html', set=zip(values, labels, colors), clasification=zip(csv_reader, clasification), legenda=zip(lbls, vals))
    		
if __name__ == '__main__':
	app.run(debug=True)