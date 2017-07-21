from flask import Flask, request, render_template
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
	
	return render_template('index.html')

@app.route('/getprediction', methods=['POST'])
def get_prediction():
	bmi = request.form.get("BMI")
	age = request.form.get("Age")

	bmi = float(bmi)
	age = float(age)
	x_factors = [bmi, age]
	
	model = joblib.load('trained_diabetes_model.pkl')
	prediction = model.predict(x_factors)
	return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
	app.debug = True
	app.run()
