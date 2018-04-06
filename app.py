from flask import Flask, request, render_template
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
	
	return render_template('index.html')

@app.route('/getprediction', methods=['POST'])
def get_prediction():
	amount = request.form.get("amount")
	term = request.form.get("term")
	income = request.form.get("income")
	state = request.form.get("state")
	rent = request.form.get("rent")
	delinquencies = request.form.get("delinquencies")
	fico_low = request.form.get("fico_low")
	fico_high = request.form.get("fico_high")
	last_fico_low = request.form.get("last_fico_low")
	last_fico_high = request.form.get("last_fico_high")
	credit_lines = request.form.get("credit_lines")
	revolving_balance = request.form.get("revolving_balance")
	inquires = request.form.get("inquires")
	employment = request.form.get("employment")
	
	x_factors = [amount, term, income, state, rent, delinquencies, fico_low, fico_high, last_fico_low, last_fico_high, credit_lines, revolving_balance, inquires, employment]
	
	model = joblib.load('trained_interest_rate_model.pkl')
	prediction = model.predict(x_factors)
	return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
	app.debug = True
	app.run()
