from flask import Flask, render_template, request
import io
import base64
import pickle
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

filename = 'car_rf_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
cylinder_options = [4, 6, 8]
origin_options = ['US', 'Europe', 'Japan']
df = pd.read_csv('data/cars_scrubbed.csv')

@app.route('/')
@app.route('/home')
def home():
  return render_template('home.html', title = "Home")

@app.route('/about')
def about():
  return render_template('about.html', title= 'About')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
  if request.method == 'POST':
    cyl = request.form['cyl']
    disp = int(request.form['disp'])
    hp = int(request.form['hp'])
    weight = int(request.form['weight'])
    acceleration = int(request.form['acceleration'])
    year = int(request.form['year'])
    origin = int(request.form['origin'])
    new_X = np.array([[cyl, disp, hp, weight, acceleration, year, origin]])
    prediction = loaded_model.predict(new_X)[0]
    prediction_str = f'{loaded_model.predict(new_X)[0]:.2f}'
    percentile = f"{len(df[df['mpg'] < prediction])/len(df['mpg'])*100:.2f}"


    fig = Figure()
    ax = fig.subplots()
    ax.set_title("Mpg Distribution")
    ax.set_xlabel("Mpg")
    ax.set_ylabel('Frequency')
    ax.grid()
    ax.hist(df['mpg'])
    ax.axvline(x = prediction, color = 'r')

    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return render_template('results.html', prediction = prediction, prediction_str = prediction_str, percentile=percentile, image=pngImageB64String)
  else:
    return render_template('form.html', cylinder_options= cylinder_options, origin_options=origin_options)

if __name__ == '__main__':
  app.run(debug=True)