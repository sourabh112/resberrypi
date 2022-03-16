from flask import Flask, render_template, jsonify
# import test
from flask import request
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import prediction
app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['POST', 'GET'])
def index():
  if request.method == "POST":
    image = request.form["img"]
    image = "./static/"+image
    name = prediction.predict(image)
    data = {
    'name':name,
    'image':image
    }
    return render_template('show.html',data=data)
  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)