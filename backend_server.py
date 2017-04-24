from flask import Flask
from PIL import Image
from flask import request

app = Flask(__name__)

@app.route("/", methods=["POST"])
def home():
    img = Image.open(request.files['file'])
    return 'Success!'