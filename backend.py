import warnings
import os
from flask import Flask, flash, request, redirect, url_for
from matplotlib.style import use
from werkzeug.utils import secure_filename
import onnx
import onnxruntime as ort
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def use_model(image):
    onnx_model = onnx.load("./model2.onnx")
    onnx.checker.check_model(onnx_model)
    image = np.array(image.resize((224,)*2))
    image = image[:, :, ::-1].T
    image = image/np.max(image)

    ort_session = ort.InferenceSession("model2.onnx", providers=[
                                       'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = ort_session.run(
        None,
        {"image": image.astype(np.float32)[None, :3, :, :]},
    )
    # print(outputs)
    pre = outputs[0][0][1] > 0
    # print(pre)
    return pre


def get_image_from_request(req_image):
    import re
    from io import BytesIO
    import base64

    image_data = re.sub('^data:image/.+;base64,', '', req_image)
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    return image


@app.route("/upload", methods=['POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.json:
            return "Failed"
        req_image = request.json.get("file")

        # if user does not select file, browser also
        # submit a empty part without filename
        if req_image == '':
            flash('No selected file')
            return redirect(request.url)

        if req_image:
            image = get_image_from_request(req_image)
            warnings.filterwarnings("ignore")
            ort.set_default_logger_severity(3)
            return {"diagnosis": int(use_model(image))}


if __name__ == '__main__':
    app.run(port=3003)
