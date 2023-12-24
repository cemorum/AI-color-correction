import os
from random import randint
from flask import Flask, render_template, request, jsonify
from predict_video import *
from predict_image import *
from time import sleep

app = Flask(__name__)

# Подготовка директории для хранения
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

params = [0,0]
@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()
    isFlares = data['flares']
    isShadows = data['shadows']
    params[0] = isFlares
    params[1] = isShadows
    return params

imageFiles = []
tempFiles = []
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if "video" in request.files:
                image = request.files['video']
                fn = image.filename
                input_path = f'static/uploads/{fn}'
                image.save(input_path)
                img_path = f'static/uploads/cc_{fn}'

                predict_video(image, img_path, 'best_model_epoch_14_loss_0.0005.pth')
            elif "image" in request.files:
                image = request.files['image']
                fn = image.filename
                input_path = f'static/uploads/{fn}'
                image.save(input_path)
                img_path = f'static/uploads/cc_{fn}'

                flare_detection = params[0] == 1
                shad_detection = params[1] == 1

                model_path = flare_detection and 'best_model_epoch_16_loss_0.0005.pth' or 'best_model_epoch_14_loss_0.0005.pth'
                
                predict_image(image, img_path, model_path, flare_detection, shad_detection)
            return jsonify({'input' : input_path, 'result': img_path})
        # Очистка временных файлов
        except Exception as e:
            print("[!]", e)

    return render_template('home.html')

# Очистка временных файлов за сессию
def collect(items):
    ...

app.run(debug=True)
collect(imageFiles)