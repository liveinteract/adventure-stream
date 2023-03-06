import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, Response, jsonify, make_response
from werkzeug.utils import secure_filename
from faceswap import faceswap, punkavatarswap
import cv2
import base64


app=Flask(__name__,template_folder='templates')


IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
INVALID_REQUEST_ERR = 231

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'traindata')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'csv'])
# trained model
MODEL_FOLDER = os.path.join(path, 'models')

MODEL_NAME = ""
jpg_as_text = ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getmodels():
    models = []
    for (_, _, file) in os.walk(MODEL_FOLDER):
        for f in file:
            if '.index' in f:
                models.append(f.replace(".index", ""))
    return models


@app.route('/')
def upload_form():
    models = getmodels()
    return render_template('upload.html', jpg_as_text = jpg_as_text)


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        type = str(request.form.get('comp_select'))        

        if 'source' not in request.files:
            flash('No source Image File')
            return redirect(request.url)

        if type == "face" and 'target' not in request.files:
            flash('No target Image File')
            return redirect(request.url)

        modelpath = os.path.join(app.config['UPLOAD_FOLDER'], MODEL_NAME)        

        if not os.path.isdir(modelpath):
            os.mkdir(modelpath)

        imgsfile = request.files.getlist('source')[0]
        sourcepath = ""
        targetpath = ""
        if imgsfile and allowed_file(imgsfile.filename):
            filename = secure_filename(imgsfile.filename)
            sourcepath = os.path.join(modelpath, filename)
            imgsfile.save(sourcepath)

        imgtfile = request.files.getlist('target')[0]
        if imgtfile and allowed_file(imgtfile.filename):
            filename = secure_filename(imgtfile.filename)
            targetpath = os.path.join(modelpath, filename)
            imgtfile.save(targetpath)


        flash('File(s) successfully uploaded')

        simg = cv2.imread(sourcepath)
        outimg = None
        if type == "face":
            if len(targetpath) > 0:
                timg = cv2.imread(targetpath)
                outimg = faceswap(simg,timg)
        else:
            outimg = punkavatarswap(simg)
        
        if outimg is not None:
            retval, buffer = cv2.imencode('.jpg', outimg)
            jpg_as_bin = base64.b64encode(buffer)
            jpg_as_text = jpg_as_bin.decode('utf-8')

        return render_template('upload.html', jpg_as_text = jpg_as_text)
        

@app.route('/loadmodel/', methods=['POST'])
def loadmodel():
    if request.method == 'POST':
        model = str(request.form.get('comp_select'))
        loadmodel(model)
    #deploy other host and load sample

    return redirect('/')

@app.route('/objectdetection', methods=['GET', 'POST'])
def objectdetection():
    if request.method == 'GET':
        return Response('objectdetection server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': IMAGE_PROCESS_ERR
        }
        return make_response(jsonify(response), 400)

    
    return make_response(jsonify("test"), 200)

@app.route('/download/<filename>',methods=['POST'])
def download(filename):    
    return send_from_directory(MODEL_FOLDER, filename) 
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5555,debug=False)
