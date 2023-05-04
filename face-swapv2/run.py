import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, Response, jsonify, make_response
from werkzeug.utils import secure_filename
from apps.DeepFaceLive.DeepFaceCSApp import DeepFaceCSApp

app=Flask(__name__, template_folder='templates', static_folder='videodata')


IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
INVALID_REQUEST_ERR = 231

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'videodata')
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['mp4', 'mjepg'])
# trained model
MODEL_FOLDER = os.path.join(path, 'models')

MODEL_NAME = "Jackie_Chan.dfm"
modelpath = os.path.join(MODEL_FOLDER, MODEL_NAME)

swapper = DeepFaceCSApp(0, modelpath)
uploadpath = UPLOAD_FOLDER

out_name = ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getmodels():
    models = []
    for (_, _, file) in os.walk(MODEL_FOLDER):
        for f in file:
            if '.dfm' in f:
                models.append(f.replace(".dfm", ""))
    return models


@app.route('/')
def upload_form():
    return render_template('upload.html', out_name = out_name)


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        type = str(request.form.get('comp_select'))

        if 'source' not in request.files:
            flash('No source Video File')
            return redirect(request.url)

        if not os.path.isdir(uploadpath):
            os.mkdir(uploadpath)

        imgsfile = request.files.getlist('source')[0]
        sourcepath = ""
        targetpath = ""
        if imgsfile and allowed_file(imgsfile.filename):
            filename = secure_filename(imgsfile.filename)
            sourcepath = os.path.join(uploadpath, filename)
            imgsfile.save(sourcepath)

        targetname = filename + "_swapped.mp4"
        targetpath = os.path.join(uploadpath, targetname)

        flash('The file is uploaded successfully.')

        if type == "face":
           swapper.convert(sourcepath,targetpath,0)
        else:
           swapper.convert(sourcepath,targetpath,1)

        return render_template('upload.html', out_name = targetname)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5555,debug=False)
