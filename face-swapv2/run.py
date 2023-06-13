import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, Response, jsonify, make_response
from werkzeug.utils import secure_filename
from apps.DeepFaceLive.DeepFaceCSApp import DeepFaceCSApp

app=Flask(__name__, template_folder='templates', static_folder='videodata')


IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
INVALID_REQUEST_ERR = 231

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

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

# run cnn filter to speed up processing time
swapper.convert("test.flv","test.flv_swapped1.flv",0, 2)
swapper.convertbypacket("test.data", 1280, 720, "65291,8041,3338,3114,11556,", "test.flv_swapped.flv", 0, 2)

#for decoding
framewidth = 1920
frameheight = 1080

out_name = ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getmodels():
    models = []
    defaultmodel = MODEL_NAME
    models.append(defaultmodel.replace(".dfm", ""))
    for (_, _, file) in os.walk(MODEL_FOLDER):
        for f in file:
            if '.dfm' in f and f != MODEL_NAME:
                models.append(f.replace(".dfm", ""))

    return models


@app.route('/')
def upload_form():
    models = getmodels()
    return render_template('upload.html', models = models, out_name = out_name)

@app.route('/faceswap', methods=['POST'])
def faceswapbypath():
    payloads = request.json
    if 'filepath' not in payloads or 'pktinfo' not in payloads:
        response = {
            'error': '"filepath" is missing in the request.'
        }
        return make_response(jsonify(response), 400)

    srcpath = payloads['filepath']
    pktinfo = payloads['pktinfo']
    framewdith = int(payloads['w'])
    framwheight = int(payloads['h'])
    dstpath = srcpath + "_swapped.flv"
    print(srcpath, dstpath, pktinfo)
    
    #res = swapper.convert(srcpath,dstpath,0,2)
    res = swapper.convertbypacket(srcpath, framewdith, framwheight, pktinfo, dstpath, 0, 2)

    if res == True:
        statuscode = 200
    else:
        statuscode = 400
    response = {
            'respath': f'"{dstpath}"'
        }
    return make_response(jsonify(response), statuscode)

@app.route('/', methods=['POST'])
def upload_file():
    global MODEL_FOLDER, MODEL_NAME, swapper
    if request.method == 'POST':
        type = str(request.form.get('type_select'))

        if 'source' not in request.files:
            flash('No source Video File')
            return redirect(request.url)

        if not os.path.isdir(uploadpath):
            os.mkdir(uploadpath)

        model = str(request.form.get('model_select')) + ".dfm"
        print(model)
        if type == "face" and model != MODEL_NAME:
            MODEL_NAME = model
            modelpath = os.path.join(MODEL_FOLDER, MODEL_NAME)
            swapper = DeepFaceCSApp(0, modelpath)

        imgsfile = request.files.getlist('source')[0]
        sourcepath = ""
        targetpath = ""
        filename = ""
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

        models = getmodels()
        return render_template('upload.html', models = models, out_name = targetname)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5555,debug=False)
