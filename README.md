# adventure-stream
We are enabling viewers of a livestream to become part of the content creation. a video pipeline that allows user to interact with content from real world live streams through AI

# face swap

## Installation
1. Install Nvidia drivers on your machine.
2. Install Pytorch , please refer here(https://pytorch.org/get-started/locally/).
3. Clone git repo
```
git clone  https://github.com/liveinteract/adventure-stream
```
4. Install python package by using following command
```
pip install -r requirements.txt
```
5. Download face pose model
```
cd models
gdown https://drive.google.com/uc?id=12BtinAcMUNA-do11QobmZ_tDQ7Qxv9Kk -O shape_predictor_68_face_landmarks.dat
```
6. run web server and connect "http://127.0.0.1:5555/" with web browser
```
cd ..
python run.py
```
## Test
