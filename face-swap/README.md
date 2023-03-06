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
## face swap Test

1. select face swap type
2. select source face image
3. select target face image
4. click "Predict" button

![image](https://user-images.githubusercontent.com/54097108/220100939-0f435b1e-6a26-4328-a980-9b4151b97324.png)

![image](https://user-images.githubusercontent.com/54097108/220101714-3453886a-4a99-41fd-bd47-d640281d4db4.png)

