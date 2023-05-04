import threading
import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union
from modelhub import onnx as onnx_models
from modelhub import DFLive
from xlib.onnxruntime import get_available_devices_info, ORTDeviceInfo
import cv2
from xlib.image import ImageProcessor
from xlib.face import FRect, FPose, FLandmarks2D, ELandmarks2D
from . import backend

from xlib import cv as lib_cv
from xlib import avecl as lib_cl
import multiprocessing

class CommonBuff:
    def __init__(self, img : np.ndarray, copy=False):
        self._frame_image : np.ndarray = img

        self._fdetect = False
        self._frect = None
        self._face_align_img = None
        self._uni_mat = None

        self._fconvert = False
        self._celeb_face = None
        self._celeb_face_mask_img = None
        self._face_align_mask_img = None

        self._merged = False
        self._merged_frame = None

class CommonList:
    def __init__(self):
        self._lock = multiprocessing.Lock()
        self._max = 20
        self._list = []
        self._start = False

    def canclose(self) -> bool:
        lock = self._lock
        lock.acquire()
        res = len(self._list) == 0 and self._start == True
        lock.release()
        return res

    def canaddItem(self):
        lock = self._lock
        lock.acquire()
        res = len(self._list) < self._max
        lock.release()
        return res

    def addItem(self, img : np.ndarray):
        lock = self._lock
        lock.acquire()
        newitem = CommonBuff(img)
        self._list.append(newitem)
        lock.release()


class DeepFaceCSApp:
    def __init__(self, gpuid, modelpath):
        devlist = get_available_devices_info(False)
        self.deviceid = devlist[gpuid]
        self.modelpath = modelpath
        self.faceDetector = onnx_models.YoloV5Face(self.deviceid)
        self.faceLandmarker = onnx_models.FaceMesh(self.deviceid)
        self.faceSwapper = DFLive.DFMModel(modelpath, self.deviceid)
        
        cldevlist = lib_cl.get_available_devices_info()
        dev = lib_cl.get_device(cldevlist[gpuid])
        dev.set_target_memory_usage(mb=512)
        lib_cl.set_default_device(dev)

        self.max_faces = 1

    def _merge_on_gpu(self, frame_image, face_resolution, face_align_img, face_align_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression ):
               
        _n_mask_multiply_op_text = [ f"float X = {'*'.join([f'(((float)I{i}) / 255.0)' for i in range(n)])}; O = (X <= 0.5 ? 0 : 1);" for n in range(5) ]

        masks = []
        
        masks.append( lib_cl.Tensor.from_value(face_align_mask_img) )
        masks.append( lib_cl.Tensor.from_value(face_swap_mask_img) )

        masks_count = len(masks)
        if masks_count == 0:
            face_mask_t = lib_cl.Tensor(shape=(face_resolution, face_resolution), dtype=np.float32, initializer=lib_cl.InitConst(1.0))
        else:
            face_mask_t = lib_cl.any_wise(_n_mask_multiply_op_text[masks_count], *masks, dtype=np.uint8).transpose( (2,0,1) )

        face_mask_t = lib_cl.binary_morph(face_mask_t, 5, 25, fade_to_border=True, dtype=np.float32)
        face_swap_img_t  = lib_cl.Tensor.from_value(face_swap_img ).transpose( (2,0,1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)

        
        face_align_img_t = lib_cl.Tensor.from_value(face_align_img).transpose( (2,0,1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)
        face_swap_img_t = lib_cl.rct(face_swap_img_t, face_align_img_t, target_mask_t=face_mask_t, source_mask_t=face_mask_t)

        frame_face_mask_t     = lib_cl.remap_np_affine(face_mask_t,     aligned_to_source_uni_mat, interpolation=lib_cl.EInterpolation.LINEAR, output_size=(frame_height, frame_width), post_op_text='O = (O <= (1.0/255.0) ? 0.0 : O > 1.0 ? 1.0 : O);' )
        frame_face_swap_img_t = lib_cl.remap_np_affine(face_swap_img_t, aligned_to_source_uni_mat, interpolation=lib_cl.EInterpolation.LINEAR, output_size=(frame_height, frame_width), post_op_text='O = clamp(O, 0.0, 1.0);' )

        #frame_image_t = lib_cl.Tensor.from_value(frame_image).transpose( (2,0,1), op_text='O = ((float)I) / 255.0' if frame_image.dtype == np.uint8 else None,
        #                                                                          dtype=np.float32 if frame_image.dtype == np.uint8 else None)
        frame_image_t = lib_cl.Tensor.from_value(frame_image).transpose( (2,0,1), op_text='O = ((float)I) / 255.0', dtype=np.float32)

        opacity = 1.0
        if opacity == 1.0:
            frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I2*I1', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, dtype=np.float32)
        else:
            frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I0*I1*(1.0-I3) + I2*I1*I3', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, np.float32(opacity), dtype=np.float32)
        return frame_final_t.transpose( (1,2,0) ).np()
    
    def convert(self, sourcepath, targetpath, swaptype: int):
        
        #get avatar image
        if swaptype > 0:
            print('convert type:', swaptype)
            avatartmppng = cv2.imread('avatar.png', cv2.IMREAD_UNCHANGED)
            if avatartmppng is None:
                print("Error opening video file")
                return False
            _, mask = cv2.threshold(avatartmppng[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            pts = cv2.findNonZero(mask)
            x,y,w,h = cv2.boundingRect(pts)
            avatarpng = avatartmppng[y:y+h, x:x+w]
            
        #set detect status
        vreader = cv2.VideoCapture(sourcepath)
        if (vreader.isOpened()== False):
            print("Error opening video file")
            return False
        width = int(vreader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vreader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vreader.get(cv2.CAP_PROP_FPS))
        vwriter= cv2.VideoWriter(targetpath, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))        
        #for debug
        frameidx = 0
        face_resolution = 224
        while(vreader.isOpened()):
            ret, frame_image = vreader.read()
            if ret == True:
                # detect face
                rects = []
                
                _,H,W,_ = ImageProcessor(frame_image).get_dims()
                rects = self.faceDetector.extract(frame_image, 0.5, 480)[0]
                rects = [ FRect.from_ltrb( (l/W, t/H, r/W, b/H) ) for l,t,r,b in rects ]
                rects = FRect.sort_by_area_size(rects)

                merged_frame = frame_image
                
                if len(rects) != 0:
                    if swaptype > 0:
                        
                        for face_id, face_urect in enumerate(rects):
                            face_image, face_uni_mat = face_urect.cut(frame_image, 1.4, 192)
                            _,fH,fW,_ = ImageProcessor(face_image).get_dims()
                            lmrks = self.faceLandmarker.extract(face_image)[0]
                            face_pose = FPose.from_3D_468_landmarks(lmrks)
                            lmrks = lmrks[...,0:2] / (fW,fH)
                            face_ulmrks = FLandmarks2D.create (ELandmarks2D.L468, lmrks)
                            face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
                            
                            facerect = face_ulmrks.get_FRect()
                            head_yaw = face_pose.as_radians()[1]

                            savatar = avatarpng
                            if(head_yaw > 0.0):
                                savatar = cv2.flip(savatar, 1)
                            
                            (il,ir,it,ib) = face_urect.as_ltrb_bbox(((W,H))).astype(int)
                            #(il,ir,it,ib) = facerect.as_ltrb_bbox(((W,H))).astype(int)
                            (l,t,r,b) = (max(0,il),max(0,it),min(ir,W),min(ib,H))
                            
                            s_img = cv2.resize(savatar, dsize=(r-l, b-t), interpolation=cv2.INTER_LINEAR)
                            alpha_s = s_img[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            
                            for c in range(0, 3):
                                merged_frame[t:b, l:r, c] = (alpha_s * s_img[:, :, c] +
                                                        alpha_l * merged_frame[t:b, l:r, c])                            
                        
                    else:
                        if self.max_faces != 0 and len(rects) > self.max_faces:
                            rects = rects[:self.max_faces]
                        for face_id, face_urect in enumerate(rects):
                            face_image, face_uni_mat = face_urect.cut(frame_image, 1.4, 192)
                            _,fH,fW,_ = ImageProcessor(face_image).get_dims()
                            lmrks = self.faceLandmarker.extract(face_image)[0]
                            #face_pose = FPose.from_3D_468_landmarks(lmrks)
                            lmrks = lmrks[...,0:2] / (fW,fH)
                            face_ulmrks = FLandmarks2D.create (ELandmarks2D.L468, lmrks)
                            face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
                            #face align & resolution
                            #head_yaw = face_pose.as_radians()[1]
                            face_align_img, uni_mat = face_ulmrks.cut(frame_image, 2.2 , face_resolution,
                                                                            exclude_moving_parts=True,
                                                                            head_yaw=None,
                                                                            x_offset= 0,
                                                                            y_offset= -0.08,
                                                                            freeze_z_rotation= False)
                            face_height, face_width = face_align_img.shape[:2]
                            frame_height, frame_width = merged_frame.shape[:2]
                            
                            image_to_align_uni_mat = uni_mat
                            aligned_to_source_uni_mat = image_to_align_uni_mat.invert()
                            aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat (face_width, face_height, frame_width, frame_height)
                            #face_align_ulmrks = face_ulmrks.transform(uni_mat)
                            #face_align_lmrks_mask_img = face_align_ulmrks.get_convexhull_mask( face_align_img.shape[:2], color=(255,), dtype=np.uint8)
                            
                            fai_ip = ImageProcessor(face_align_img)
                            face_align_image = fai_ip.get_image('HWC')
                            celeb_face, celeb_face_mask_img, face_align_mask_img = self.faceSwapper.convert(face_align_image, morph_factor=0.75)
                            celeb_face, celeb_face_mask_img, face_align_mask_img = celeb_face[0], celeb_face_mask_img[0], face_align_mask_img[0]

                            
                            merged_frame = self._merge_on_gpu(merged_frame, face_resolution, face_align_img, face_align_mask_img, 
                                                            celeb_face, celeb_face_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, True )

                            merged_frame = ImageProcessor(merged_frame, copy=True).to_uint8().get_image('HWC')
                            

                            '''
                            if frameidx > 0:
                                face_align_image = ImageProcessor(face_align_image, copy=True).to_uint8().get_image('HWC')
                                file_ext, cv_args = '.jpg', [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                                filename = f'{frameidx:06}'
                                cv2.imwrite("F:/userdata/images1-in/" + (filename+file_ext), face_align_image)
                            '''

                    frameidx += 1
                
                vwriter.write(merged_frame)
            else:
                break
        vwriter.release()
        vreader.release()
        return True