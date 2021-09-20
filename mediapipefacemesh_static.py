import cv2
import numpy as np
# mediapipe 0.8.7.0
import mediapipe as mp
import config
import tensorflow as tf
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import math
import pdb
resolution = 192
M_PI = 3.14159265
def normalize_radians(angle):
    return angle - 2 * M_PI * math.floor((angle - (-M_PI)) / (2 * M_PI))

def rot_vec(p, rotation):
    rx = p[0]*math.cos(rotation) - p[1]*math.sin(rotation)
    ry = p[0]*math.sin(rotation) + p[1]*math.cos(rotation)
    return [rx, ry]

interpreter = tf.lite.Interpreter(model_path='face_landmark.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'], [1, 192, 192, 3])
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)
# 考虑到facemesh官方代码默认是short range的，稍远的人脸检测不出，这里改为full range
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(np.flip(image,1), cv2.COLOR_BGR2RGB)
        image_copy = image.copy()
        results = face_detection.process(image)
        h, w = image.shape[:2]
        if results.detections:
            for detection in results.detections:
                # 直接使用face detection的结果来实现对齐操作，测试发现，当人脸旋转角度较大，检出不稳定，而且视频前后一致性较差(抖动较大)
                # 具体参考如下:
                # https://github.com/google/mediapipe/blob/v0.8.7/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
                # https://github.com/google/mediapipe/blob/v0.8.7/mediapipe/calculators/util/rect_transformation_calculator.cc
                face_info = {}
                score = detection.score
                face_info['score'] = score
                face_info['keypoints'] = []
                location = detection.location_data
                for keypoint in location.relative_keypoints:
                    p = (int(keypoint.x*w), int(keypoint.y*h))
                    face_info['keypoints'].append([keypoint.x*w, keypoint.y*h])

                bbox = location.relative_bounding_box
                tl = (int(bbox.xmin*w),int(bbox.ymin*h))
                br = (int(bbox.xmin*w + bbox.width*w),int(bbox.ymin*h + bbox.height*h))
                cv2.rectangle(image, tl, br, (0, 0, 255), 2)

                face_info['bbox'] = [bbox.xmin*w, bbox.ymin*h, bbox.xmin*w + bbox.width*w, bbox.ymin*h + bbox.height*h]

                target_angle = 0
                x0, y0 = face_info['keypoints'][0]
                x1, y1 = face_info['keypoints'][1]
                rotation = target_angle - math.atan2(-(y1 - y0), x1 - x0)
                face_info['rotation'] = normalize_radians(rotation)
                bbox_w = face_info['bbox'][2] - face_info['bbox'][0]
                bbox_h = face_info['bbox'][3] - face_info['bbox'][1]
                face_cx = face_info['bbox'][0] + bbox_w*0.5
                face_cy = face_info['bbox'][1] + bbox_h*0.5
                long_side = max(bbox_w, bbox_h)
                face_w = long_side*1.5
                face_h = long_side*1.5
                face_info['face_cx'] = face_cx
                face_info['face_cy'] = face_cy
                face_info['face_w'] = face_w
                face_info['face_h'] = face_h
                dx = face_w*0.5
                dy = face_h*0.5
                face_info['face_pos'] = np.array([[-dx, -dy],
                                                [ dx, -dy],
                                                [ dx, dy],
                                                [-dx, dy]])
                for i in range(4):
                    rot_point = rot_vec(face_info['face_pos'][i], face_info['rotation'])
                    face_info['face_pos'][i][0] = rot_point[0] + face_cx
                    face_info['face_pos'][i][1] = rot_point[1] + face_cy
                cv2.polylines(image, [face_info['face_pos'].astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                points = np.array([[-1,-1, 1, 1],
                                 [-1, 1,-1, 1]]).reshape(1,2,4)
                points = points*np.array(face_info['face_w']).reshape(-1, 1, 1)/2
                theta = np.array(face_info['rotation']).reshape(-1, 1, 1)
                R = np.concatenate((
                    np.concatenate((np.cos(theta),-np.sin(theta)),2),
                    np.concatenate((np.sin(theta),np.cos(theta)),2)
                ), 1)
                center = np.concatenate((np.array(face_cx).reshape(-1, 1, 1), np.array(face_cy).reshape(-1, 1, 1)), 1)
                points = R@points + center

                res = resolution
                points1 = np.array([[0, 0, res-1], [0, res-1, 0]], dtype='float32').T
                pts = points[0,:,:3].T.astype('float32')
                M = cv2.getAffineTransform(pts, points1)
                M_INV = cv2.invertAffineTransform(M).astype('float32')
                crop_im = cv2.warpAffine(image_copy, M, (res, res), borderValue=127.5)
                cv2.imshow('MediaPipe Crop', crop_im[:,:,::-1])
                crop_im = crop_im/127.5 - 1.0
                interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.expand_dims(crop_im, axis=0).astype(np.float32))
                interpreter.invoke()
                landmarks = interpreter.get_tensor(output_details[0]['index'])
                face_flag = interpreter.get_tensor(output_details[1]['index'])
                face_score = 1/(1 + np.exp(-face_flag)).reshape(1)
                print(face_score)
                landmarks = landmarks.reshape(468,3)
                landmarks[:,:2] = (M_INV[:,:2]@landmarks[:,:2].T  + M_INV[:,2:]).T
                '''
                for i in range(len(landmarks)):
                    p = landmarks[i]
                    cv2.circle(image,(int(p[0]), int(p[1])),1,(0,0,255),1)
                '''
                for i in range(len(config.face_tris)//3):
                    p1 = landmarks[config.face_tris[3*i]].astype(np.int32)[:2]
                    p2 = landmarks[config.face_tris[3*i+1]].astype(np.int32)[:2]
                    p3 = landmarks[config.face_tris[3*i+2]].astype(np.int32)[:2]
                    cv2.line(image,p1,p2, (255,255,255),1)
                    cv2.line(image, p1, p3, (255, 255, 255), 1)
                    cv2.line(image, p2, p3, (255, 255, 255), 1)
        cv2.imshow('MediaPipe FaceMesh', image[:,:,::-1])
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
