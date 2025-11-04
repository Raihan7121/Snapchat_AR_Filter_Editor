 # -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 01:37:35 2025

@author: Acer
"""

import cv2
import numpy as np
import math

def detect_eyes_manual(face_roi):
   
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
  
    h, w = gray.shape
    top_half = gray[0:h//2, :]

    eq = cv2.equalizeHist(top_half)
    
    blurred = cv2.medianBlur(eq, 5) 
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=w//4,
        param1=100,   
        param2=5,    
        minRadius=int(w*0.02),
        maxRadius=int(w*0.15)
    )
    
    eyes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (cx, cy, r) in circles[0, :2]:  
           
            x, y = int(cx) - int(r), int(cy) - int(r)
            w_box, h_box = 2*r, 2*r
            eyes.append((x, y, w_box, h_box))
 
            
            cv2.circle(face_roi, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(face_roi, (cx, cy), 2, (0, 0, 255), 3)
   
    cv2.imshow("Roi", face_roi)
    #cv2.imshow("Face buler", blurred)
    #cv2.imshow("Face eq", eq)
    cv2.waitKey(0)
    return eyes

def rotate_image_with_alpha(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w = int(h*sin + w*cos)
    new_h = int(h*cos + w*sin)
    M[0,2] += new_w/2 - w/2
    M[1,2] += new_h/2 - h/2
    print(M)
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0)
    )
    return rotated

def overlay_transparent(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    x1, y1 = max(x,0), max(y,0)
    x2, y2 = min(x+w, bg.shape[1]), min(y+h, bg.shape[0])
    overlay_x1, overlay_y1 = x1-x, y1-y
    overlay_x2, overlay_y2 = overlay_x1 + (x2-x1), overlay_y1 + (y2-y1)
    
    if overlay_x1 >= overlay_x2 or overlay_y1 >= overlay_y2:
        return bg
    
    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3:]/255.0
    bg[y1:y2, x1:x2] = (1-alpha)*bg[y1:y2, x1:x2] + alpha*overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
    return bg

def detect_face(img):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,48,80], dtype=np.uint8)
    upper = np.array([20,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No face detected.")
        return None
    else:
        face = max(contours, key=cv2.contourArea)
    print(contours)
    cv2.imshow("facemask", mask)
    cv2.waitKey(0)
    return face


def eye_des(face,img):
    x, y, w, h = cv2.boundingRect(face)
    face_roi = img[y:y+h, x:x+w]
    eyes = detect_eyes_manual(face_roi)

    eye_centers = []
    for (ex,ey,ew,eh) in eyes:
        cx = int(x) + int(ex) + int(ew//2)
        cy = int(y) + int(ey) + int(eh//2)
        eye_centers.append((cx,cy))
        cv2.circle(img, (cx, cy), 5, (0,0,255), -1)

    if len(eye_centers) == 2:
        left_eye, right_eye = sorted(eye_centers, key=lambda p: p[0])
        dx = int(right_eye[0]) - int(left_eye[0])
        dy = int(right_eye[1]) - int(left_eye[1])
        angle = -math.degrees(math.atan2(dy, dx))
        eye_mid = (
            int((left_eye[0] + right_eye[0]) // 2),
            int((left_eye[1] + right_eye[1]) // 2)
        )
        eye_distance = int(math.hypot(dx, dy))
        return eye_mid,eye_distance,angle
    else:
       print(f"⚠️ Detected {len(eye_centers)} eyes. Using face center as fallback.")
       # Fallback: use face center
       eye_mid = (x + w//2, y + h//2)
       eye_distance = w // 2
       angle = 0
       return eye_mid, eye_distance, angle
    return None,None,None

def apply_sunglass_mask(img,result=None):
    if result is None:
        result=img.copy()
    face = detect_face(img)

    if face is not None: 
            x, y, w, h = cv2.boundingRect(face)
            eye_mid,eye_distance,angle =eye_des(face,img)
           
            sunglass = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/Project/Project/filters/glasses.png", cv2.IMREAD_UNCHANGED)
            
           
            #scale = eye_distance / sunglass.shape[1] * 1.5
            scale = (w*0.7) / sunglass.shape[1] * 1.5
            new_w = int(sunglass.shape[1]*scale)
            new_h = int(sunglass.shape[0]*scale)
            sunglass = cv2.resize(sunglass, (new_w, new_h))

            rotated_mask = rotate_image_with_alpha(sunglass, angle)
            
            
            x_offset = eye_mid[0] - rotated_mask.shape[1]//2
            y_offset = eye_mid[1] - rotated_mask.shape[0]//2
            result = overlay_transparent(result, rotated_mask, x_offset, y_offset)

            cv2.imshow("Detected Eyes", img)
            cv2.imshow("Sunglasses Filter", result)
            cv2.waitKey(0)
            return result
        
def apply_hat_mask(img,result=None):
    if result is None:
        result=img.copy()
    face = detect_face(img)

    if face is not None: 
            x, y, w, h = cv2.boundingRect(face)
            eye_mid,eye_distance,angle =eye_des(face,img)
            
            
            hat = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/Project/Project/filters/hat.png", cv2.IMREAD_UNCHANGED)
            
            
            scale_hat = w / hat.shape[1] * 1.3
            new_w_hat = int(hat.shape[1] * scale_hat)
            new_h_hat = int(hat.shape[0] * scale_hat)
            hat = cv2.resize(hat, (new_w_hat, new_h_hat))
            
      
            rotated_hat = rotate_image_with_alpha(hat, angle)
            
            hat_shift = int(h * 0.5)   # how far above eyes
            dx = int(hat_shift * math.sin(math.radians(angle)))
            dy = int(hat_shift * math.cos(math.radians(angle)))
            
            
            hat_center_x = eye_mid[0] - dx
            hat_center_y = eye_mid[1] - dy
            
    
            x_offset_hat = hat_center_x - rotated_hat.shape[1] // 2
            y_offset_hat = hat_center_y - rotated_hat.shape[0] // 2
            
           
            result = overlay_transparent(result, rotated_hat, x_offset_hat, y_offset_hat)
            

            cv2.imshow("Hat Filter", result)
            cv2.waitKey(0)
            return result
def apply_nose_mask(img,result=None):
    if result is None:
        result=img.copy()
    face = detect_face(img)

    if face is not None: 
            x, y, w, h = cv2.boundingRect(face)
            eye_mid,eye_distance,angle =eye_des(face,img)
            
            nose_mask = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/Project/Project/filters/nosefilter2.png", cv2.IMREAD_UNCHANGED)
            if nose_mask is None:
                print("Nose mask not found")
            else:

                nose_shift = int(h * 0.18)
            
                
                dx = int(nose_shift * math.sin(math.radians(angle)))
                dy = int(nose_shift * math.cos(math.radians(angle)))
            
      
                nose_center_x = eye_mid[0] + dx
                nose_center_y = eye_mid[1] + dy
            
       
                scale_nose = (w / nose_mask.shape[1]) * 0.30
                new_w_nose = max(1, int(nose_mask.shape[1] * scale_nose))
                new_h_nose = max(1, int(nose_mask.shape[0] * scale_nose))
                nose_resized = cv2.resize(nose_mask, (new_w_nose, new_h_nose), interpolation=cv2.INTER_AREA)
       
                rotated_nose = rotate_image_with_alpha(nose_resized, angle)
            

                x_offset_nose = nose_center_x - rotated_nose.shape[1] // 2
                y_offset_nose = nose_center_y - rotated_nose.shape[0] // 2
            
       
                result = overlay_transparent(result, rotated_nose, x_offset_nose, y_offset_nose)
            
       
                cv2.circle(result, (nose_center_x, nose_center_y), 3, (0, 255, 255), -1)
            
            cv2.imshow("Nose Filter (orientation-aware)", result)
            cv2.waitKey(0)
            return result
def apply_mustache_mask(img,result=None):
    if result is None:
        result=img.copy()
    face = detect_face(img)

    if face is not None: 
            x, y, w, h = cv2.boundingRect(face)
            eye_mid,eye_distance,angle =eye_des(face,img)
            
            nose_shift = int(h * 0.18)
        
           
            dx = int(nose_shift * math.sin(math.radians(angle)))
            dy = int(nose_shift * math.cos(math.radians(angle)))
        
      
            nose_center_x = eye_mid[0] + dx
            nose_center_y = eye_mid[1] + dy
            
            mustache = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/Project/Project/filters/mustache.png", cv2.IMREAD_UNCHANGED)
            if mustache is None:
                print("Mustache mask not found")
            else:
                # shift downward from nose tip (~0.07*h works well, tune if needed)
                mustache_shift = int(h * 0.07)
            
                dx = int(mustache_shift * math.sin(math.radians(angle)))
                dy = int(mustache_shift * math.cos(math.radians(angle)))
            
                mustache_center_x = nose_center_x + dx
                mustache_center_y = nose_center_y + dy
            
       
                scale_mustache = (w / mustache.shape[1]) * 0.45   
                new_w_mustache = max(1, int(mustache.shape[1] * scale_mustache))
                new_h_mustache = max(1, int(mustache.shape[0] * scale_mustache))
                mustache_resized = cv2.resize(mustache, (new_w_mustache, new_h_mustache), interpolation=cv2.INTER_AREA)
            
                
                rotated_mustache = rotate_image_with_alpha(mustache_resized, angle)
            
                
                x_offset_mustache = mustache_center_x - rotated_mustache.shape[1] // 2
                y_offset_mustache = mustache_center_y - rotated_mustache.shape[0] // 2
            
                result = overlay_transparent(result, rotated_mustache, x_offset_mustache, y_offset_mustache)
            
               
                cv2.circle(result, (mustache_center_x, mustache_center_y), 3, (255, 0, 255), -1)
            cv2.imshow("Nose Filter (orientation-aware) moustache", result)
            cv2.waitKey(0)
            return result
        
def apply_dog_mask(img,result=None):
    if result is None:
        result=img.copy()
    face = detect_face(img)

    if face is not None: 
            x, y, w, h = cv2.boundingRect(face)
            eye_mid,eye_distance,angle =eye_des(face,img)
            
            
            dog_mask = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/Project/Project/filters/dog.png", cv2.IMREAD_UNCHANGED)
            if dog_mask is None:
                print("Dog mask not found")
            else:
       
                scale_dog = (w / dog_mask.shape[1]) * 1.2   # 1.2 to slightly overshoot face
                new_w_dog = max(1, int(dog_mask.shape[1] * scale_dog))
                new_h_dog = max(1, int(dog_mask.shape[0] * scale_dog * 0.7))
                dog_resized = cv2.resize(dog_mask, (new_w_dog, new_h_dog), interpolation=cv2.INTER_AREA)
            

                rotated_dog = rotate_image_with_alpha(dog_resized, angle)
            
           
                face_center_x = x + w // 2
                face_center_y = y + h // 2 - int(0.08 * h)
            
                x_offset_dog = face_center_x - rotated_dog.shape[1] // 2
                y_offset_dog = face_center_y - rotated_dog.shape[0] // 2
            
                result = overlay_transparent(result, rotated_dog, x_offset_dog, y_offset_dog)
            
              
                #cv2.circle(result, (face_center_x, face_center_y), 4, (0, 255, 0), -1)
                cv2.imshow("Dog Mask Applied", result)
            cv2.waitKey(0)
            return result
 
img = cv2.imread(r"C:/Users/Acer/Desktop/imagelab/Project/Project/test/t2.jpg")
img = cv2.resize(img, (500, int(img.shape[0]*500/img.shape[1])))
result=img.copy()
apply_sunglass_mask(img)
apply_hat_mask(img)
apply_nose_mask(img, result)
apply_mustache_mask(img, result)
apply_dog_mask(img, result)

cv2.destroyAllWindows()


