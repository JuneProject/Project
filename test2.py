import cv2
import tensorflow
import keras
from PIL import Image ,ImageOps
import numpy as np
import requests
import time

webcam = cv2.VideoCapture(0)
success,image_bgr = webcam.read()
face_cascade = 'myhaar.xml'
face_classifier = cv2.CascadeClassifier(face_cascade)
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model("keras_Model.h5")
size = (224, 224)
while True:
    success,image_bgr = webcam.read()
    image_org = image_bgr.copy()
    image_bw = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(image_bw)

    for face in  faces:
        x,y,w,h = face
        cv2.rectangle(image_bgr,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite('GG_1.jpg',image_org[y:y+h,x:x+w])
        cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = cface_rgb
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        print(prediction)

        if prediction[0][0] > prediction[0][1]:
            cv2.putText(image_bgr,'no hat',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.rectangle(image_bgr ,(x,y),(x+w,y+h),(0,0,255),2)
            time.sleep(10)
            if face in  faces:
                def send_line_notify(token, message, image_path):
                    url = 'https://notify-api.line.me/api/notify'
                    headers = {'Authorization': f'Bearer {token}'}

                    payload = {'message': message}
                    r = requests.post(url, headers=headers, data=payload)

                    files = {'imageFile': open(image_path, 'rb')}
                    r = requests.post(url, headers=headers, data=payload, files=files)

                    if r.status_code == 200:
                        print('Message sent successfully.')
                    else:
                        print('Failed to send message.')

                access_token = 'cqoSp1kU7jtQCsu2lonGNdwHb1uixlzyTxaKJ6gNPdx'
                message = 'คนไม่ใส่หมวกกันน็อค'
                image_path = 'GG_1.jpg'
                send_line_notify(access_token, message, image_path)
            
        else :
            cv2.putText(image_bgr,'hat',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            cv2.rectangle(image_bgr ,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("detector", image_bgr)
    k = cv2.waitKey(1)
    if  k%256 == 27:
        break  
webcam.release()
cv2.destroyAllWindows() 
  

