import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('./face_detector/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

labels_dict={0:'YES',1:'NO'}
color_dict={0:(0,255,0),1:(0,0,255)}
model=load_model('MaskNet.model')

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img=frame[y:y+h,x:x+w]
        resize=cv2.resize(face_img,(224,224))
        resize=resize/255.0
        reshape=np.reshape(resize,(1,224,224,3))
        result=model.predict(reshape)
        #print(result)
        label=np.argmax(result[0])
        #print(label)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,225,225),2)
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()