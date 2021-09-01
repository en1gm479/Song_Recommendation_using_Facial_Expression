import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json


harr_face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
json_file = open('model/data_model.json', 'r') 
model = model_from_json(json_file.read())
model.load_weights("model/data_model.h5")
json_file.close()

emotion_labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#cap = cv2.VideoCapture(0)
frame = cv2.imread("first.jpg")  #name of the image to test


def song_recommendation(dmi):
    mood = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', ]
    suggestions = {0: ['ACHYUTAM KESHAVAM KRISHNA DAMODARAM'],
                1: ['Maula Mere Le Le Meri Jaan - Krishna | Salim'],
                2: ['The Score - The Fear'],
                3: ['DJ Wale Babu - Squirrel'],
                4: ['OneRepublic - No Vacancy'],
                5: ['Zindagi Ki Yahi Reet Hai - Soumitra Dev Burman'],
                6: ['As We Fall | Varus - League of Legends'], }
    suggestions_links = {0: ['https://youtu.be/pzzPowh241o'],
                        1: ['https://youtu.be/i_FmOdPF96E'],
                        2: ['https://youtu.be/K5U7b_E14cE'],
                        3: ['https://youtu.be/xuS_lJ2Dh6k'],
                        4: ['https://youtu.be/qXiuVQ-GgA4'],
                        5: ['https://youtu.be/BqO6EOIiYrk'],
                        6: ['https://youtu.be/vzNcSvKCOyA'], }

    # displaying songs as image
    blank = np.full((400, 800, 3), (125, 168, 50), dtype='uint8')
    cv2.putText(blank, str('For your detected mood : ' + mood[dmi]), (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 50, 168), thickness=2)
    cv2.putText(blank, str('We recommend following song:'), (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 162, 200), thickness=2)
    cv2.putText(blank, str('Song Name : ' + suggestions[dmi][0]), (100, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 161, 242), thickness=2)
    cv2.putText(blank, str('Link : ' + suggestions_links[dmi][0]), (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 161, 242), thickness=2)
    #    To change color relace this           B ,  G ,  R

    # songs in terminal
    print(f'\nFor your detected mood : {mood[dmi]}')
    print(f'We recommend following song:')
    print(f'Song Name: {suggestions[dmi][0]}')
    print(f'Link : {suggestions_links[dmi][0]}', '\n')
    cv2.imshow('Recommended Songs', blank)
    cv2.waitKey(0)


# while True:
    #ret,frame = cap.read()
    # frame = cv2.resize(frame, (540,480))
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
no_of_faces = harr_face_classifier.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors=4)

for (x,y,w,h) in no_of_faces:
    cv2.rectangle(frame,(x,y),(x+w+10,y+h+10),(255,127,0),2)
    crop_gray = gray[y:y+h-10,x:x+w-10]
    roi_gray = cv2.resize(crop_gray,(300,300))
    cropped_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1), 0)

    preds = model.predict(cropped_gray)
    index_pred = int(np.argmax(preds))
    label = emotion_labels[index_pred]
    print(label)
    label_position =(x+5,y)
    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    
    
cv2.imshow('Mood Detector',frame)
song_recommendation(index_pred)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
cv2.waitKey(0)
# cap.release()
cv2.destroyAllWindows()


