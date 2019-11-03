import numpy as np
import math
import cv2
#from msvcrt import getch
import os

subjects = ["", "Positive Archie", "Not Positive Archie"]
#cap = cv2.VideoCapture('http://192.168.1.39:8080/video')
#cap = cv2.VideoCapture('http://192.168.0.47:8080/video')
cap = cv2.VideoCapture(0)
#fps = 120
#cap.set(cv2.CAP_PROP_FPS, fps)
#seconds = 1/60
#multiplier = fps * seconds
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
#count = 0

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
    
    if(len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y : y + w, x : x + h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x,y), (x + w, y+ h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img

if __name__ == '__main__':
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    frameID = 0
    
    while True:
        #frameID += 1
        ret, frame = cap.read()

        #if frameID % multiplier == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        captured_faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2,
                                                      minNeighbors = 5)
        for (x, y, w, h) in captured_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label, confidence = face_recognizer.predict(gray[y : y + w, x : x + h])
            subject_label = subjects[label]
            cv2.putText(frame, subject_label + " " + str(confidence), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) 

        #cv2.waitKey(1)
        cv2.imshow("Expression Recognition", frame)
        
        #cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #cv2.waitKey(25)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        #cv2.imshow("Expression Recognition", frame)
        
        #if(char == 13):
        #    count += 1
        #    cv2.imwrite("frame%d.jpg" % count, frame)
        #    print("Predicting image...")
        #    test_img = cv2.imread("frame%d.jpg" % count)
        #    predicted_img = predict(test_img)
        #    print("Prediction complete")
        #    cv2.imshow("Predicted Image", cv2.resize(predicted_img, (400,500)))

        #cv2.imwrite("frame.jpg", frame)
        #test_img = cv2.imread("frame.jpg")
        #predicted_img = predict(test_img)
        #cv2.imshow("Predicted Image", cv2.resize(predicted_img, (400,500)))
        
        #try:
        #    os.remove("frame.jpg")
        #except: pass

    cap.release()
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
