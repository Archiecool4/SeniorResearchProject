from tkinter import font, ttk, Canvas
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from RealtimeExpressions import *

last_frame = np.zeros((480, 640, 3), dtype = np.uint8)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.0.47:8080/video')
cap.set(3, 640)
cap.set(4, 480)

def quit_(root):
    root.destroy()

def func(root):
    print("What's up?")

def show_frame():
    image = predict_face()
    last_frame = image.copy()
    cv2image = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image = imgtk)
    lmain.after(10, show_frame)

def predict_face():
    ref, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    captured_faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2,
                                                   minNeighbors = 5)
    for(x, y, w, h) in captured_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label, confidence = face_recognizer.predict(gray[y : y + w, x : x + h])
        subject_label = subjects[label]
        cv2.putText(image, subject_label, (x, y), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 2)

    return image

if __name__ == '__main__':
    #faces, labels = prepare_training_data("training-data")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #face_recognizer.train(faces, np.array(labels))
    #face_recognizer.save('archieexpressions.yml')
    face_recognizer.read('archieexpressions.yml')
    
    root = tk.Tk()
    root.title("2018 Senior Research Project Prototype")
    
    menubar = tk.Menu(root)
    
    filemenu = tk.Menu(menubar, tearoff = 0)
    filemenu.add_command(label = "Open", command = lambda: func(root))
    filemenu.add_command(label = "Save", command = lambda: func(root))
    filemenu.add_separator()
    filemenu.add_command(label = "Exit", command = lambda: quit_(root))
    menubar.add_cascade(label = "File", menu = filemenu)

    aboutmenu = tk.Menu(menubar, tearoff = 0)
    aboutmenu.add_command(label = "Exprec", command = lambda: func(root))
    menubar.add_cascade(label = "About", menu = aboutmenu)

    root.config(menu = menubar)
    
    nb = ttk.Notebook(root)
    page1 = ttk.Frame(nb)

    lmain = tk.Label(page1)
    lmain.grid(column = 0, rowspan = 3, padx = 5, pady = 5)
    show_frame()
    
    arial = font.Font(family = "Arial", size = 15)

    quit_button = tk.Button(page1, text = "Quit", bg = "red3", fg = "white",
                            command = lambda: quit_(root))
    quit_button['font'] = arial
    quit_button.grid(column = 1, row = 1, padx = 5, pady = 5)

    page2 = ttk.Frame(nb)

    canvas = Canvas(page2)
    canvas.create_oval(225, 100, 525, 400, outline = "#f11", fill = "#1f1", width = 2)
    canvas.create_arc(225, 100, 525, 400, start = 0, extent = 120, outline = "#f11",
                      fill = "gray", width = 2)
    canvas.create_text(350, 300, font = "Times",
                       text = "Happy: 67%")
    canvas.create_text(425, 175, font = "Times",
                       text = "Sad: 33%")
    canvas.pack(expand = 1, fill = "both")
    
    nb.add(page1, text = 'Expression Recognition')
    nb.add(page2, text = 'Statistics')
    nb.pack(expand = 1, fill = "both")

    root.mainloop()
    cap.release()
