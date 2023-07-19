from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
import speech_recognition as sr
import os

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+" * 50, "loadin gmmodel")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choose_singer', methods=["POST"])
def choose_singer():
    info['language'] = request.form['language']
    print(info)
    return render_template('choose_singer.html', data=info['language'])


@app.route('/homepage', methods=["POST"])
def choose_type():
    info['singer'] = request.form['singer']
    return render_template("homepage.html")


@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    print(info)
    found = False

    cap = cv2.VideoCapture(0)
    while not (found):
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))

    roi = roi / 255.0

    roi = np.reshape(roi, (1, 48, 48, 1))

    prediction = model.predict(roi)

    print(prediction)

    prediction = np.argmax(prediction)
    prediction = label_map[prediction]

    cap.release()

    link = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction}+{info['language']}+song"
    webbrowser.open(link)

    return render_template("emotion_detect.html", data=prediction, link=link)


@app.route('/listen_page', methods=['POST'])
def listen_audio():
    form = """
            <form action="/" method="POST">
                <input type="submit" value="Speak">
            </form>
        """

    return render_template("listen_page.html", form=form)


@app.route('/process', methods=['POST'])
def process():
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        audio = recognizer.listen(mic)

    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "I couldn't understand what you said."
    print(text)


    link = f"https://www.youtube.com/results?search_query={info['singer']}+{text}+{info['language']}+song"
    webbrowser.open(link)

    return render_template("index.html", data=text, link=link)


if __name__ == "__main__":
    app.run(debug=True)