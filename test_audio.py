from flask import Flask, render_template, request
import speech_recognition as sr

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('listen_page.html')

@app.route('/process', methods=['POST'])
def process():
    audio_file = request.files['audio_data']
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

if __name__ == '__main__':
    app.run(debug=True)
