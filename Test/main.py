import speech_recognition as sr
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    form = """
        <form action="/" method="POST">
            <input type="submit" value="Speak">
        </form>
    """
    return render_template("index.html", form=form)

@app.route("/", methods=["POST"])
def transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        audio = recognizer.listen(mic)

    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "I couldn't understand what you said."
    print(text)
    return render_template("index.html", text=text)

if __name__ == "__main__":
    app.run(debug=True)
