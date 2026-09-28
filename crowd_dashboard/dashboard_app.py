import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, session, url_for
import requests

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Processing server URL
PROCESSING_SERVER = os.getenv("STAMPEDE_PROCESSING_SERVER", "http://127.0.0.1:5000")

# Demo login
USERNAME = os.getenv("STAMPEDE_USERNAME")
PASSWORD = os.getenv("STAMPEDE_PASSWORD")


@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']

        if USERNAME and PASSWORD and user == USERNAME and pw == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))

        return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect('/')

    return render_template("dashboard.html",
                           stream_url=f"{PROCESSING_SERVER}/video",
                           metrics_url=f"{PROCESSING_SERVER}/metrics")


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


if __name__ == "__main__":
    app.run(port=7000)