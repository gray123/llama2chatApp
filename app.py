import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template
from flask_socketio import SocketIO, send
from huggingface_hub import login
from langchain_mdl import durham_langchain

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
SECRET_KEY = os.environ.get("SECRET_KEY")

timeStart = time.time()
gpt = durham_langchain()
gpt.init_gpt_mdl()
gpt.init_vdb()
print("Load model time: {:.3} sec".format(time.time()-timeStart))

app = Flask(__name__)
app.config['SECRET'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('message')
def handle_message(message):
  print("Received message: " + message)
  if message != "User connected!":
    send(message, broadcast=True)
    if message.find("@bot") != -1:
      question = message.split("@bot",1)[1]

      timeStart = time.time()
      answers = gpt.infer(question) + "\n\n bot gen time {:.3} sec".format(time.time()-timeStart)
      send(answers, broadcast=True)
     

@app.route('/')
def index():
  return render_template("index.html")

if __name__ == "__main__":
  socketio.run(app, host="localhost")
