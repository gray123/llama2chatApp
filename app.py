import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template
from flask_socketio import SocketIO, send
from huggingface_hub import login

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

SECRET_KEY = os.environ.get("SECRET_KEY")

login(token=SECRET_KEY)

timeStart = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print("Load model time: ", -timeStart + time.time())

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
      inputs = tokenizer.encode(
        question,
        return_tensors="pt"
      )

      outputs = model.generate(
        inputs,
        max_new_tokens=int(50),
      )

      answers = tokenizer.decode(outputs[0]) + " bot gen time {:.3}sec".format(time.time()-timeStart)
      send(answers, broadcast=True)
     

@app.route('/')
def index():
  return render_template("index.html")

if __name__ == "__main__":
  socketio.run(app, host="localhost")