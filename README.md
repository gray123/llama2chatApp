1. Go to huggingface.co to create an account and get token_key
2. create .env under the project dir with one line "SECRET_KEY=YOUR_TOKEN_KEY_ON_HUGGINEFACE"
3. install conda
4. conda create -n py_3_11_lamma2_run python=3.11 -y
5. conda activate py_3_11_lamma2_run
6. pip install transformers torch accelerate flask flask_socketio huggingface_hub
7. python app.py
8. open your browser with "localhost:5000"
9. During chat, anytime, "@bot" 4 char key word will send the string after this key word to Llamma2 local model
10. Response from the model will be send to the screen