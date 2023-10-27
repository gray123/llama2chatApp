# Conda env setup
  - Go to https://www.anaconda.com/download to install conda
  - Conda env setup inside conda terminal
  - "conda create -n py_3_11_chatBot_run python=3.11 -y"
  - "conda activate py_3_11_chatBot_run"
  - "conda install -c conda-forge faiss-gpu cudatoolkit=11.8"

# Install python dependency in conda env
  - "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
  - "pip install transformers==4.34.1"
  - "pip install langchain xformers sentence-transformers python_dotenv atlassian-python-api markdownify flask flask_socketio"

# Run the app
  - Obtain huggingface token to download and use Llama2 model
    - Go to huggingface.co to create an account and get token_key
	- create .env under the project dir with one line: SECRET_KEY=YOUR_TOKEN_KEY_ON_HUGGINEFACE
  - "python app.py"

# Have fun with it
  - open your browser with "localhost:5000"
  - During chat, anytime, "@bot" 4 char key word will send the string after this key word to Llamma2 local model
