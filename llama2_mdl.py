# torch langchain transformers faiss-gpu xformers sentence-transformers python_dotenv atlassian-python-api markdownify

import torch
from torch import cuda, bfloat16

import transformers
from transformers import StoppingCriteria, StoppingCriteriaList

import os
from os.path import join, dirname
from dotenv import load_dotenv

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class llama2_mdl:
  def __init__(self, model_id='meta-llama/Llama-2-7b-chat-hf'):
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    self.mdl_id = model_id
    self.SECRET_KEY = os.environ.get("SECRET_KEY")

  def init_mdl(self):
    mdl_config = transformers.AutoConfig.from_pretrained(
      self.mdl_id,
      use_auth_token=self.SECRET_KEY
    )

    mdl = transformers.AutoModelForCausalLM.from_pretrained(
      self.mdl_id,
      trust_remote_code=True,
      config=mdl_config,
      device_map='auto',
      use_auth_token=self.SECRET_KEY
    )

    # enable evaluation mode to allow model inference
    mdl.eval()
    #print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
      self.mdl_id,
      use_auth_token=self.SECRET_KEY
    )

    stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    #print(f"{stop_token_ids}")


    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
      def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
          pt_stop_ids = torch.tensor(stop_ids, device=device)
          if torch.eq(input_ids[0][-len(stop_ids):], pt_stop_ids).all():
            return True
        return False


    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    self.generate_text = transformers.pipeline(
      model=mdl,
      tokenizer=tokenizer,
      return_full_text=True,  # langchain expects the full text
      task='text-generation',
      # we pass model parameters here too
      stopping_criteria=stopping_criteria,  # without this model rambles during chat
      temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      max_new_tokens=512,  # max number of tokens to generate in the output
      repetition_penalty=1.1  # without this output begins repeating
    )

  def infer(self, prompt):
    res = self.generate_text(prompt)
    return res[0]["generated_text"]

if __name__ == '__main__':
  llama = llama2_mdl()
  llama.init_mdl()
  print(llama.infer("Explain to me what is ETF?"))