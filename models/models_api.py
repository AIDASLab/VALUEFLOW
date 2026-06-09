# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# code from Valuebench (https://github.com/ValueByte-AI/ValueBench)

import os
from abc import ABC
import concurrent.futures
from tqdm import tqdm
import time
from dotenv import load_dotenv

from openai import OpenAI
from google import genai
from google.genai import types  

class LLMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     do_sample=True,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        return out

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)

class OpenAIModel(LLMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models or Llama API models.

    Inherits from LLMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is 'You are a helpful assistant.').
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt=None, openai_key=None):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)    
        self.openai_key = openai_key
        self.system_prompt = system_prompt

    def predict(self, input_text, kwargs={}):
        client = OpenAI(api_key=self.openai_key if self.openai_key is not None else os.environ['OPENAI_API_KEY'])
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        messages.insert(0, system_messages)
    
        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response_format = kwargs['response_format'] if 'response_format' in kwargs else None
        
        for attempt in range(1000):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    n=n,
                    response_format={"type": "json_object"} if response_format=="json" else None,
                )
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({attempt + 1})...")
                time.sleep(1)
            
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content
            
        return result

    def multi_predict(self, input_texts, **kwargs):
        """
        An example of input_texts:
        input_texts = ["Hello!", "How are you?", "Tell me a joke."]
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            args = [(messages, kwargs) for messages in input_texts]
            contents = executor.map(lambda p: self.predict(*p), args)
        return list(contents)

    def batch_predict(self, input_texts, **kwargs):
        assert "n" not in kwargs or kwargs["n"] == 1, "n > 1 is not supported for batch prediction."
        responses_list = []
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 200
        for start_idx in tqdm(range(0, len(input_texts), batch_size)):
            end_idx = min(start_idx + batch_size, len(input_texts))
            batch_input_texts = input_texts[start_idx: end_idx]
            batch_results_list = self.multi_predict(batch_input_texts, **kwargs)
            responses_list.extend(batch_results_list)
            # Save responses to file
            with open(f"temp-file-responses-{self.model_name}.txt", "a") as f:
                for response in batch_results_list:
                    f.write(response + "\n")
        return responses_list

class GeminiModel(LLMBaseModel):
    """
    Language model class for interfacing with Google's Gemini models.

    Inherits from LLMBaseModel and sets up a model interface for Gemini models.

    Parameters:
    -----------
    model : str
        The name of the PaLM model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    gemini_key : str, optional
        The Gemini API key (default is None).
    """
    def __init__(self, model_name, max_new_tokens, temperature, device='auto', dtype=None, system_prompt=None):
        super(GeminiModel, self).__init__(model_name, max_new_tokens, temperature)        
        self.system_prompt = system_prompt or "You are a helpful assistant."

        load_dotenv() 
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=gemini_key)
    
    def predict(self, input_text, **kwargs):

        # Set up the model
        generation_config = types.GenerateContentConfig(
            temperature= self.temperature,            
            max_output_tokens= self.max_new_tokens,
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )   
        
        response = self.client.models.generate_content(
            model = self.model_name,
            config=generation_config,
            contents = input_text
        )        
        print(response.text)
        return response.text

class GrokAPIModel(ABC):
    """
    Language model class for interfacing with xAI's Grok models via OpenAI-compatible API.

    Parameters
    ----------
    model_name : str
        The Grok model name (e.g., "grok-4").
    max_new_tokens : int
        Max tokens in the completion (mapped to max_tokens).
    temperature : float
        Sampling temperature.
    system_prompt : str, optional
        System message; defaults to a generic assistant instruction.
    xai_key : str, optional
        xAI API key. If None, uses the XAI_API_KEY environment variable.
    base_url : str, optional
        API base URL. Defaults to "https://api.x.ai/v1".
    """
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt=None, xai_key=None, base_url=None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or "You are a helpful assistant."
        load_dotenv()
        self.xai_key = os.getenv("XAI_API_KEY")
        if not self.xai_key:
            raise ValueError("Missing xAI API key. Set XAI_API_KEY env var or pass xai_key=")
        self.base_url = "https://api.x.ai/v1"

    def predict(self, input_text, kwargs={}):
        """
        Run a single chat completion call to Grok.

        input_text can be:
          - str: treated as a single user message
          - dict: a single message object
          - list[dict]: full messages list (system message is prepended automatically)
        Optional kwargs:
          - n: number of completions (default 1)
          - temperature: override temperature
          - max_new_tokens: override max tokens
          - response_format: "json" to request JSON object
        """
        client = OpenAI(api_key=self.xai_key, base_url=self.base_url)

        # # Build messages
        # if isinstance(input_text, list):
        #     messages = input_text
        # elif isinstance(input_text, dict):
        #     messages = [input_text]
        # else:
        #     messages = [{"role": "user", "content": input_text}]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        response_format = kwargs.get("response_format", None)

        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_new_tokens,
            # n=n,
            # response_format={"type": "json_object"} if response_format == "json" else None,
        )

        print(resp)
        return resp.choices[0].message.content

        # last_err = None
        # for attempt in range(10):
        #     try:
                
        #         if n > 1:
        #             return [c.message.content for c in resp.choices]
                
                
            
        #     except Exception as e:
        #         last_err = e
        #         print(e)
        #         # simple backoff
        #         time.sleep(min(2 ** attempt, 10))

        # raise RuntimeError(f"GrokAPIModel request failed after retries: {last_err}")

    def __call__(self, input_text, **kwargs):        
        return self.predict(input_text, kwargs=kwargs)