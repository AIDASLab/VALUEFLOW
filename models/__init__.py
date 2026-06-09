# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# code from Valuebench (https://github.com/ValueByte-AI/ValueBench)

from tqdm import tqdm

USE_LOCAL_MODELS = False

try:
    from .models import *
    USE_LOCAL_MODELS = True
except ImportError:
    print("load from api")
    from .models_api import *

# Conditional MODEL_LIST and SUPPORTED_MODELS setup
if USE_LOCAL_MODELS:
    MODEL_LIST = {
        T5Model: ['google/flan-t5-large'],   
        PhiModel: ['phi-1.5', 'phi-2'],
        PaLMModel: ['palm'],    
        VicunaModel: ['vicuna-7b', 'vicuna-13b', 'vicuna-13b-v1.3'],
        UL2Model: ['google/flan-ul2'],    
        MistralModel: ['mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.3'],
        MixtralModel: ['mistralai/Mixtral-8x7B-v0.1'],
        YiModel: ['01-ai/Yi-6B', '01-ai/Yi-34B', '01-ai/Yi-6B-Chat', '01-ai/Yi-34B-Chat'],
        BaichuanModel: ['baichuan-inc/Baichuan2-7B-Base', 'baichuan-inc/Baichuan2-13B-Base',
                        'baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'],
        Qwen3Model: ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B', 'Qwen/Qwen3-14B', 'Qwen/Qwen3-32B'],
        Qwen2Model: ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'],
        Phi4Model: ['microsoft/phi-4', 'microsoft/Phi-4-reasoning','microsoft/Phi-4-mini-instruct'],
        GLMModel: ['zai-org/GLM-4-32B-0414', 'THUDM/GLM-4-9B-0414', 'THUDM/GLM-4-32B-0414', "THUDM/GLM-4-32B-Base-0414"],
        Gemma3Model: ['google/gemma-3-4b-it','google/gemma-3-12b-it','google/gemma-3-27b-it'],
        Llama2Model: ["meta-llama/Llama-2-13b-chat-hf"],
        Llama3Model: ["meta-llama/Llama-3.1-8B-Instruct"],
        Mistral3Model: ["mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"],
        GPTOSSModel: ["openai/gpt-oss-20b","openai/gpt-oss-120b"]
    }
else:
    MODEL_LIST = {
        OpenAIModel: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'],
        GeminiModel: ['gemini-pro', 'gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro'],
        GrokAPIModel: ['grok-4-0709']
    }

SUPPORTED_MODELS = [model for model_class in MODEL_LIST for model in MODEL_LIST[model_class]]

class LLMModel(object):
    """
    A class providing an interface for various language models.
    """

    @staticmethod
    def model_list():
        return SUPPORTED_MODELS

    def __init__(self, model, max_new_tokens=20, temperature=0, device="cuda", dtype="auto", model_dir=None, system_prompt=None, api_key=None):
        self.model_name = model
        self.model = self._create_model(max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key)

    def _create_model(self, max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key):
        model_mapping = {model: model_class for model_class in MODEL_LIST for model in MODEL_LIST[model_class]}
        model_class = model_mapping.get(self.model_name)

        if not model_class:
            raise ValueError("The model is not supported!")
        
        return model_class(self.model_name, max_new_tokens, temperature, device, dtype, system_prompt)

    def __call__(self, input_texts, **kwargs):        
        responses = []
        for input_text in tqdm(input_texts):
            response = self.model.predict(input_text, **kwargs)
            responses.append(response)
            print(f"prompt: {input_text}")
            print(f"response: {response}")
        return responses
