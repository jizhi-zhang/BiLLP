from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    HumanMessage
)
import openai
import os

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content
            
class OpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        self.model_name = kwargs.get('model_name', 'gpt-3.5-turbo-16k') 
        self.temperature = kwargs.get('temperature', 0)
        self.max_tokens= kwargs.get('max_tokens', 8000)
        
        self.model_kwargs = kwargs.get('model_kwargs', {"stop": "\n"})
        self.openai_api_key = kwargs.get('openai_api_key', os.environ['OPENAI_API_KEY'])
        self.openai_api_base = kwargs.get('openai_api_base', os.environ['OPENAI_API_BASE'])
    
    def __call__(self, prompt: str):
        openai.api_key = self.openai_api_key
        openai.api_base = self.openai_api_base
        response = openai.ChatCompletion.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens = self.max_tokens,
                stop = self.model_kwargs['stop'],
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )["choices"][0]["message"]["content"]
        return response