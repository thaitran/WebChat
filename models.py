import os
import openai
import anthropic
import huggingface_hub
import tiktoken # Tokenizer for OpenAI GPT models
import sentencepiece # Tokenizer for LLaMA 2 model

MAX_TOKENS = 1000  # Max number of tokens that each model should generate

class Model:
    """
    Common interface for all chat model API providers
    """
    def __init__(self, model_name, context_size):
        self.model_name = model_name
        self.context_size = context_size

    def generate(self, system_message, new_user_message, history=[], temperature=1):
        """
        Return a generator that will stream the completions from the model.

        The history is a list of prior (user_message, assistant_response) 
        pairs from the chat.
        """
        return None
    
    def parse_completion(self, completion):
        """
        Convert the output from the stream generator into a string.
        """
        return None
    
    def count_tokens(self, str):
        """
        Count the number of tokens in the string
        """
        return None

class OpenAIModel(Model):
    """
    Interface for OpenAI's GPT models
    """
    def generate(self, system_message, new_user_message, history=[], temperature=1):
        if 'OPENAI_API_KEY' not in os.environ:
            raise Exception("This model will be run from www.openai.com - Please obtain an API key from https://platform.openai.com/account/api-keys and then set the following environment variable before running this app:\n```\nexport OPENAI_API_KEY=<your key>\n```")

        messages = [
            { "role": "system", "content": system_message},
        ]

        for user_message, assistant_response in history:
            messages.append({ "role": "user", "content": user_message })
            messages.append({ "role": "assistant", "content": assistant_response })

        messages.append({ "role": "user", "content": new_user_message })

        stream = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stream=True
        )

        return stream
    
    def parse_completion(self, completion):
        delta = completion["choices"][0]["delta"]
        if "content" in delta:
            return delta["content"]
        else:
            return None
        
    def count_tokens(self, str):
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(str))

class AnthropicModel(Model):
    """
    Interface for Anthropic's Claude models
    """
    def generate(self, system_message, new_user_message, history=[], temperature=1):
        if 'ANTHROPIC_API_KEY' not in os.environ:
            raise Exception("This model will be run from www.anthropic.com - Please obtain an API key from https://console.anthropic.com/account/keys and then set the following environment variable before running this app:\n```\nexport ANTHROPIC_API_KEY=<your key>\n```")

        client = anthropic.Anthropic()
        prompt = system_message + "\n"

        for user_message, assistant_response in history:
            prompt += anthropic.HUMAN_PROMPT + user_message + "\n" + anthropic.AI_PROMPT + assistant_response + "\n"
            
        prompt += anthropic.HUMAN_PROMPT + new_user_message + anthropic.AI_PROMPT

        stream = client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens_to_sample=MAX_TOKENS,
            stream=True
        )

        return stream
    
    def parse_completion(self, completion):
        return completion.completion
    
    def count_tokens(self, str):
        client = anthropic.Anthropic()
        return client.count_tokens(str)

class HuggingFaceLlama2Model(Model):
    """
    Interface for Meta's LLaMA 2 Chat models, served via the Hugging Face Inteference API
    """
    def generate(self, system_message, new_user_message, history=[], temperature=1):
        try:
            hf_username = huggingface_hub.whoami()
        except huggingface_hub.utils._headers.LocalTokenNotFoundError:
            raise Exception("This model will be run from www.huggingface.co inference API - Please sign up for a Hugging Face Pro account and obtain an access token from https://huggingface.co/settings/tokens and then run:\n```\nhuggingface-cli login\n```\nor set the following environment variable:\n```\nexport HUGGING_FACE_HUB_TOKEN=<your token>\n```")

        client = huggingface_hub.InferenceClient()

        # The system message and the first user message are enclosed in the same [INST] tag
        # All subsequent user messages are enclosed in their own [INST] tags
        prompt = f"[INST]<<SYS>>{system_message}<</SYS>>\n\n"

        first_message = True

        for user_message, assistant_response in history:
            if first_message:
                first_message = False
            else:
                prompt += "[INST]"

            prompt += user_message + "[/INST]" + assistant_response + "\n\n"

        if first_message:
            first_message = False
        else:
            prompt += "[INST]"

        prompt += new_user_message + "[/INST]"

        stream = client.text_generation(
            prompt,
            model=self.model_name,
            temperature=temperature,
            max_new_tokens=MAX_TOKENS,
            stream=True
        )
        
        return stream
        
    def parse_completion(self, completion):
        return completion
    
    def count_tokens(self, str):
        sp = sentencepiece.SentencePieceProcessor(model_file="llama/tokenizer.model")
        return len(sp.EncodeAsIds(str))
    