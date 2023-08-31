# WebChat
This is a chatbot built using Gradio that can access Google Search and webpages to answer questions:
* Supports LLMs: [GPT-3.5](https://openai.com/blog/chatgpt), [GPT-4](https://openai.com/research/gpt-4), [Claude 2](https://www.anthropic.com/index/claude-2), [Llama 2 (70B)](https://ai.meta.com/llama/)
* Supports browsers: Chrome, Firefox, Safari, Edge

See [this article](https://medium.com/@thait/webchat-building-a-chatbot-with-access-to-a-web-browser-238602ee751f) for an overview of the implementation.

![demo](https://github.com/thaitran/WebChat/assets/432859/c6561f15-affa-4183-b07f-01d4ca22fe98)

## Installation

Clone this repo:
```
git clone https://github.com/thaitran/WebChat.git
cd WebChat
```

Install required Python modules:
```
pip install -r requirements.txt
```

## Setup access to LLMs

If you'd like to use GPT-3.5 or GPT-4, sign up for an [OpenAI developer account](https://platform.openai.com/), obtain an API key from https://platform.openai.com/account/api-keys and then set the following environment variable:
```
export OPENAI_API_KEY=<your key>
```

If you'd like to use Claude 2, sign up for an [Anthropic developer account](https://console.anthropic.com/), obtain an API key from https://console.anthropic.com/account/keys and then set the following environment variable:
```
export ANTHROPIC_API_KEY=<your key>
```

If you'd like to use Llama 2, sign up for a [Hugging Face Pro account](https://huggingface.co/pricing) -- this gives you unlimited usage of Llama 2 inference for $9/month.  Obtain an access token from https://huggingface.co/settings/tokens and then run:
```
huggingface-cli login
```
or set the following environment variable:
```
export HUGGING_FACE_HUB_TOKEN=<your token>
```

## Running the app

```
python app.py
```

Then open your web browser to http://127.0.0.1:7860

## Feedback

If you'd like to provide feedback, you can reach the author at:
* LinkedIn: https://www.linkedin.com/in/thait/
* Twitter: [@thait](https://twitter.com/thait)
* Threads: [@thai](https://www.threads.net/@thai)
