import re
import datetime
import gradio as gr

from models import OpenAIModel, AnthropicModel, HuggingFaceLlama2Model
from tools import tools_prompt, run_tool

verbose = False

MAX_ACTIONS = 10   # Max number of actions we'll execute for each question
SYSTEM_MESSAGE_TEMPLATE = "prompt.txt"

EXAMPLES = [
    "How much can I earn on $10k in savings this year?",
    "Find a coffeeshop halfway between Burlingame and Palo Alto",
    "Should I go to the beach in Santa Cruz tomorrow?",
    "Who are the top 5 contenders for the Republican presidential nomination?",
    "What is the cause of the conflict between Zuck and Elon?",
    "Summarize today's NYT headlines",
]

MODELS = {
    "LLaMA 2": HuggingFaceLlama2Model("meta-llama/Llama-2-70b-chat-hf", 4096),
    "GPT-3.5": OpenAIModel("gpt-3.5-turbo-16k", 16384),
    "GPT-4": OpenAIModel("gpt-4", 8192),
    "Claude 2": AnthropicModel("claude-2", 100000),
}

selected_model_name = "LLaMA 2"
temperature = 0.1

def create_system_message():
    """
    Return system message, including today's date and the available tools.
    """
    with open(SYSTEM_MESSAGE_TEMPLATE) as f:
        message = f.read()

    now = datetime.datetime.now()
    current_date = now.strftime("%B %d, %Y")

    message = message.replace("{{CURRENT_DATE}}", current_date)
    message = message.replace("{{TOOLS_PROMPT}}", tools_prompt())

    return message

def generate(new_user_message, history):
    """
    Generate a response from the LLM to the user message while using the 
    available tools.  The history contains a list of prior 
    (user_message, assistant_response) pairs from the chat.
    This function is intended to be called by a Gradio ChatInterface.

    Within this function, we iteratively build up a prompt which includes
    the complete reasoning chain of:

       Question -> [Thought -> Action -> Result?] x N -> Conclusion

    Ideally, we would include the Result for every Action.  For most models,
    we quickly use up the entire context window when including the contents
    of web pages, so we only include the Result for the most recent Action.

    Note that Claude 2 supports a 100k context window, but in practice, I've
    found that the Anthropic API will return a rate limit error if I actually 
    try to send a large number of tokens, so unfortuantely I use the same logic
    with Claude 2 as the other models.
    """
    ACTION_REGEX = r'(\n|^)Action: (.*)\[(.*)\](\n|$)'
    CONCLUSION_REGEX = r'(\n|^)Conclusion: .*'

    prompt = f"Question: {new_user_message}\n\n"

    # full_response is displayed to the user in the ChatInterface and is the
    # same as the prompt, except it omits the Question and Result to improve
    # readability.
    full_response = ""

    iteration = 1

    model = MODELS[selected_model_name]
    system_message_token_count = model.count_tokens(system_message)

    try:
        while True:
            if verbose:
                print("=" * 60)
                print("PROMPT:")
                print(prompt)

            stream = model.generate(
                system_message,
                prompt,
                history=history,
                temperature=temperature
            )

            partial_response = ""

            for chunk in stream:
                completion = model.parse_completion(chunk)

                if completion:
                    # Stream each completion to the ChatInterface
                    full_response += completion
                    partial_response += completion
                    yield full_response

                    # When we find an Action in the response, stop the
                    # generation, run the tool specified in the Action,
                    # and create a new prompt that includes the Results.
                    matches = re.search(ACTION_REGEX, partial_response)
                    if matches:
                        tool = matches.group(2).strip()
                        params = matches.group(3).strip()
                        
                        result = run_tool(tool, params)

                        prompt = f"Question: {new_user_message}\n\n"
                        prompt += f"{full_response}\n\n"

                        # Calculate the number of tokens available in the
                        # context window, after accounting for the system
                        # message and previous responses
                        history_token_count = 0
                        for user_message, assistant_response in history:
                            history_token_count += model.count_tokens(user_message) + model.count_tokens(assistant_response)

                        prompt_token_count = model.count_tokens(prompt)
                        result_token_count = model.count_tokens(result)

                        available_tokens = int(0.8 * (model.context_size - system_message_token_count - history_token_count - prompt_token_count))

                        # Truncate the result if it is longer than the available tokens
                        if result_token_count > available_tokens:
                            ratio = available_tokens/result_token_count
                            truncate_result_len = int(len(result) * ratio)
                            result = result[:truncate_result_len]

                            full_response += f"\n\n*Note:  Only {ratio*100:.0f}% of the result was shown to the model due to context size limits.*"
                            yield full_response

                        prompt += f"Result: {result}\n\n"
                        break

            # Stop when we either see the Conclusion or we cannot find an 
            # Action in the response
            if re.search(CONCLUSION_REGEX, partial_response) or not re.search(ACTION_REGEX, partial_response):
                return
                        
            if not partial_response.endswith("\n"):
                full_response += "\n\n"
                yield full_response
            
            # Stop when we've exceeded MAX_ACTIONS
            if iteration >= MAX_ACTIONS:
                full_response += f"Stopping after running {MAX_ACTIONS} actions."
                yield full_response
                return
            else:
                iteration += 1
    
    except Exception as e:
        full_response += f"\nError: {e}"
        yield full_response


# Create Gradio app
system_message = create_system_message()
if verbose:
    print("=" * 60)
    print("SYSTEM MESSAGE:")
    print(system_message)
    print("=" * 60)

CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { flex-grow: 1; overflow: auto; }
#chatbot { flex-grow: 1; overflow: auto; }
"""

with gr.Blocks(css=CSS) as app:
    chatinterface = gr.ChatInterface(fn=generate, examples=EXAMPLES)
    chatinterface.chatbot.elem_id = "chatbot"

    with gr.Row():
        model_selector = gr.Radio(label="Model", choices=list(MODELS.keys()), value=selected_model_name)
        temperature_slider = gr.Slider(label="Temperature", minimum=0, maximum=1, value=temperature)

    def change_model(new_model_name):
        global selected_model_name
        selected_model_name = new_model_name

    def change_temperature(new_temperature):
        global temperature
        temperature = new_temperature

    model_selector.change(fn=change_model, inputs=model_selector)
    temperature_slider.change(fn=change_temperature, inputs=temperature_slider)

app.queue().launch(debug=True, share=False)
