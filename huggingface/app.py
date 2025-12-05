from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from duckduckgo_search import DDGS
import gradio as gr
import torch

base_model = "unsloth/Llama-3.2-3B-Instruct"
peft_model_path = "Yuxin020807/Iris"

config = PeftConfig.from_pretrained(peft_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,   
)


def duckduckgo_search(query, max_results=5):
    ddgs = DDGS()

    results = list(ddgs.news(query, max_results=max_results))
    if not results:
        results = list(ddgs.text(query, max_results=max_results))

    info = []
    for r in results:
        info.append(f"- {r.get('title', '')}: {r.get('body', '')}")

    return "\n".join(info) if info else "No results found."


def chat_function(message, history, system_prompt, max_new_tokens, temperature):


    if message.lower().startswith("search:"):
        query = message.split(":", 1)[1].strip()

        realtime_context = duckduckgo_search(query)

        message = f"""The user asked: "{query}",
                    I searched online and found the following information:{realtime_context}
                    Please use this information to provide a helpful, clear answer to the user."""

    if any(word in message.lower() for word in ["latest", "current", "today", "real-time"]):
            realtime_context = duckduckgo_search(message)
            message = message + f"\n\n[REALTIME INFO]\n{realtime_context}"


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    prompt = gen_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    terminators = [
        gen_pipeline.tokenizer.eos_token_id,
        gen_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = gen_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )

    response_text = outputs[0]["generated_text"][len(prompt):]
    return response_text


demo_chatbot = gr.ChatInterface(
    chat_function,
    textbox=gr.Textbox(
        placeholder="Enter message here (type 'search: <query>' to search the web)",
        container=False,
        scale=7,
    ),
    chatbot=gr.Chatbot(height=450),
    additional_inputs=[
        gr.Textbox("You are a helpful AI assistant.", label="System Prompt"),
        gr.Slider(100, 4000, value=500, label="Max New Tokens"),
        gr.Slider(0, 1, value=0.7, label="Temperature")
    ],
    title="LoRA Chatbot with DuckDuckGo Search",
    description="Use 'search: <your query>' to fetch real-time information.",
)

demo_chatbot.launch(ssr_mode=False)


