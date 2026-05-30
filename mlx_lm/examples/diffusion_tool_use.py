# Copyright © 2025 Apple Inc.
"""
This is an example of tool use with diffusion generation in mlx_lm.

It mirrors examples/tool_use.py, but uses llada_generate() instead of the
autoregressive generate() path.
"""

from mlx_lm import llada_generate, load


# Specify the checkpoint
checkpoint = "GSAI-ML/LLaDA-8B-Instruct"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)

if not tokenizer.has_tool_calling:
    raise RuntimeError(
        "This tokenizer/chat template does not advertise tool calling. "
        "Use a tool-calling-capable model/template to run this example."
    )


def multiply(a: float, b: float):
    """
    A function that multiplies two numbers.

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """

    return a * b


tools = {"multiply": multiply}

# Specify the prompt and conversation history
prompt = "Multiply 12234585 and 48838483920."
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=list(tools.values()),
)

# Generate the initial tool call
response = llada_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    mode="faithful_llada",
    steps=32,
    gen_length=64,
    block_length=16,
)

text = response.text[0] if isinstance(response.text, list) else response.text

# Parse the tool call
start_tool = text.find(tokenizer.tool_call_start) + len(tokenizer.tool_call_start)
end_tool = text.find(tokenizer.tool_call_end)
tool_call = tokenizer.tool_parser(text[start_tool:end_tool].strip(), list(tools.values()))
tool_result = tools[tool_call["name"]](**tool_call["arguments"])

# Put the tool result in the prompt
messages = [
    {"role": "user", "content": prompt},
    {"role": "tool", "name": tool_call["name"], "content": tool_result},
]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

# Generate the final response
response = llada_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    mode="faithful_llada",
    steps=32,
    gen_length=64,
    block_length=16,
)

text = response.text[0] if isinstance(response.text, list) else response.text
print(text)
