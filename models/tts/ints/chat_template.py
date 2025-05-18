# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def format_chat_prompt_phi3(messages, add_assistant_token=True):
    """
    Convert the messages list into the phi-3 chat template format.

    Args:
        messages: A list of messages containing role and content.

    Returns:
        str: The formatted prompt string.
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Add corresponding tags for system and user messages
        if role in ["system", "user"]:
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        # For assistant messages, add only the start tag if it's the last one
        elif role == "assistant" and msg != messages[-1]:
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        elif role == "assistant" and msg == messages[-1]:
            prompt += f"<|{role}|>\n{content}"

    # If the last message is not from the assistant, add the assistant tag
    if messages[-1]["role"] != "assistant" and add_assistant_token:
        prompt += "<|assistant|>"
    return prompt


def gen_chat_prompt_for_tts(text):
    template = [
        {
            "role": "system",
            "content": "You are a powerful AI assistant for speech understanding and generation.",
        },
        {
            "role": "user",
            "content": f"Please speak the following text out loud: {text}",
        },
    ]
    return format_chat_prompt_phi3(template)
