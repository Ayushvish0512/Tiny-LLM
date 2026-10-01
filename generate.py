def build_prompt(user_input: str) -> str:
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def clean_chunk(text: str) -> str:
    for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    return text