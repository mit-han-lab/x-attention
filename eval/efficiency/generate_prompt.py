import torch

def generate_prompt(tokenizer,target_len,):
    context = "A quick brown fox jumps over the lazy dog. \n"
    with open("demo/xattention.txt", "r") as f:
        needle = f.read()

    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_repetitions = target_len // num_tokens_context

    text = (
        "This is a very long story book with knowledge of XAttention, which you need to remember for later question: <book> "
        + context * int(num_repetitions * 0.5)
        + needle
        + context * int(num_repetitions * 0.5)
        + "</book>\n Based on the content of the book, please briefly tell me about XAttention.\nAnswer:"
    )

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    suffix_len = len(tokenizer("</book>\n Based on the content of the book, please briefly tell me about XAttention.\nAnswer:", add_special_tokens=False))
    over_len = input_ids.shape[1] - target_len
    input_ids = torch.cat([input_ids[:, :-suffix_len-100-over_len], input_ids[:, -suffix_len-100:]], dim=1) if over_len > 0 else input_ids
    return input_ids
