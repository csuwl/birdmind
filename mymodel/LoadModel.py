import torch
from transformers import AutoTokenizer

from BirdMindModel import BirdMindModel, BirdMindConfig


def generate(tokenizer , model:BirdMindModel ,str, max_seq_len):
    tokens = tokenizer(
        str,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).input_ids
    for _ in range(max_seq_len):
        # batch_size,1, vocab_size
        logits = model.generate_my(tokens, 0)
        logits = logits.softmax(dim=-1)
        last_token: torch.Tensor = logits.argmax(-1)

        # cat response
        last_token = last_token.unsqueeze(0)
        tokens = torch.cat((tokens, last_token), dim=-1)
        print(tokenizer.decode(tokens[0].tolist()))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        print("use cuda")
        print(torch.__version__)
        print(torch.version.cuda)
    else:
        print("use cpu")

    args = BirdMindConfig(device = device, vocab_size=6400, embedding_dim=512)
    tokenizer, model = BirdMindModel.init_model(args,"./model.pth")



    generate(tokenizer, model, ['什么动物好看'], 40)


