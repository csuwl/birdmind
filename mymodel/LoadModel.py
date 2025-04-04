import torch
from transformers import AutoTokenizer

from Model import Model, ModelArgs


def generate(tokenizer , model:Model ,str, max_seq_len):
    tokens = tokenizer(
        str,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).input_ids
    for _ in range(max_seq_len):
        # batch_size,1, vocab_size
        logits = model.generate(tokens, 0)
        logits = logits.softmax(dim=-1)
        last_token: torch.Tensor = logits.argmax(-1)

        # cat response
        last_token = last_token.unsqueeze(0)
        tokens = torch.cat((tokens, last_token), dim=-1)
        print(tokenizer.decode(tokens[0].tolist()))

    print(tokenizer.decode(tokens.tolist()))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        print("use cuda")
        print(torch.__version__)
        print(torch.version.cuda)
    else:
        print("use cpu")

    args = ModelArgs(device = device, vocab_size=6400, embedding_dim=512)
    tokenizer, model = Model.init_model(args,"./sft_model.pth")



    generate(tokenizer, model, ['你好'], 40)


