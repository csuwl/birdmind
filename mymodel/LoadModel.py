import torch
from transformers import AutoTokenizer

from mymodel.Model import Model, ModelArgs


def generate(tokenizer , model:Model ,str, max_seq_len):
    tokens = tokenizer(
        str,
        max_length=512,
        padding='max_length',
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
    tokenizer = AutoTokenizer.from_pretrained('./minimind_tokenizer')

    vocab_size = 6400
    args = ModelArgs(vocab_size=vocab_size, embedding_dim=512)
    model = Model(args)

    model.load_state_dict(torch.load('./model.pth'))



    generate(tokenizer, model, ['你好'], 40)


