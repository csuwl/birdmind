import tiktoken
import torch

from mymodel.Model import Model, ModelArgs

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("cl100k_base")

    vocab_size = tokenizer.n_vocab
    args = ModelArgs(vocab_size=vocab_size, embedding_dim=64)
    model = Model(args)

    model.load_state_dict(torch.load('./model.pth'))

    tokens = torch.tensor(tokenizer.encode('你'), dtype=torch.long)
    tokens = tokens.unsqueeze(0)


    for _ in range(40):
        # batch_size, 1，vocab_size
        logits = model.generate(tokens, 0)
        logits = logits.softmax(dim=-1)
        last_token = logits.argmax(1)
        res = tokenizer.decode(last_token.tolist())
        print(res)
        # cat response
        last_token = last_token.unsqueeze(-1)
        tokens = torch.cat((tokens, last_token), dim=-1)
