import torch
from Model import Model,ModelArgs

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        print("use cuda")
        print(torch.__version__)
        print(torch.version.cuda)
    else:
        print("use cpu")

    args = ModelArgs(device = device, vocab_size=6400, embedding_dim=512)
    tokenizer, model = Model.init_model(args)

    