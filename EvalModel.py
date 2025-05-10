import torch
from models.BirdMindModel import BirdMindModel,BirdMindConfig

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        print("use cuda")
        print(torch.__version__)
        print(torch.version.cuda)
    else:
        print("use cpu")

    args = BirdMindConfig(device = device, vocab_size=10000, embedding_dim=512,block_size=16)
    tokenizer, model = BirdMindModel.init_model(args,'./sft_model_10000_nomoe.pth')


    for index, prompt in enumerate(iter(lambda: input('请输入:'),'')):
        # messages = []
        # messages.append({"role": "user", "content": prompt})

        # new_prompt = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False
        # )
        # print(new_prompt)
        
        answer = prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_length=args.max_seq_len,
                temperature=0.4,
                top_p=0.8,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('🤖️: ', end='')
            try:
                history_idx = 0
                for y in outputs:
                    answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                    if (answer and answer[-1] == '�') or not answer:
                        continue
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')
