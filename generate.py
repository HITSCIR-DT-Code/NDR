import json
from model import Mymodel,tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import random
from tqdm.auto import tqdm
import time
import os
from transformers import Trainer, TrainingArguments,DataCollatorWithPadding
from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.bleu_score import corpus_bleu

class MyDataset(Dataset):
    def __init__(self,rawdata,delete_rate=0,orderindex=0):
        self.rawdata=rawdata
        self.delete_rate=delete_rate
        self.orderindex=orderindex
        self.corrupt_data=self._construct_input(self.rawdata)
        
    def _construct_input(self,rawdata):
        #to random delete the orginal input sentence word piece
        corrupt_input=[]
        for _,data in enumerate(rawdata):
            word_squence=data["prompt"].split()
            delete_index=np.random.binomial(1,self.delete_rate,len(word_squence))
            new_word_squence=[word_squence[i] for i in range(len(word_squence)) if delete_index[i]==0]
            corrupt_input.append(" ".join(new_word_squence))
        assert len(corrupt_input)==len(rawdata), "length error!"
        return corrupt_input
        
    def __getitem__(self, index):
        return {"input":self.corrupt_data[index], 
                "class_labels":torch.tensor(int(self.rawdata[index]["source"][-1])-1,dtype=torch.long),
                "labels":self.rawdata[index]["prompt"]
        }
    def __len__(self):
        return len(self.rawdata)

def train(model,train_data,val_data,tokenizer,config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    savepath="savedmodel/"+time.asctime()[4:].replace(" ","-").replace(":", "-")+"/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    with open(savepath+"config.json","w",encoding="utf-8") as f:
        json.dump(config,f,ensure_ascii=False,indent=4)
    train_set,val_set=MyDataset(train_data,orderindex=1),MyDataset(val_data,orderindex=1)
    train_loader=DataLoader(dataset=train_set,batch_size=config["batch_size"],shuffle=True,drop_last=False)
    val_loader=DataLoader(dataset=val_set,batch_size=config["batch_size"],shuffle=False,drop_last=False)
    #training setting
    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["lr"])
    num_training_steps = config["Epochs"] * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    with open(savepath+'0output.txt', 'a') as f:
        f.writelines(f"epoch:{config['Epochs']},each epoch's batchs{len(train_loader)}, total batch{num_training_steps}.\n")
    for epoch in range(config["Epochs"]):
        for i,batch in enumerate(train_loader):
            model.train()
            input=tokenizer(batch["input"],padding=True,return_tensors="pt")
            labels=tokenizer(batch["labels"],padding=True,return_tensors="pt",return_attention_mask=False).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            input_ids=input.input_ids.to(device)
            attention_mask=input.attention_mask.to(device)
            class_labels=batch["class_labels"].to(device)
            batch_loss=model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,class_labels=class_labels,is_train=True).loss
            with open(savepath+'0output.txt', 'a') as f:
                f.writelines(f"[{epoch}-{i+1}] batch_train_loss={batch_loss.item()}\n")
            batch_loss=batch_loss/config["accumulation_steps"]
            batch_loss.backward()
            progress_bar.update(1)
            if ((i+1)%config["accumulation_steps"]==0) | ((i+1)==len(train_loader)):  
                optimizer.step()
                optimizer.zero_grad()
                with open(savepath+'0output.txt', 'a') as f:
                    f.writelines("start test!\n")
                model.eval()
                with torch.no_grad():
                    total_val_loss=0
                    val_output=[]
                    for _,val_batch in enumerate(val_loader):
                        val_input=tokenizer(val_batch["input"],padding=True,return_tensors="pt")
                        val_input_ids=val_input.input_ids.to(device)
                        val_attention_mask=val_input.attention_mask.to(device)
                        val_labels=tokenizer(val_batch["labels"],padding=True,return_tensors="pt",return_attention_mask=False).input_ids.to(device)
                        val_labels[val_labels == tokenizer.pad_token_id] = -100
                        val_class_labels=val_batch["class_labels"].to(device)
                        val_batch_loss=model(input_ids=val_input_ids,attention_mask=val_attention_mask,labels=val_labels,class_labels=val_class_labels,is_train=True).loss
                        total_val_loss+=val_batch_loss.item() 
                    with open(savepath+'0output.txt', 'a') as f:
                        f.writelines(f"[{epoch}-{i+1}] batch_train_loss={round(batch_loss.item(),4)},ValSet_loss={round(total_val_loss,4)}\n")
                torch.save(model, savepath+f"model-{epoch}-{i+1}.pth")

def prompt_enchancement(model_path,test_data,tokenizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=torch.load(model_path).to(device)
    model.eval()

    test_set=MyDataset(test_data,delete_rate=0,orderindex=0)
    test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=False,drop_last=False)
    with torch.no_grad():
        generate_outputs=[]
        generate_outputs.append({"model":model_path})
        for _,batch in enumerate(tqdm(test_loader)):
            test_input=tokenizer(batch["input"],padding=True,return_tensors="pt")
            batch_output=model.generate(input_ids=test_input.input_ids.to(device),
                                        attention_mask=test_input.attention_mask.to(device),
                                        max_new_tokens=512,do_sample=True,top_k=10,temperature=1.1,L=0.7
                                        )
            for j,predict_sentence in enumerate(batch_output):
                generate_outputs.append({
                    "prompt":batch["labels"][j],
                    "predict":tokenizer.decode(predict_sentence, skip_special_tokens=True),
                    })
    return generate_outputs

    
    
if __name__ == '__main__':
    #load your dataset here
    with open("data/xxx.json","r",encoding="utf-8") as f:
        train_data=json.load(f)
    with open("data/xxx.json","r",encoding="utf-8") as f:
        val_data=json.load(f)
    with open("config.json","r",encoding="utf-8") as f:
        config=json.load(f)
    
    #train
    model=Mymodel.from_pretrained("./models/t5-base/")
    for name, parameter in model.encoder.named_parameters():
        if name=="classify_layer.weight" or name=="null_space_layer_norm.weight":
            pass
        else:
            parameter.requires_grad = False
    model.decoder.embed_tokens.weight.requires_grad = False
    train(model,train_data,val_data,tokenizer,config)
    
    #test
    with open("data/xxx.json","r",encoding="utf-8") as f:
        test_data=json.load(f)
    model_path="savedmodel/final/xxx.pth"
    generate_prompts=prompt_enchancement(model_path,test_data,tokenizer)
