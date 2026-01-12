import json
from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import random
from tqdm.auto import tqdm
import time
import os
from transformers import Trainer, TrainingArguments,DataCollatorWithPadding


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
            #delete_index=np.random.binomial(1,self.delete_rate,len(word_squence))
            if self.orderindex==1:
                delete_index=[np.random.binomial(1,1-i/len(word_squence)) for i in range(1,len(word_squence)+1)]
                new_word_squence=[word_squence[i] for i in range(len(word_squence)) if delete_index[i]==0]
                random.shuffle(new_word_squence)
                corrupt_input.append(" ".join(new_word_squence))
            elif self.orderindex==-1:
                delete_index=[np.random.binomial(1,i/len(word_squence)) for i in range(1,len(word_squence)+1)]
                new_word_squence=[word_squence[i] for i in range(len(word_squence)) if delete_index[i]==0]
                random.shuffle(new_word_squence)
                corrupt_input.append(" ".join(new_word_squence))
            elif self.delete_rate!=0:
                delete_index=np.random.binomial(1,self.delete_rate,len(word_squence))
                new_word_squence=[word_squence[i] for i in range(len(word_squence)) if delete_index[i]==0]
                corrupt_input.append(" ".join(word_squence))
            else:
                corrupt_input.append(" ".join(word_squence))
            #randomly order delected sentences
            # if self.delete_rate !=0:
            #     random.shuffle(new_word_squence)
            
        assert len(corrupt_input)==len(rawdata)
        return corrupt_input
        
    def __getitem__(self, index):
        return {"input":self.corrupt_data[index], 
                "labels":self.rawdata[index]["prompt"]
        }
    def __len__(self):
        return len(self.rawdata)
    
tokenizer = T5Tokenizer.from_pretrained("./models/t5-base/")
model = T5ForConditionalGeneration.from_pretrained("./models/t5-base/")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
savepath="savedmodel/ablation/"
with open("config.json","r",encoding="utf-8") as f:
    config=json.load(f)
with open("data/train.json","r",encoding="utf-8") as f:
    train_data=json.load(f)
with open("data/val.json","r",encoding="utf-8") as f:
    val_data=json.load(f)
train_set,val_set=MyDataset(train_data,orderindex=1),MyDataset(val_data,orderindex=1)
train_loader=DataLoader(dataset=train_set,batch_size=32,shuffle=True,drop_last=False)
val_loader=DataLoader(dataset=val_set,batch_size=64,shuffle=False,drop_last=False)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=config["lr"])
num_training_steps = config["Epochs"] * len(train_loader)
progress_bar = tqdm(range(num_training_steps))
with open(savepath+'0output.txt', 'a') as f:
    f.writelines(f"epoch is:{config['Epochs']},each epoch's total batchs{len(train_loader)},total batchs{num_training_steps}。\n")
for epoch in range(config["Epochs"]):
    for i,batch in enumerate(train_loader):
        model.train()
        input=tokenizer(batch["input"],padding=True,return_tensors="pt")
        labels=tokenizer(batch["labels"],padding=True,return_tensors="pt",return_attention_mask=False).input_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        input_ids=input.input_ids.to(device)
        attention_mask=input.attention_mask.to(device)
        batch_loss=model(input_ids=input_ids,attention_mask=attention_mask,labels=labels).loss
        with open(savepath+'0output.txt', 'a') as f:
            f.writelines(f"[{epoch}-{i+1}] batch_train_loss={batch_loss.item()}\n")
        batch_loss=batch_loss/config["accumulation_steps"]
        batch_loss.backward()
        progress_bar.update(1)
        if ((i+1)%config["accumulation_steps"]==0) | ((i+1)==len(train_loader)):  
            optimizer.step()
            optimizer.zero_grad()
            with open(savepath+'0output.txt', 'a') as f:
                f.writelines("start evaluated\n")
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
                    val_batch_loss=model(input_ids=val_input_ids,attention_mask=val_attention_mask,labels=val_labels).loss
                    total_val_loss+=val_batch_loss.item() 
                """val_batch_output=model.generate(input_ids=val_input_ids,attention_mask=val_attention_mask,max_new_tokens=500)
                    for j,predict_sentence in enumerate(val_batch_output):
                        val_output.append({
                            "predict":tokenizer.decode(predict_sentence, skip_special_tokens=True),
                            "true":val_batch["labels"][j]
                            })
                blue_score,chrf_score=compute_score([item["true"] for item in val_output],[item["predict"] for item in val_output])
                with open(savepath+'0output.txt', 'a') as f:
                    f.writelines(f"[{epoch}-{i+1}] batch_train_loss={round(batch_loss.item(),4)},ValSet_loss={round(total_val_loss,4)},ValBlue={round(blue_score,3)},ValChrf={round(chrf_score,3)}\n")
                    """
                with open(savepath+'0output.txt', 'a') as f:
                    f.writelines(f"[{epoch}-{i+1}] batch_train_loss={round(batch_loss.item(),4)},ValSet_loss={round(total_val_loss,4)}\n")
            torch.save(model, savepath+f"model-{epoch}-{i+1}.pth")