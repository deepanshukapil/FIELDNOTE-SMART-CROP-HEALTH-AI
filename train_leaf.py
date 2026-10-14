import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from leaf_model import LeafCNN

def main():
    p=argparse.ArgumentParser(description="Train CNN using class-wise labeled leaf image folders")
    p.add_argument("--data",default="data/plantvillage"); p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--out",default="models")
    args=p.parse_args(); root=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tfm=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(12),transforms.ColorJitter(brightness=.12,contrast=.12),transforms.ToTensor(),normalize])
    val_tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    train_ds=datasets.ImageFolder(root,transform=train_tfm); val_ds=datasets.ImageFolder(root,transform=val_tfm)
    if len(train_ds.classes)<2: raise SystemExit("Need at least two class folders containing labeled leaf images")
    if len(train_ds)<10: raise SystemExit("Need at least 10 images for a basic train/validation split")
    n_val=max(1,int(len(train_ds)*.2)); gen=torch.Generator().manual_seed(42); perm=torch.randperm(len(train_ds),generator=gen).tolist(); val_idx=perm[:n_val]; train_idx=perm[n_val:]
    train=DataLoader(Subset(train_ds,train_idx),batch_size=args.batch_size,shuffle=True,num_workers=0); valid=DataLoader(Subset(val_ds,val_idx),batch_size=args.batch_size,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=LeafCNN(len(train_ds.classes)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss(); best=-1.0
    print(f"Images={len(train_ds)}, classes={len(train_ds.classes)}, device={device}")
    for epoch in range(args.epochs):
        model.train(); total=correct=loss_sum=0
        for x,y in train:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
        model.eval(); hits=count=0
        with torch.no_grad():
            for x,y in valid:
                logits=model(x.to(device)); hits+=(logits.argmax(1).cpu()==y).sum().item(); count+=len(y)
        val_acc=hits/max(1,count); print(f"Epoch {epoch+1}/{args.epochs} train_loss={loss_sum/max(1,total):.4f} train_acc={correct/max(1,total):.3f} val_acc={val_acc:.3f}")
        if val_acc>=best:
            best=val_acc; torch.save(model.cpu().state_dict(),out/"leaf_cnn.pt"); model.to(device)
            (out/"leaf_classes.json").write_text(json.dumps(train_ds.classes,indent=2))
    print(f"Best validation accuracy={best:.3f}; checkpoint={out/'leaf_cnn.pt'}")
    print("Note: random split of near-duplicate leaves can overstate field performance; test on a separate, field-collected dataset before deployment.")
if __name__=="__main__": main()
