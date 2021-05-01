import numpy as np
import torch

from torch.utils.data import DataLoader,TensorDataset
from model import MODEL_SIGMOID,MODEL_TANH
import argparse
from plot_data import plot_output,plot_loss,plot_output2
import random

def init_weights(m):
    if type(m) == torch.nn.Linear:
        print(m)
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0.01)

def train(trainset, valset, testset, hidden, batch,epochs,dataset,save_,lr):
    train_xy= torch.from_numpy(trainset[:, 0:2]).float()
    train_z = torch.from_numpy(trainset[:, 2]).float().view(-1, 1)
    val_xy = torch.from_numpy(valset[:, 0:2]).float()
    val_z = torch.from_numpy(valset[:, 2]).float().view(-1, 1)
    test_xy = torch.from_numpy(testset[:, 0:2]).float()
    test_z = torch.from_numpy(testset[:, 2]).float().view(-1, 1)

    trainset = TensorDataset(train_xy,train_z)
    trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    valset = TensorDataset(val_xy, val_z)
    valloader = DataLoader(trainset, batch_size=batch, shuffle=True)
    testset = TensorDataset(test_xy, test_z)
    testloader = DataLoader(testset, batch_size=batch, shuffle=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    #model part

    if torch.cuda.is_available():
        device = torch.device("cuda:0");
        print("using cuda:0")
    else:
        device = torch.device("cpu");
        print("using CPU")
    print("Device ? : {0}".format(device))

    model = MODEL_SIGMOID(hidden=hidden).to(device)
    # model = MODEL_TANH(hidden=hidden).to(device)
    # model.apply(init_weights)

    # model = MODEL_TANH(hidden=hidden).to(device)
    # model = MODEL_TANH(hidden=hidden).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)



    criterion = torch.nn.MSELoss(reduction="mean")
    best_train=1000
    best_val =1000
    sum_loss = [[],[],[]]
    model.train()
    for epoch in range(epochs):
        print("{0} Epoch Start!!".format(epoch+1))
        total_loss=0
        train_loss = 0
        for idx, data in enumerate(trainloader):
            input,label = data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            loss = torch.sqrt(loss)
            total_loss += loss.item()
            train_loss = total_loss / (idx+1)
        scheduler.step()
        sum_loss[0].append(train_loss)
        print("{0} epoch Train end.".format(epoch+1))


        model.eval()
        with torch.no_grad():
            total_v_loss = 0; val_loss=0
            for idx,data in enumerate(valloader):
                input, label = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                output = model(input)
                loss = torch.sqrt(criterion(output,label))
                total_v_loss += loss.item()
                val_loss = total_v_loss/(idx+1)
        sum_loss[1].append(val_loss)
        print("{0} epoch Validation end.".format(epoch + 1))
        print("Train Loss {0} Valid Loss {1}".format(train_loss,val_loss))
        if best_train > train_loss:
            best_train = train_loss
        if best_val > val_loss:
            best_val = val_loss
    torch.save(model, dataset + "_"+str(hidden)+"_model")
    model.eval()
    total_test_loss =0
    test_loss =0
    for idx, data in enumerate(testloader):
        input, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = torch.sqrt(criterion(output,label))
        total_test_loss += loss.item()
        test_loss = total_test_loss / (idx + 1)

        dd = input.cpu().data.numpy()
        d = output.cpu().data.numpy()
        c = np.hstack([dd, d])
        if idx !=0:
            result_ = np.vstack([c, result_])
        else:
            result_ = c
    plot_output(result_,"Hidden : "+str(hidden),save_)
    sum_loss[2].append(test_loss)
    print("Test loss : {0}".format(test_loss))
    return train_loss,val_loss,test_loss,sum_loss




def read_data(name):
    #name = "./data/FOLD#/CV#/
    trainset = np.loadtxt(name+"trainset.txt")
    valset = np.loadtxt(name + "valset.txt")
    testset = np.loadtxt(name[:-4] + "testset.txt")
    return trainset,valset,testset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="helper")
    parser.add_argument("--lr", default=0.05,type=float, help="learning rate?")
    parser.add_argument("--dataset", default="./data/FOLD2/CV3/", help="data?")
    parser.add_argument("--batch", type=int, default=1024, help="batch?")
    parser.add_argument("--epochs", type=int, default=100, help="epochs?")
    parser.add_argument("--hidden", type=int, default=8, help="hidden?")

    args = parser.parse_args()
    print("""
    Dataset :{0}
    Batch : {1}
    Epoch : {2}
    hidden : {3}
    LR     : {4}
    """.format(args.dataset,args.batch,args.epochs,args.hidden,str(args.lr)))
    # input("START ? :")
    title = "Hidden : {0} / Dataset: {1} ".format(args.hidden,args.dataset)
    trainset,valset,testset = read_data(args.dataset)
    train_loss,val_loss,test_loss,sum_loss = train(
        trainset,valset,testset,args.hidden,
        args.batch,args.epochs,args.dataset,args.dataset+str(args.hidden)+"_plane.png",lr=args.lr)
    plot_loss(sum_loss,args.epochs,title,args.dataset+str(args.hidden)+"_loss.png")

    print("""
        Dataset :{0}
        Batch : {1}
        Epoch : {2}
        hidden :{3}
        LR     : {4}
    """.format(args.dataset,args.batch,args.epochs,args.hidden,str(args.lr)))
    print(" \nTrain End : BEST Train Loss {0} Valid Loss {1}".format(train_loss, val_loss,test_loss))
    print(train_loss)
    print(val_loss)
    print(test_loss)
