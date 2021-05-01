import numpy as np
import torch
import os
def save_txt(name,data):
    f = open(name,"w")
    for i in data:
        input_ = str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n"
        f.writelines(input_)
    f.close()
def split_fold(save_,data):
    make_floders = [save_+"CV1",save_+"CV2",save_+"CV3"]
    for folder in make_floders:
        os.mkdir(folder)
    total = len(data)
    # FOLD#/CV1
    trainset = data[int(total * 3 / 17):, :]
    valset = data[0:int(total * 3/17), :]
    save_train = save_ +"CV1"+"/trainset.txt"
    save_val = save_ +"CV1"+"/valset.txt"
    save_txt(save_train,trainset)
    save_txt(save_val, valset)

    # FOLD#/CV2
    trainset = np.vstack([data[:int(total * 3 / 17):, :] ,data[int(total * 6 / 17):, :]])
    valset = data[int(total * 3 / 17):int(total * 6 / 17), :]
    save_train = save_ + "CV2" + "/trainset.txt"
    save_val = save_ + "CV2" + "/valset.txt"
    save_txt(save_train, trainset)
    save_txt(save_val, valset)

    # FOLD#/CV3
    trainset = data[0:int(total * 14 / 17):, :]
    valset = data[int(total * 14 / 17):, :]
    save_train = save_ + "CV3" + "/trainset.txt"
    save_val = save_ + "CV3" + "/valset.txt"
    save_txt(save_train, trainset)
    save_txt(save_val, valset)

    print(save_+" FINISH")



def split_cv(name):
    data = np.loadtxt(name)
    total = len(data)

    #FOLD1
    learningset = data[0:int(total * 0.85), :]
    testset = data[int(total * 0.85):, :]
    save_txt("./data/FOLD1/learningset.txt", learningset)
    save_txt("./data/FOLD1/testset.txt", testset)
    print("FOLD1 FINISH")
    split_fold("./data/FOLD1/",learningset)

    #FOLD2
    learningset = np.vstack([data[:int(total * 0.15), :] , data[int(total * 0.3):, :]])
    testset = data[int(total * 0.15):int(total*0.3), :]
    save_txt("./data/FOLD2/learningset.txt", learningset)
    save_txt("./data/FOLD2/testset.txt", testset)
    print("FOLD2 FINISH")
    split_fold("./data/FOLD2/", learningset)

    #FOLD3
    learningset = data[int(total * 0.15):, :]
    testset = data[:int(total * 0.15), :]
    save_txt("./data/FOLD3/learningset.txt", learningset)
    save_txt("./data/FOLD3/testset.txt", testset)
    print("FOLD3 FINISH")
    split_fold("./data/FOLD3/", learningset)


if __name__ == "__main__":
    split_cv("./data/total.txt")
