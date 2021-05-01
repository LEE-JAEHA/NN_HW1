import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def fxy(x, y):
    return (-1) * ((1 + np.cos(12 * np.sqrt(x ** 2 + y ** 2))) / (0.5 * (x ** 2 + y ** 2) + 2))


def plot(x, y, z, title):
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    plot_ = fig.gca(projection='3d')
    plot_.plot_surface(x, y, z, rstride=1, cstride=1, cmap="summer")  # rsride cstride => row column
    plot_.set_title(title)
    plt.show()

def plot_output(data, title,save_):
    x,y,z = data[:,0],data[:,1],data[:,2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='Greens')
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.suptitle(title, fontsize=16)
    plt.show()
    # plt.savefig(save_,dpi=300)

def make_data():
    x_ = np.linspace(-2,0,num=250)
    y_ = np.linspace(-2,0,num=250)
    x,y = np.meshgrid(x_,y_)
    x2 = np.linspace(-2, 2, num=500)
    y2 = np.linspace(0, 2, num=250)
    x2, y2 = np.meshgrid(x2, y2)
    x3 = np.linspace(0, 2, num=250)
    y3 = np.linspace(0, -2, num=250)
    x3, y3 = np.meshgrid(x3, y3)

    z = fxy(x,y)
    #gaussian noise
    # z1= fxy(x,y)+0.01*np.random.randn(500,500)
    # z2 = fxy(x, y) + 0.02 * np.random.randn(500, 500)
    # z3 = fxy(x, y) + 0.03 * np.random.randn(500, 500)
    # z4 = fxy(x, y) + 0.04 * np.random.randn(500, 500)
    z5 = fxy(x, y) + 0.05 * np.random.randn(250, 250)
    z5_2 = fxy(x2, y2)
    z5_3 = fxy(x3, y3)

    plot(x_,y_,z,"f(x,y) without Gaussian Noise")
    # plot(x_, y_, z1, "f(x,y) without Gaussian Noise 0.01")
    # plot(x_, y_, z2, "f(x,y) without Gaussian Noise 0.02")
    # plot(x_, y_, z3, "f(x,y) without Gaussian Noise 0.03")
    # plot(x_, y_, z4, "f(x,y) without Gaussian Noise 0.04")
    # plot(x_, y_, z5, "f(x,y) with Gaussian Noise 0.05")
    a = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z5.reshape(-1, 1)])
    b = np.hstack([x2.reshape(-1,1),y2.reshape(-1,1),z5_2.reshape(-1,1)])
    c = np.hstack([x3.reshape(-1, 1), y3.reshape(-1, 1), z5_3.reshape(-1, 1)])
    d = np.vstack([a,b,c])
    np.random.shuffle(d)
    plot_output(d, "Gaussian Noises of f(x,y)","./s")
    return d


def divide_data(data):
    np.random.shuffle(data)
    total = len(data)
    trainset = data[0:int(total*0.7),:]
    valset = data[int(total*0.7):int(total*0.85),:]
    testset = data[int(total*0.85):,:]
    return trainset,valset,testset

def write(name,data):
    import os;
    if not os.path.exists("./data") :
        os.mkdir("./data")
    save_ = "./data/"+name
    f = open(save_,"w")

    for i in data:
        input_ = str(i[0])+" "+str(i[1])+" "+str(i[2])+"\n"
        f.writelines(input_)
    print(name + " FINISH")


if __name__ == "__main__":
    data = make_data()
    write("total.txt", data)
    # trainset,valset,testset = divide_data(data)
    # write("trainset.txt",trainset)
    # write("testset.txt", valset)
    # write("valset.txt", testset)
    # import pdb;pdb.set_trace()


