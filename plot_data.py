import matplotlib.pyplot as plt
import numpy as np

def plot_output2(data,title,save_):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



def plot_output(data, title,save_):
    x,y,z = data[:,0],data[:,1],data[:,2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter3D(x, y, z, c=z, cmap='Greens')
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.suptitle(title, fontsize=16)
    plt.show()
    plt.savefig(save_,dpi=300)
    plt.clf()


def plot_loss(losses,epochs,title,save_):
    train_loss,val_loss,test_loss= losses
    tag_ = ["train","val","test"]
    x = [i for i in range(1, epochs+1, 1)]
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x, train_loss,label=tag_[0],marker="o")
    plt.plot(x, val_loss,alpha=5, label=tag_[1],marker="s")
    plt.title(title)
    # plt.legend()
    plt.show()
    plt.savefig(save_, dpi=300)
    plt.clf()
    print('Finish Draw Loss graph')



