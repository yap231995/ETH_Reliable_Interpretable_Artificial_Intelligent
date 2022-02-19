import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter
from model import Net, ConvNet
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

import matplotlib.pyplot as plt


np.random.seed(42)
torch.manual_seed(42)



# loading the dataset
# note that this time we do not perfrom the normalization operation, see next cell
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307)/0.3081

# we load the body of the neural net trained with mnist_train.ipynb...
model = torch.load('model.net', map_location='cpu') 

# ... and add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), model)

# and here we also create a version of the model that outputs the class probabilities
model_to_prob = nn.Sequential(model, nn.Softmax())

# we put the neural net into evaluation mode (this disables features like dropout)
model.eval()
model_to_prob.eval()


# define a show function for later
def show(original, adv, model_to_prob):
    p0 = model_to_prob(original).detach().numpy()
    p1 = model_to_prob(adv).detach().numpy()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[0].set_title("Original, class: " + str(p0.argmax()))
    axarr[1].imshow(adv.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[1].set_title("Adversial, class: " + str(p1.argmax()))
    print("Class\t\tOrig\tAdv")
    for i in range(10):
        print("Class {}:\t{:.2f}\t{:.2f}".format(i, float(p0[:, i]), float(p1[:, i])))


def fgsm_targeted(model, x, target, eps):
    input_ = x.clone().detach_() #create copy. detach does is to stop all previous computation into the gradient graph
    input_.requires_grad_()
    L = nn.CrossEntropyLoss()
    model.zero_grad()
    target = torch.LongTensor([target])
    loss = L(model(input_),target)
    loss.backward()
    output=input_-eps*torch.sign(input_.grad)
    return output

def fgsm_untargeted(model, x, label, eps):
    input_ = x.clone().detach_()  # create copy. detach does is to stop all previous computation into the gradient graph
    input_.requires_grad_()
    L = nn.CrossEntropyLoss()
    model.zero_grad()
    target = torch.LongTensor([label])
    loss = L(model(input_), target)
    loss.backward()
    output = input_ + eps * torch.sign(input_.grad)
    return output

def pgd_targeted(model, x, target, k, eps, eps_step):
    # TODO: implement
    return x

def pgd_untargeted(model, x, label, k, eps, eps_step, clip_min, clip_max):
    # TODO: implement
    return x



# try out our attacks
original = torch.unsqueeze(test_dataset[0][0], dim=0)

# adv_fgsm_targeted = fgsm_targeted(model, original, 7, 20)

label = torch.argmax(model(original))
print(int(label))
adv_fgsm_untargeted = fgsm_untargeted(model, original, int(label), 10)
# adv_pgd_targeted = pgd_targeted(model, original, 7, 10, 0.08, 0.05)
# adv_pdg_untargeted = pgd_untargeted(model, original, 7, 10, 0.08, 0.05, clip_min=0, clip_max=1.0)

show(original, adv_fgsm_untargeted, model_to_prob)
plt.show()
