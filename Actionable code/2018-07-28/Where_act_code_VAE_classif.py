batch_size = 50
test_batch_size = 1
valid_size = .2
epochs = 100
lr = 0.001
momentum = 0.48
no_cuda = True
num_processes = 1
seed = 42
log_interval = 10
size = 64
LATENT_DIM = 32
mean = 0.
std = 1.
dimension = 25
verbose = False

NB_LABEL = 10
#HIDDEN_SIZE = 1024 #256
BIAS_CONV = True
BIAS_POS = False
BIAS_LABEL = False
BIAS_TRANSCODE = True
BIAS_DECONV = True


import os
import numpy as np
import torch
#torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

from scipy.stats import multivariate_normal

import torch.optim as optim

class Data:
    def __init__(self, args):

        ## Charger la matrice de certitude
        path = "MNIST_accuracy.npy"
        if os.path.isfile(path):
            self.accuracy =  np.load(path)
            if verbose:
                print('Loading accuracy... min, max=', self.accuracy.min(), self.accuracy.max())
        else:
            print('No accuracy data found.')

        kwargs = {'num_workers': 1, 'pin_memory': True} if not args.no_cuda else {'num_workers': 1, 'shuffle': True}

        self.data_loader = torch.utils.data.DataLoader(
                datasets.MNIST('/tmp/data',
                               train=True,     # def the dataset as training data
                               download=True,  # download if dataset not present on disk
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(args.mean,), std=(args.std,))])),
                               batch_size=batch_size,
                               **kwargs)

        self.args = args
        # GPU boilerplate
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        # if self.args.verbose:
        #     print('cuda?', self.args.cuda)
        #
    def show(self, gamma=.5, noise_level=.4, transpose=True):

        images, foo = next(iter(self.data_loader))
        from torchvision.utils import make_grid
        npimg = make_grid(images, normalize=True).numpy()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=((13, 5)))
        import numpy as np
        if transpose:
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
        else:
            ax.imshow(npimg)
        plt.setp(ax, xticks=[], yticks=[])

        return fig, ax


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
            
         # Visual input dim 256 encoding
        self.conv1 = nn.Conv2d(1, 16, 4, bias = BIAS_CONV, stride=4, padding=0)
        self.conv2 = nn.Conv2d(16, 64, 4, bias = BIAS_CONV, stride=4, padding=0) # dummy
        self.conv3 = nn.Conv2d(64, 256, 4, bias = BIAS_CONV, stride=4, padding=0)         
#         if BIAS_CONV == True:
#             self.conv1.bias.data.fill_(0)    
#             self.conv2.bias.data.fill_(0)    
#             self.conv3.bias.data.fill_(0)

        # Mu, logVar Category encoding
        self.fc_mu_d1 = nn.Linear(256, LATENT_DIM, bias = False)
        self.fc_logvar_d1 = nn.Linear(256, LATENT_DIM, bias = False)     
        self.fc_mu_d2 = nn.Linear(256, LATENT_DIM, bias = False)
        self.fc_logvar_d2 = nn.Linear(256, LATENT_DIM, bias = False)
        
        # Mu, logVar position encoding
        self.fc_x = nn.Linear(256, 32, bias = False)        
        self.fc_mu = nn.Linear(32, 2, bias = False)
        self.fc_logvar = nn.Linear(32, 2, bias = False)       
        
        # transcoding from feature space to visual space
        self.fc_transcode_d1 = nn.Linear(LATENT_DIM, 256, bias = BIAS_TRANSCODE)
        self.fc_transcode_d2 = nn.Linear(LATENT_DIM, 256, bias = BIAS_TRANSCODE)
#         if BIAS_TRANSCODE == True:
#             self.fc_transcode_d1.bias.data.fill_(0)
#             self.fc_transcode_d2.bias.data.fill_(0)
            
        '''self.deconv3 = nn.ConvTranspose2d(256, 64, 4, bias = BIAS_DECONV, stride=4, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 16, 4, bias = BIAS_DECONV, stride=4, padding=0) # dummy
        self.deconv1 = nn.ConvTranspose2d(16, 1, 4, bias = True, stride=4, padding=0)
#         if BIAS_DECONV == True:
#             self.deconv4.bias.data.fill_(0)
#             self.deconv3.bias.data.fill_(0)
#             self.deconv2.bias.data.fill_(0)
        self.deconv1.bias.data.fill_(0)'''
        
        self.fc_z = nn.Linear(256, 10, bias = False)
    
    def forward(self, x, z, u_in):  
        # Visual Input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x))   
        n = self.num_flat_features(x)
        x = x.view(-1, n)    
        
        # Category encoding (Linear)
        mu_d1 = self.fc_mu_d1(x)
        logvar_d1 = self.fc_logvar_d1(x)
        std_d1 = torch.exp(0.5*logvar_d1)
        eps_d1 = torch.randn_like(std_d1) #, requires_grad=True)
        z_d1 = eps_d1.mul(std_d1).add_(mu_d1)
        
        mu_d2 = self.fc_mu_d2(x)
        logvar_d2 = self.fc_logvar_d2(x)
        std_d2 = torch.exp(0.5*logvar_d2)
        eps_d2 = torch.randn_like(std_d2) #, requires_grad=True)
        z_d2 = eps_d2.mul(std_d2).add_(mu_d2)
        
        # position encoding
        x = F.relu(self.fc_x(x))                        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # Sample epsilon and generate u
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) #, requires_grad=True)        
        # Estimated position from epsilon sampling (dim 2)
        u_out = eps.mul(std).add_(mu)
                                
        # translation  (transformation)
        u_d1 = u_out[:,0].view(-1, 1)  #torch.cat((u[:,0],u[:,0],u[:,0],u[:,0],u[:,0],u[:,0],u[:,0],u[:,0]), 1)
        u_d2 = u_out[:,1].view(-1, 1)
        for _ in range(5):
            u_d1 = torch.cat((u_d1, u_d1), 1)
            u_d2 = torch.cat((u_d2, u_d2), 1)
        
        # transcoding (Linear)
        transfo_d1 = z_d1 - u_d1
        transfo_d2 = z_d2 - u_d2        
        z_hat = F.relu(self.fc_transcode_d1(transfo_d1) + self.fc_transcode_d2(transfo_d2))
        z_hat_logit = self.fc_z(z_hat)
        
        '''x_hat = x_hat.view(-1, 256, 1, 1)
        x_hat = F.relu(self.deconv3(x_hat))   
        x_hat = F.relu(self.deconv2(x_hat))   
        x_hat_logit = self.deconv1(x_hat)'''
        
        return mu_d1, logvar_d1, z_d1, mu_d2, logvar_d2, z_d2, mu, logvar, u_out, z_hat_logit, transfo_d1, transfo_d2  
            
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class ML():
    def __init__(self, args):
        self.args = args
        # GPU boilerplate
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        if self.args.verbose:
            print('cuda?', self.args.cuda)

        if self.args.cuda:
            self.model.cuda()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            
        # self.args.classes = self.dataset.classes
        # MODEL
        self.model = Net(self.args).to(self.device)
                
        # DATA
        self.dataset = Data(self.args)
        
        # LOSS
        #self.loss_func = torch.nn.BCEWithLogitsLoss(size_average=False)
        self.loss_func = torch.nn.CrossEntropyLoss(size_average=False)
        #self.loss_func = torch.nn.MSELoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr) #, weight_decay=0.0001*self.args.momentum)
        #self.optimizer = optim.SGD(self.model.parameters(),
        #                        lr=self.args.lr, momentum=self.args.momentum)
        
    # def forward(self, img):
    #     # normalize img
    #     return (img - self.mean) / self.std

    def train(self, path=None):
        # cosmetics
        try:
            from tqdm import tqdm_notebook as tqdm
            verbose = 1
        except ImportError:
            verbose = 0
        if self.args.verbose == 0 or verbose == 0:
            def tqdm(x, desc=None):
                if desc is not None: print(desc)
                return x

        # setting up training
        self.model.train()

        if path is not None:
            # using a data_cache
            import os
            import torch
            if os.path.isfile(path):
                self.model.load_state_dict(torch.load(path))
                print('Loading file', path)
            else:
                print('Training model...')
                for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
                    self.train_epoch(epoch, rank=0)
                    torch.save(self.model.state_dict(), path) #save the neural network state
                print('Model saved at', path)
        else:
            for epoch in tqdm(range(1, self.args.epochs + 1), desc='Train Epoch' if self.args.verbose else None):
                self.train_epoch(epoch, rank=0)

    def train_epoch(self, epoch, rank=0):
        torch.manual_seed(self.args.seed + epoch + rank*self.args.epochs)

        for batch_idx, (data, target) in enumerate(self.dataset.data_loader):
            # computes the couple
            data, target = data.to(self.device), target.to(self.device)
            data_full = np.zeros((batch_size, 1, self.args.size, self.args.size))
            label_full = np.zeros((batch_size, NB_LABEL))
            pos_full = np.zeros((batch_size, 2))

            for idx in range(batch_size):   
                mid = np.int(self.args.size / 2)
                draw = np.random.multivariate_normal((0,0),((1,0),(0,1)))
                i_offset = min(max(-mid, np.int(draw[0] * mid / 3)), mid)
                j_offset = min(max(-mid, np.int(draw[1] * mid / 3)), mid)
                #print(draw, i_offset, j_offset)
                data_full[idx, 0, :, :], label_full[idx, :], pos_full[idx,:] = couples_gen(data[idx, 0, :, :],
                                target[idx], 
                                i_offset, j_offset, size=self.args.size, contrast=1.)

            data_full = Variable(torch.DoubleTensor(data_full))
            label_full = target #Variable(torch.LongTensor(label_full))
            #label_full = Variable(torch.DoubleTensor(label_full))
            pos_full = Variable(torch.DoubleTensor(pos_full))
            # print(data.shape, data_full.shape)
            # Clear all accumulated gradients
            self.optimizer.zero_grad()
            # Predict classes using images from the train set
            mu_d1_output, logvar_d1_output, z_d1_output, mu_d2_output, logvar_d2_output, z_d2_output, mu_output, logvar_output, u_output, z_hat_logit_output = self.model(data_full, label_full, pos_full)[0:10]

            # print(output.shape, acc_full.shape)
            # Compute the loss based on the predictions and actual labels
            
            loss_z_hat = self.loss_func(z_hat_logit_output, label_full)
                  
            KL_FLAG = 1
            #KL_loss = -0.5 * (torch.sum(1 + logvar_output - mu_output.pow(2) - logvar_output.exp(), dim = 1)).mean() * KL_FLAG
            KL_loss_d1 = -0.5 * (torch.sum(1 + logvar_d1_output - mu_d1_output.pow(2) - logvar_d1_output.exp())) * KL_FLAG
            KL_loss_d2 = -0.5 * (torch.sum(1 + logvar_d2_output - mu_d2_output.pow(2) - logvar_d2_output.exp())) * KL_FLAG
            KL_loss = -0.5 * (torch.sum(1 + logvar_output - mu_output.pow(2) - logvar_output.exp())) * KL_FLAG
          
            # Backpropagate the loss
            loss = loss_z_hat + (KL_loss_d1 + KL_loss_d2 + KL_loss) / (64)
            loss.backward()
            # Adjust parameters according to the computed gradients
            self.optimizer.step()
            if self.args.verbose and self.args.log_interval>0:
                if batch_idx % self.args.log_interval == 0:
                    print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.2f}\tKL_d1: {:.2f}\tKL_d2: {:.2f}\tKL: {:.2f}'.format(
                        epoch, batch_idx * len(data), len(self.dataset.data_loader.dataset),
                        100. * batch_idx / len(self.dataset.data_loader), loss_z_hat.item(), 
                        KL_loss_d1.item(), KL_loss_d2.item(), KL_loss.item()))

    def test(self, dataloader=None):
        if dataloader is None:
            dataloader = self.dataset.data_loader
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            data_full = np.zeros((batch_size, 1, self.args.size, self.args.size))

            for idx in range(batch_size):
                i_offset = np.random.randint(self.args.size)
                j_offset = np.random.randint(self.args.size)
                data_full[idx, 0, :, :], pos[idx, :] = couples_gen(data[idx, 0, :, :],
                                i_offset, j_offset, size=self.args.size, contrast=1.)

            data_full, acc_full = Variable(torch.DoubleTensor(data_full)), Variable(torch.DoubleTensor(acc_full))

            output = self.model(data_full)

            # TODO FINISH ...

        if self.args.log_interval>0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataset.data_loader.dataset),
            100. * correct / len(self.dataset.data_loader.dataset)))
        return correct.numpy() / len(self.dataset.data_loader.dataset)

    def show(self, gamma=.5, noise_level=.4, transpose=True, only_wrong=False):

        data, target = next(iter(self.dataset.data_loader))
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        if only_wrong and not pred == target:
            print('target:' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in target))
            print('pred  :' + ' '.join('%5s' % self.dataset.dataset.classes[j] for j in pred))

            from torchvision.utils import make_grid
            npimg = make_grid(data, normalize=True).numpy()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=((13, 5)))
            import numpy as np
            if transpose:
                ax.imshow(np.transpose(npimg, (1, 2, 0)))
            else:
                ax.imshow(npimg)
            plt.setp(ax, xticks=[], yticks=[])

            return fig, ax
        else:
            return None, None

    def protocol(self, path=None):
        # TODO: make a loop for the cross-validation of results
        self.train(path=path)
        Accuracy = self.test()
        print('Test set: Final Accuracy: {:.3f}%'.format(Accuracy*100)) # print que le pourcentage de réussite final
        return Accuracy

def couples_gen(data, target,  i_offset, j_offset, size, contrast=1.):

    data_full = np.zeros((size, size))
    s_x, s_y = data.shape
    data_full[:s_x, :s_y] = data
    data_full = np.roll(data_full, -s_x//2 + i_offset + size // 2, axis=0)
    data_full = np.roll(data_full, -s_y//2 + j_offset + size // 2, axis=1)
    
    target_full = np.zeros(NB_LABEL, dtype = 'int')
    target_full[target] = 1
    
    pos_full = np.zeros(2)
    pos_full[0] = i_offset
    pos_full[1] = j_offset

    '''from scipy.signal import convolve2d
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])/4.
    data_full = convolve2d(data_full, kernel, mode='same')
    
    #noise_full = np.zeros((size, size))
    noise_full = np.random.randn(size ** 2).reshape(size, size)'''

    '''grid_x, grid_y = np.abs(np.mgrid[-size/2:size/2, -size/2:size/2]) * 8 / size
    pos = np.empty((size, size, 2))
    pos[:, :, 0] = grid_x; pos[:, :, 1] = grid_y
    d = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    mask = d.pdf(pos)
    mask2 = np.exp(-20 * mask)'''
    #_ = plt.imshow(mask2)
    
    '''grid_x, grid_y = np.mgrid[-size/2:size/2, -size/2:size/2] * 8 / size
    #dist = np.sqrt(grid_x **2 + grid_y**2)
    dist = 1
    noise_amp =  0.1
    data_full += noise_full * dist * noise_amp'''

    return data_full, target_full, pos_full


def init(batch_size=batch_size, test_batch_size=test_batch_size, valid_size=valid_size, epochs=epochs,
            lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
            log_interval=log_interval, size=size, mean=mean, std=std,
            dimension=dimension, verbose=verbose):
    # Training settings
    kwargs = {}
    kwargs.update(batch_size=batch_size, test_batch_size=test_batch_size, valid_size=valid_size, epochs=epochs,
                lr=lr, momentum=momentum, no_cuda=no_cuda, num_processes=num_processes, seed=seed,
                log_interval=log_interval, size=size, mean=mean, std=std,
                dimension=dimension, verbose=verbose)
    # print(kwargs)
    import easydict
    return easydict.EasyDict(kwargs)

if __name__ == '__main__':
    args = init_cdl()
    ml = ML(args)
    ml.main()
