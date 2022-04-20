import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Subset
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader

import os
import argparse
import threading

BATCH_SIZE = 60
EPOCHS = 10
LEARNING_RATE = 0.1
MOMENTUM = 0.5
LOG_INTERVAL = 10



class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # remote control to network2
        self.rref1 = rpc.remote(
            'worker1',
            Network2,
            timeout = 0
        )

    def forward(self, x):
        # Server 1:
        # Convolutional Layer/Pooling Layer/Activation
        z1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Convolutional Layer/Dropout/Pooling Layer/Activation
        z2 = self.conv2(z1)
        z3 = F.relu(F.max_pool2d(self.conv2_drop(z2), 2))
        z3 = z3.view(-1, 320)
        x_rref = RRef(z3)
        # recive output data from server2
        out_futures = self.rref1.rpc_sync().forward(x_rref)
        return out_futures
    
    def parameter_rrefs(self):
        remote_params = []
        # server1 parameters
        remote_params.extend([RRef(p) for p in self.parameters()])
        # server2 parameters
        remote_params.extend(self.rref1.remote().parameter_rrefs().to_here())
        return remote_params

class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

        self._lock = threading.Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, z3):
        # recive data from server1
        x = z3.to_here().to(self.device)
        with self._lock:
            # Server 2:
            # Fully Connected Layer/Activation
            z4 = F.dropout(F.relu(self.fc1(x)))
            # Fully Connected Layer/Activation
            z5 = self.fc2(z4)
            # Softmax gets probabilities
            output = F.log_softmax(z5, dim=1)
        return output.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


def run_master():
    train_data = datasets.MNIST(root='data',train=True,download=True,transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='data',train=False,download=True,transform=transforms.ToTensor())

    # decrease train and test size
    train_data = Subset(train_data, indices=range(len(train_data) // 10))
    test_data = Subset(test_data, indices=range(len(test_data) // 10))

    train_loader = DataLoader(train_data,batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data,batch_size=BATCH_SIZE)

    model = Network1()
    model.cpu()

    dist_optim = DistributedOptimizer(
        optimizer_class=optim.SGD,
        params_rref=model.parameter_rrefs(),
        lr=LEARNING_RATE,momentum=MOMENTUM,
    )
    
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with dist_autograd.context() as context_id:
                output = model(data)
                loss = [F.nll_loss(output, target)]
                dist_autograd.backward(context_id, loss)
                dist_optim.step(context_id)
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss[0]))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).data # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)))

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Distributed Machine Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--rank', type=int, metavar='R',
        help="""Number of rank""")
    parser.add_argument('--world_size', type=int, default=2, metavar='N',
        help="""Number of workers""")
    parser.add_argument('--interface', type=str, default="eth0", metavar='I',
        help="""Interface that current device is listening on. It will default to eth0 if 
        not provided.""")
    parser.add_argument('--master_addr', type=str, default="localhost", metavar='MA',
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument('--master_port', type=str, default="29500", metavar='MP',
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")


    args = parser.parse_args()
    assert args.rank is not None, "Must provide rank argument."

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['GLOO_SOCKET_IFNAME'] = args.interface
    os.environ["TP_SOCKET_IFNAME"] = args.interface

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=0)

    if args.rank == 0:
        rpc.init_rpc(
            name="master",
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options
        )
        run_master()
    else:
        rpc.init_rpc(
            name=f"worker{args.rank}",
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options
        )
        pass

    rpc.shutdown()