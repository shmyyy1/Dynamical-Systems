import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm import tqdm
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Channel(object):
    def __init__(self, d0, gamma, s, N0):
        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0
    
    def pathloss(self, d):
        
        d = np.asarray(d)
        d = np.maximum(d, 1e-9)
        return (self.d0 / d) ** self.gamma

    def fading_channel(self,d, Q):
       
        Exp_h = (self.d0 / d) ** self.gamma
        h_til = np.random.exponential(self.s, size=(1, Q))
        h = Exp_h * h_til / self.s
        return h

    def build_fading_capacity_channel(self, h, p):
        return np.log(1 + h * p / self.N0)
    
    def capacity(self, h, p):
        return np.log(1.0 + h * p / self.N0)
    
    def build_fading_capacity_channel(self, h, p):
        return np.log(1.0 + h * p / self.N0)

import numpy as np
import matplotlib.pyplot as plt

d = np.arange(1, 101)   

d0 = 1.0        
gamma = 2.2
s = 2.0
N0 = 1e-6       

channel = Channel(d0, gamma, s, N0)
Eh = channel.pathloss(d)

plt.figure()
plt.plot(d, Eh)
plt.xlabel("Distance d (m)")
plt.ylabel("E[h]")
plt.title("Pathloss E[h] vs distance (linear scale)")
plt.grid(True)

plt.figure()
plt.semilogy(d, Eh)
plt.xlabel("Distance d (m)")
plt.ylabel("E[h]")
plt.title("Pathloss E[h] vs distance (log scale)")
plt.grid(True)

plt.show()

import numpy as np
import matplotlib.pyplot as plt


dists = np.arange(1, 101)


d0 = 1.0       
gamma = 2.2
s = 2.0
N0 = 1e-6      
p = 0.05       
Q = 100        

channel = Channel(d0, gamma, s, N0)

cap_mean = np.zeros_like(dists, dtype=float)
cap_std  = np.zeros_like(dists, dtype=float)

for i, d in enumerate(dists):
    
    h_samples = channel.fading_channel(d, Q)      
    h_samples = np.asarray(h_samples).ravel()     

    c_samples = channel.build_fading_capacity_channel(h_samples, p)

    cap_mean[i] = np.mean(c_samples)
    cap_std[i]  = np.std(c_samples)

plt.figure()
plt.errorbar(dists, cap_mean, yerr=cap_std, fmt='o-', capsize=3)
plt.xlabel("Distance d (m)")
plt.ylabel("Capacity c(h)")
plt.title("Channel capacity vs distance (mean ± std over Q=100)")
plt.grid(True)
plt.show()

Q = 100
dist = np.arange(1, 101)          

h_sim = np.zeros((len(dist), Q))  

for i, d in enumerate(dist):
   
    h_samples = channel.fading_channel(d, Q)   
    h_sim[i, :] = np.asarray(h_samples).ravel()

h_mean = np.mean(h_sim, axis=1)
h_var  = np.var(h_sim, axis=1)

plt.figure()
plt.errorbar(dist, h_mean, yerr=h_var, fmt='o-', ecolor="orange", capsize=3)
plt.xlabel("Distance (m)")
plt.ylabel("Fading h(d)")
plt.title("Fading channel vs distance (mean ± var, Q=100)")
plt.grid(True)
plt.savefig("FadingChannel.png", dpi=200)
plt.show()

cap = channel.build_fading_capacity_channel(h_sim, 0.05)

cap_mean = np.mean(cap, axis = 1)
cap_var = np.var(cap, axis = 1)

plt.figure()
plt.errorbar(dist, cap_mean, cap_var, ecolor="orange")
plt.ylabel('Capacity')
plt.xlabel('Distance')
plt.title('Channel Capacity vs. Distance')
plt.savefig('ChannelCapacities.png', dpi = 200)
plt.show()


p = 0.05      
cap = channel.build_fading_capacity_channel(h_sim, p)  

cap_mean = np.mean(cap, axis=1)
cap_std  = np.std(cap, axis=1)

plt.figure()
plt.errorbar(dist, cap_mean, yerr=cap_std, fmt='o-', ecolor="orange", capsize=3)
plt.xlabel("Distance (m)")
plt.ylabel("Capacity c(h)")
plt.title("Channel capacity vs distance (mean ± std, Q=100)")
plt.grid(True)
plt.savefig("ChannelCapacities.png", dpi=200)
plt.show()

class WirelessNetwork(object):
    def __init__(self, wx, wy, wc, n, d0, gamma, s, N0):
    
        self.wx = wx
        self.wy = wy
        self.wc = wc
        self.n = n

        self.t_pos, self.r_pos = self.determine_positions()

        self.dist_mat = distance_matrix(self.t_pos, self.r_pos)

        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0

        self.channel = Channel(self.d0, self.gamma, self.s, self.N0)

    def determine_positions(self):
        t_x_pos = np.random.uniform(0, self.wx, (self.n, 1))
        t_y_pos = np.random.uniform(0, self.wy, (self.n, 1))
        t_pos = np.hstack((t_x_pos, t_y_pos))
        r_distance = np.random.uniform(0, self.wc, (self.n, 1))
        r_angle = np.random.uniform(0, 2 * np.pi, (self.n, 1))
        r_rel_pos = r_distance * np.hstack((np.cos(r_angle), np.sin(r_angle)))
        r_pos = t_pos + r_rel_pos
        return t_pos, r_pos

    def generate_pathloss_matrix(self):
        return self.channel.pathloss(self.dist_mat)

    def generate_interference_graph(self, Q):
        return self.channel.fading_channel(self.dist_mat, Q)

    def generate_channel_capacity(self, p, H):
        num = torch.diagonal(H, dim1=-2, dim2=-1) * p
        den = H.matmul(p.unsqueeze(-1)).squeeze() - num + self.N0
        return torch.log(1 + num / den)

    def plot_network(self):
        plt.scatter(self.t_pos[:,0], self.t_pos[:,1], s = 4, label = "Transmitters")
        plt.scatter(self.r_pos[:,0], self.r_pos[:,1], s = 4, label = "Receivers", c = "orange")
        plt.xlabel("Area Length")
        plt.ylabel("Area Width")
        plt.title("Wireless Network")
        plt.savefig('WirelessNetwork.png', dpi = 200)
        plt.legend()
        return plt.show()

d0 = 1.0
gamma = 2.2
s = 2.0
N0 = 1e-6

rho = 0.05      
wx = 200     
wy = 100      
wc = 50         

n = int(rho * wx * wy)  
net = WirelessNetwork(wx, wy, wc, n, d0, gamma, s, N0)
net.plot_network()


class Generator:
    def __init__(self, n, wx, wy, wc, d0=1, gamma=2.2, s=2, N0=1, device="cpu", batch_size=64, random=False):

        self.n = n
        self.wx = wx
        self.wy = wy
        self.wc = wc

        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0

        self.device = device
        self.batch_size = batch_size

        self.random = random

        self.train = None
        self.test = None

        self.network = WirelessNetwork(self.wx, self.wy, self.wc, self.n, self.d0,
                                       self.gamma, self.s, self.N0)
        self.H1 = self.network.generate_pathloss_matrix()

    def __next__(self):
        if self.random:

            self.network = WirelessNetwork(self.wx, self.wy, self.wc, self.n, self.d0,
                                       self.gamma, self.s, self.N0)
            self.H1 = self.network.generate_pathloss_matrix()
        H2 = np.random.exponential(self.s, (self.batch_size, self.n, self.n))
        H = self.H1 * H2
        eigenvalues, _ = np.linalg.eig(H)
        S = H / np.max(eigenvalues.real)

        H = torch.from_numpy(H).to(torch.float).to(self.device)
        S = torch.from_numpy(S).to(torch.float).to(self.device)
        return H, S, self.network

def train(model, update, generator, iterations):
    pbar = tqdm(range(iterations), desc=f"Training for n={generator.n}")
    for i in pbar:
        H, S, network = next(generator)
        p = model(S)
        c = network.generate_channel_capacity(p, H)
        update(p, c)
        pbar.set_postfix({'Capacity Mean': f" {c.mean().item():.3e}",
                          'Capacity Var': f" {c.var().item():.3e}",
                          'Power Mean': f" {p.mean().item():.3f}",
                          'Power Var': f" {p.var().item():.3f}"})

def test(model: Callable[[torch.Tensor], torch.Tensor], generator: Generator, iterations=100):
    with torch.no_grad():
        powers = []
        capacities = []
        loss = []
        for i in tqdm(range(iterations), desc=f"Test for n={generator.n}"):
            H, S, network = next(generator)
            p = model(S)
            c = network.generate_channel_capacity(p, H)
            loss.append(criterion(p, c).item())
            capacities.append(c.mean().item())
            powers.append(p.mean().item())
        print()
        print("Testing Results:")
        print(f"\tLoss mean: {np.mean(loss):.4f}, variance {np.var(loss):.4f}"
            f"| Capacity mean: {np.mean(capacities):.4e}, variance {np.var(capacities):.4e}"
            f"| Power mean: {np.mean(powers):.4f}, variance {np.var(powers):.4f}")

class GraphFilter(nn.Module):
    def __init__(self, k: int, f_in=1, f_out=1):
        super().__init__()
        self.k = k
        self.f_in = f_in
        self.f_out = f_out

        self.weight = nn.Parameter(torch.ones(self.k, self.f_in, self.f_out))
        self.bias = nn.Parameter(torch.zeros(self.f_out,))

        torch.nn.init.normal_(self.weight, 0.3, 0.1)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor):
        B = x.shape[0] 
        N = S.shape[-1] 

        x = x.reshape([B, N, self.f_in])
        S = S.reshape([B, N, N])

        y = x @ self.weight[0]
        for k in range(1, self.k):
            x = x.permute(0, 2, 1)  
            x = torch.matmul(x, S)  
            x = x.permute(0, 2, 1)  
            y += x @ self.weight[k]

        if self.bias is not None:
            y = y + self.bias
        return y

class GraphNeuralNetwork(nn.Module):
    def __init__(self,
                 ks: Union[List[int], Tuple[int]] = (5,),
                 fs: Union[List[int], Tuple[int]] = (1, 1)):
        super().__init__()
        self.n_layers = len(ks)

        self.layers = []
        for i in range(self.n_layers):
            f_in = fs[i]
            f_out = fs[i + 1]
            k = ks[i]
            gfl = GraphFilter(k, f_in, f_out)
            activation = nn.ReLU() if i < self.n_layers - 1 else nn.Identity()
            self.layers += [gfl, activation]
            self.add_module(f"gfl{i}", gfl)
            self.add_module(f"activation{i}", activation)

    def forward(self, x, S):
        for i, layer in enumerate(self.layers):
            x = layer(x, S) if i % 2 == 0 else layer(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn

    def forward(self, S):
        batch_size = S.shape[0]
        n = S.shape[1]
        p0 = torch.ones(batch_size, n, device=device)
        p = self.gnn(p0, S).abs()
        return torch.squeeze(p)

mu_unconstrained = 0.01 
step_size = 0.01
unconstrained = Model(GraphNeuralNetwork([5, 5, 5], [1, 8, 4, 1]).to(device))
optimizer = torch.optim.Adam(unconstrained.parameters(), step_size)
criterion = lambda p, c: -c.mean() + mu_unconstrained * p.mean()

def update_unconstrained(p, c):
    global mu, optimizer
    optimizer.zero_grad()
    criterion(p, c).backward()
    optimizer.step()

N0 = 1e-6
train_iterations = 200
test_iterations = 100
batch_size = 100

generator_small = Generator(160, 80, 40, 20, device=device, N0=N0, batch_size=batch_size)
train(unconstrained, update_unconstrained, generator_small, train_iterations)
test(unconstrained, generator_small, test_iterations)

generator_random = Generator(160, 80, 40, 20, device=device, N0=N0, batch_size=batch_size, random=True)
test(unconstrained, generator_random, test_iterations)

generator_large = Generator(360, 120, 60, 30, device=device, N0=N0, batch_size=batch_size, random=True)
test(unconstrained, generator_large, test_iterations)

pmax = 1e-3
primal_step = 0.01
dual_step = 0.001

def update_constrained(p, c):
    global mu, optimizer
    optimizer.zero_grad()

    # primal step
    (mu.unsqueeze(0) * (p - pmax) - c).mean().backward()
    optimizer.step()
    # dual step
    with torch.no_grad():
        mu = torch.relu(mu + dual_step * torch.mean((p - pmax), 0))

pmax = 1e-3
primal_step = 0.01
dual_step = 0.001

mu = torch.zeros(generator_small.n, device=device)
constrained = Model(GraphNeuralNetwork([5, 5, 5], [1, 8, 4, 1]).to(device))
optimizer = torch.optim.Adam(constrained.parameters(), primal_step)

train(constrained, update_constrained, generator_small, train_iterations)
test(constrained, generator_small, test_iterations)

test(constrained, generator_random, test_iterations)
test(constrained, generator_large, test_iterations)