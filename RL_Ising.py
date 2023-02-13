import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import os
import numba as nb

spec = [
    ('S', nb.int32[:, :]),
    ('L', nb.int32),
    ('epsilon', nb.float32),
    ('alpha', nb.float32),
    ('Q', nb.float32[:, :, :, :]),
]

@nb.experimental.jitclass(spec)
class Simulation():
    def __init__(self, L, S, epsilon, Q, alpha=0.05):
        self.L = L
        self.epsilon = epsilon
        self.S = S
        self.alpha = alpha
        self.Q = Q

    def update(self, time):
        for _ in range(time):
            for x in range(self.L):
                for y in range(self.L):
                    # 判断属于大多数or少数
                    near = self.S[(x+1)%self.L, y]+self.S[(x-1)%self.L, y]+self.S[x, (y+1)%self.L]+self.S[x, (y-1)%self.L] #将0也划分为大多数
                    if near == 0 : stat = 1
                    else : stat = int((np.sign(self.S[x, y]*near)+1)/2)

                    # 选择是否翻转
                    if np.random.random() < self.epsilon :
                        action = np.random.choice(2)
                        if action == 1 : self.S[x, y] = -1 * self.S[x, y]
                    else:
                        if self.Q[x, y, stat, 1] < self.Q[x, y, stat, 0]:
                            self.S[x, y] = -1 * self.S[x, y]
                            action = 1
                        else: action = 0

                    # 更新
                    if int((np.sign(self.S[x, y]*near)+1)/2) == 1 or near == 0 : c = 0
                    else: c = 1

                    self.Q[x, y, stat, action] = self.Q[x, y, stat, action] + self.alpha*(c-self.Q[x, y, stat, action])

def run(L, alpha, n_epsilon, PATH, loop=200, realize=1000, warm=10000):
    S = np.ones((L, L), dtype=np.int32)
    for i in range(L):
        for j in range(L):
            if np.random.choice((0, 1)):
                S[i, j] *= -1
    Q = np.zeros((L, L, 2, 2), dtype=np.float32)
    PATH = PATH + "/L=%d/alpha=%.3f" %(L, alpha)
    os.makedirs(PATH, exist_ok=True)
    epsilon = 1.0

    epsilon_list = np.linspace(0.20, 0.16, n_epsilon)
    M_list = np.zeros(n_epsilon)

    Ising = Simulation(L, S, epsilon, Q, alpha)
    for mark, epsilon in enumerate(epsilon_list):
        Ising.Q = np.zeros((L, L, 2, 2), dtype=np.float32)
        Ising.epsilon = epsilon
        Ising.update(warm)

        for num in range(loop):
            '''
            with open(local_PATH+'/S_%d.pickle' %num, 'wb') as f:
                pickle.dump(Ising.S, f)
            '''
            M_list[mark] += np.abs(np.sum(Ising.S)/L**2)
            Ising.update(realize)

        M_list[mark] = M_list[mark]/loop

    with open(PATH+'/M.pickle', 'wb') as f:
        pickle.dump(M_list, f)
    with open(PATH+'/epsilon.pickle', 'wb') as f:
        pickle.dump(epsilon_list, f)

    plt.plot(epsilon_list, M_list)
    plt.savefig(PATH+'/M_epsilon.png')
    plt.close()


if __name__ == "__main__":
    pool = Pool(10)

    L_list = [16, 32, 64, 96]
    alpha_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    n_epsilon = 50
    PATH = os.getcwd()+'/data'
    loop = 100000
    realize = 200
    warm = 10000

    for L in L_list:
        for alpha in alpha_list:
            print('L=%d, alpha=%.4f'%(L, alpha))
            pool.apply_async(run, (L, alpha, n_epsilon, PATH, loop, realize, warm))
    pool.close()
    pool.join()