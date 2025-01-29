from graph_tool.all import *

import numpy as np
from numpy.random import laplace
import pandas as pd

import networkx as nx

import community as community_louvain
import comm
import time
import random

import itertools
from heapq import *

from heapq import nlargest

#

import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse


def calc_all_densities(adj_matrix, block_list):

    block_labels, block_sizes = np.unique(block_list, return_counts=True)
    nblocks = len(block_labels)
    density_matrix = np.zeros((nblocks, nblocks))

    for i, j, v in zip(adj_matrix.row, adj_matrix.col, adj_matrix.data):
        density_matrix[block_list[i], block_list[j]] += 1

    for i in range(len(block_labels)):
        for j in range(len(block_labels)):
            density_matrix[block_labels[i], block_labels[j]] = np.max([density_matrix[block_labels[i], block_labels[j]], 1])/(block_sizes[i]*block_sizes[j])


    return(density_matrix)


def add_SBM(adj_matrix, block_list, density_matrix, epsilon):

    W_01 = np.zeros_like(density_matrix)
    W_11 = np.zeros_like(density_matrix)
    N_C = np.zeros_like(density_matrix)

    block_labels, block_sizes = np.unique(block_list, return_counts=True)
    for i in range(len(block_labels)):
        for j in range(len(block_labels)):
            p_00, p_11 = prob_from_eps_density(epsilon, density_matrix[i,j])
            N_C[i,j] = block_sizes[block_labels[i]]*block_sizes[block_labels[j]]
            W_01[i,j] = int((1-p_00)*N_C[i,j])
            W_11[i,j] = int(p_11*N_C[i,j])

    np.fill_diagonal(W_11, W_11.diagonal()*2)    # diagonal should have twice the avg num of edges

    g_keep = generate_sbm(block_list, W_11, micro_ers=True)   
    m_keep = get_coo_matrix(g_keep, symmetric=True)

    M_keep = adj_matrix.multiply(m_keep)
    M_keep.tocoo().data = np.ones_like(M_keep.tocoo().data)
    print(m_keep.sum(), M_keep.sum(), adj_matrix.sum())


    g_add = generate_sbm(block_list, W_01)
    m_add = get_coo_matrix(g_add, symmetric=True)


    # prompt: generate a sparse coo matrix with the pairs in m_add but not in M

    # Find pairs present in m_add but not in M
    m_add_pairs = set(zip(m_add.row, m_add.col))
    M_pairs = set(zip(adj_matrix.row, adj_matrix.col))

    pairs_in_m_add_not_in_M = list(m_add_pairs - M_pairs)

    # Create a sparse COO matrix from the identified pairs
    rows = [pair[0] for pair in pairs_in_m_add_not_in_M]
    cols = [pair[1] for pair in pairs_in_m_add_not_in_M]
    data = [1] * len(rows)  # Assuming a value of 1 for the new matrix

    M_add = sparse.coo_matrix((data, (rows, cols)), shape=adj_matrix.shape)

    Noisy_M = M_add + M_keep
    Noisy_M = Noisy_M.tocoo()
    return(Noisy_M)


def get_coo_matrix(G, symmetric=False):
    nnodes = G.num_vertices()

    rows = []
    cols = []
    data = []

    for e in G.edges():
        src = e.source()
        tgt = e.target()
        rows.append(src)
        cols.append(tgt)
        data.append(1)  # unweighted graph

        if symmetric:
        # For undirected graphs
            rows.append(tgt)
            cols.append(src)
            data.append(1)

    # Create the adjacency matrix in COO format
    adj_matrix_coo = coo_matrix((data, (rows, cols)), shape=(nnodes, nnodes))
    return adj_matrix_coo


def get_submatrix(adj_matrix_coo, block_list, block1, block2): ## For undirected

    nodes_in_block1 = [i for i, block in enumerate(block_list) if block == block1]
    nodes_in_block2 = [i for i, block in enumerate(block_list) if block == block2]

    # print('bloc', block1, 'elements:', len(nodes_in_block1), nodes_in_block1)
    # print('bloc', block2, 'elements:', len(nodes_in_block2), nodes_in_block2)

    new_rows = []
    new_cols = []
    new_data = []

    for i, j, v in zip(adj_matrix_coo.row, adj_matrix_coo.col, adj_matrix_coo.data):
        if i in nodes_in_block1 and j in nodes_in_block2:
            new_rows.append(nodes_in_block1.index(i))
            new_cols.append(nodes_in_block2.index(j))
            new_data.append(v)

    new_shape = (len(nodes_in_block1), len(nodes_in_block2))

    if new_rows:
        new_submatrix_coo = coo_matrix((new_data, (new_rows, new_cols)),
                                       shape=new_shape)
        return new_submatrix_coo
    else:
      return coo_matrix(new_shape)

def prob_from_eps_density(epsilon, density):
    density = max(density, 1e-10)  # or any small positive value
    denom = (np.exp(epsilon)- 1 + (1/density))
    z = 1/denom
    v = (1 - z, np.exp(epsilon) * z)  # vector v=(p_00, p_11) that preserves Sparseness and Epsilon-LDP
    # print(v)
    return v


def Noise_edges(nnodes_A, nnodes_B, M, bipartite=True):
    # np.random.seed(42)

    row  = np.random.choice(nnodes_A, size=M, replace=True)
    col  = np.random.choice(nnodes_B, size=M, replace=True)
    data = [1]*M

    NoiseM = sparse.coo_matrix((data, (row, col)), shape=(nnodes_A, nnodes_B))
    return(NoiseM)


def add_noise_eps(matrix_base, epsilon, density, bipartite=True):
    # np.random.seed(42) # should be OFF for more than 1 iteration

    nrows, ncols = matrix_base.shape
    E = matrix_base.data.shape[0]

    # Current solution for cluster-pair density=0  # Add one random edge
    # if E == 0:
    #     matrix_base = Noise_edges(nrows, ncols, 1)
    #     E = matrix_base.data.shape[0]
    density = max(density, 1e-10)  # or any small positive value

    p_00, p_11 = prob_from_eps_density(epsilon, density)

    N = nrows*ncols
    if bipartite == False:
        N = (nrows-1)*(ncols-1)

    NoiseM = Noise_edges(nrows, ncols, int((1-p_00)*(1-density)*N))

    noisy_matrix = matrix_base.copy()
    noisy_matrix.data = np.random.binomial(n=1, p=p_11, size=noisy_matrix.data.shape[0])
    added = noisy_matrix + NoiseM
    # matrix_base.data = np.random.binomial(n=1, p=p_11, size=int(density*N)) #np.min([1,p_11]), size=E) # min to avoid problems with p_11
    # added = matrix_base + NoiseM
    added.eliminate_zeros()
    return(added)


def block_LDP(adj_matrix, block_list, epsilon):
    block_labels, block_sizes = np.unique(block_list, return_counts=True)
    nblocks = len(block_labels)


    dict_blocks = {}
    for block_i in block_labels:
        for block_j in block_labels:
            submatrix = get_submatrix(adj_matrix, block_list, block_i, block_j)
            density = np.max([submatrix.getnnz(), 1])/(submatrix.shape[0] * submatrix.shape[1])

            bipartite=True
            if block_i == block_j:
                density = np.max([submatrix.getnnz(), 1])/((submatrix.shape[0] - 1)**2)
                bipartite=False

            # print('density', density, block_i, block_j)
            Noisy_block = add_noise_eps(matrix_base=submatrix, epsilon=epsilon, density=density, bipartite=bipartite)

            dict_blocks[(block_i, block_j)] = [submatrix, density, Noisy_block]

    noisy_adj = join_coo_matrices(dict_blocks, block_list, adj_matrix)
    return(noisy_adj)

def join_coo_matrices(dict_blocks, block_list, adj_matrix): # he afegit: , block_list, adj_matrix

  rows = []
  cols = []
  data = []

  for block_pair, block_data in dict_blocks.items():    # related to dict_blocks
      noisy_submatrix = block_data[2]

     # convert to coo_matrix
      if not isinstance(noisy_submatrix, coo_matrix):
          noisy_submatrix = noisy_submatrix.tocoo()


      # Iterate through non-zero entries of the noisy submatrix
      for row_idx, col_idx, value in zip(noisy_submatrix.row, noisy_submatrix.col, noisy_submatrix.data):
          # Convert local indices to global indices
          # Assuming the global node indices are contiguous within each block
          block_i, block_j = block_pair

          # Find the global indices using block_list and the local indices
           # Get the indices of nodes within each block
          nodes_in_block_i = np.where(block_list == block_i)[0]
          nodes_in_block_j = np.where(block_list == block_j)[0]

          global_row_idx = nodes_in_block_i[row_idx]
          global_col_idx = nodes_in_block_j[col_idx]

          # Append the global indices to the combined lists
          rows.append(global_row_idx)
          cols.append(global_col_idx)
          data.append(value)

  return sparse.coo_matrix((data, (rows, cols)), shape=adj_matrix.shape)


# prompt: read graphtool graph from coo matrix

def coo_matrix_to_graphtool(coo_matrix):
  g = Graph(directed=False)  # Create an undirected graph
  g.add_vertex(coo_matrix.shape[0])

  for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
      g.add_edge(g.vertex(i), g.vertex(j))

  remove_parallel_edges(g)
  remove_self_loops(g)
  return g

def nx_to_gt(graph_nx):
  graph_gt = Graph(directed=graph_nx.is_directed())
  node_map = list(graph_gt.add_vertex(len(graph_nx))) 


  for u, v in graph_nx.edges():
      graph_gt.add_edge(node_map[u], node_map[v])
      
  return graph_gt


def coo_matrix_to_networkx(coo_matrix):
    graph = nx.Graph()
    graph.add_nodes_from(range(coo_matrix.shape[0]))  # Add nodes
    
    for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        if v > 0:  # Add edges only for non-zero values
            graph.add_edge(i, j)
            
    return graph
#

def get_mat(data_path):
    # data_path = './data/' + dataset_name + '.txt'
    data = np.loadtxt(data_path)

    
    # initial statistics
    dat = (np.append(data[:,0],data[:,1])).astype(int)
    dat_c = np.bincount(dat)

    d = {}
    node = 0
    mid = []
    for i in range(len(dat_c)):
        if dat_c[i] > 0:
            d[i] = node
            mid.append(i)
            node = node + 1
    mid = np.array(mid,dtype=np.int32)

    # initial statistics
    Edge_num = data.shape[0] 
    c = len(d) 


    # genarated adjancent matrix
    mat0 = np.zeros([c,c],dtype=np.uint8)
    for i in range(Edge_num):
        mat0[d[int(data[i,0])],d[int(data[i,1])]] = 1


    # transfer direct to undirect
    mat0 = mat0 + np.transpose(mat0)
    mat0 = np.triu(mat0,1)
    mat0 = mat0 + np.transpose(mat0)
    mat0[mat0>0] = 1
    return mat0,mid

def community_init(mat0,mat0_graph,epsilon,nr,t=1.0):

    # t1 = time.time()
    # Divide the nodes randomly
    g1 = list(np.zeros(len(mat0)))
    ind = -1

    for i in range(len(mat0)):
        if i % nr == 0:
            ind = ind + 1
        g1[i] = ind

    random.shuffle(g1)

    mat0_par3 = {}
    for i in range(len(mat0)):
        mat0_par3[i] = g1[i]

    gr1 = max(mat0_par3.values()) + 1

    # mat0_mod3 = community.modularity(mat0_par3,mat0_graph)
    # print('mat0_mod2=%.3f,gr1=%d'%(mat0_mod3,gr1)) 

    
    mat0_par3_pv = np.array(list(mat0_par3.values()))
    mat0_par3_pvs = []
    for i in range(gr1):
        pv = np.where(mat0_par3_pv==i)[0]
        pvs = list(pv)
        mat0_par3_pvs.append(pvs)
    mat_one_level = np.zeros([gr1,gr1])

    for i in range(gr1):
        pi = mat0_par3_pvs[i]
        mat_one_level[i,i] = np.sum(mat0[np.ix_(pi,pi)])
        for j in range(i+1,gr1):
            pj = mat0_par3_pvs[j]
            mat_one_level[i,j] = np.sum(mat0[np.ix_(pi,pj)])
    # print('generate new matrix time:%.2fs'%(time.time()-t1))
    
    lap_noise = laplace(0,1/epsilon,gr1*gr1).astype(np.int32)
    lap_noise = lap_noise.reshape(gr1,gr1)

    ga = get_uptri_arr(mat_one_level,ind=1)
    ga_noise = ga + laplace(0,1/epsilon,len(ga))
    ga_noise_pp = FO_pp(ga_noise)
    mat_one_level_noise = get_upmat(ga_noise_pp,gr1,ind=1)


    noise_diag = np.int32(mat_one_level.diagonal() + laplace(0,2/epsilon,len(mat_one_level)))

    # keep consistency
    noise_diag = FO_pp(noise_diag)
  
    mat_one_level_noise = np.triu(mat_one_level_noise,1)
    mat_one_level_noise = mat_one_level_noise + np.transpose(mat_one_level_noise)

    row,col = np.diag_indices_from(mat_one_level_noise) 
    mat_one_level_noise[row,col] = noise_diag
    mat_one_level_noise[mat_one_level_noise<0] = 0

    mat_one_level_graph = nx.from_numpy_array(mat_one_level_noise,create_using=nx.Graph)
    
    # Apply the Louvain method
    mat_new_par = community_louvain.best_partition(mat_one_level_graph,resolution=t)
    gr2 = max(mat_new_par.values()) + 1 
    mat_new_pv = np.array(list(mat_new_par.values()))
    mat_final_pvs = []
    for i in range(gr2):
        pv = np.where(mat_new_pv==i)[0]
        mat_final_pv = []
        for j in range(len(pv)):
            pvj = pv[j]
            mat_final_pv.extend(mat0_par3_pvs[pvj])
        mat_final_pvs.append(mat_final_pv)

    label1 = np.zeros([len(mat0)],dtype=np.int32)
    for i in range(len(mat_final_pvs)):
        label1[mat_final_pvs[i]] = i

    return label1



def get_uptri_arr(mat_init,ind=0):
    a = len(mat_init)
    res = []
    for i in range(a):
        dat = mat_init[i][i+ind:]
        res.extend(dat)
    arr = np.array(res)
    return arr


def get_upmat(arr,k,ind=0):
    mat = np.zeros([k,k],dtype=np.int32)
    left = 0
    for i in range(k):
        delta = k - i - ind
        mat[i,i+ind:] = arr[left:left+delta]
        left = left + delta
        
    return mat

# Post processing
def FO_pp(data_noise,type='norm_sub'):
    if type == 'norm_sub':
        data = norm_sub_deal(data_noise)
        
    if type == 'norm_mul':
        data = norm_mul_deal(data_noise)
    
    return data

def norm_sub_deal(data):
    data = np.array(data,dtype=np.int32)
    data_min = np.min(data)
    data_sum = np.sum(data)
    delta_m = 0 - data_min
    
    if delta_m > 0:
        dm = 100000000
        data_seq = np.zeros([len(data)],dtype=np.int32)
        for i in range(0,delta_m):
            data_t = data - i
            data_t[data_t<0] = 0
            data_t_s = np.sum(data_t)
            dt = np.abs(data_t_s - data_sum)
            if dt < dm:
                dm = dt
                data_seq = data_t
                if dt == 0:
                    break
                
    else:
        data_seq = data
    return data_seq
        



# generate graph(intra edges) based on degree sequence
def generate_intra_edge(dd1,div=1):
    dd1 = np.array(dd1,dtype=np.int32)
    dd1[dd1<0] = 0
    dd1_len = len(dd1)
    dd1_p = dd1.reshape(dd1_len,1) * dd1.reshape(1,dd1_len)
    s1 = np.sum(dd1)

    dd1_res = np.zeros([dd1_len,dd1_len],dtype=np.int8)
    if s1 > 0:
        batch_num = int(dd1_len / div)
        begin_id = 0
        for i in range(div):
            if i == div-1:
                batch_n = dd1_len - begin_id
                dd1_r = np.random.randint(0,high=s1,size=(batch_n,dd1_len))
                res = dd1_p[begin_id:,:] - dd1_r
                res[res>0] = 1
                res[res<1] = 0
                dd1_res[begin_id:,:] = res
            else:
                dd1_r = np.random.randint(0,high=s1,size=(batch_num,dd1_len))
                res = dd1_p[begin_id:begin_id+batch_num,:] - dd1_r
                res[res>0] = 1
                res[res<1] = 0
                dd1_res[begin_id:begin_id+batch_num,:] = res
                begin_id = begin_id + batch_num
    
    # make sure the final adjacency matrix is symmetric
    dd1_out = np.triu(dd1_res,1)
    dd1_out = dd1_out + np.transpose(dd1_out)
    return dd1_out

# calculate the diameter
def cal_diam(mat):
    mat_graph = nx.from_numpy_array(mat,create_using=nx.Graph)
    max_diam = 0
    for com in nx.connected_components(mat_graph):
        com_list = list(com)
        mat_sub = mat[np.ix_(com_list,com_list)]
        sub_g = nx.from_numpy_array(mat_sub,create_using=nx.Graph)
        diam = nx.diameter(sub_g)
        if diam > max_diam:
            max_diam = diam
    return max_diam

# calculate the overlap 
def cal_overlap(la,lb,k):
    la = la[:k]
    lb = lb[:k]
    la_s = set(la)
    lb_s = set(lb)
    num = len(la_s & lb_s)
    rate = num / k
    return rate


# calculate the KL divergence
def cal_kl(A,B): 
    p = A / sum(A)
    q = B / sum(B)
    if A.shape[0] > B.shape[0]:
        q = np.pad(q,(0,p.shape[0]-q.shape[0]),'constant',constant_values=(0,0))
    elif A.shape[0] < B.shape[0]:
        p = np.pad(p,(0,q.shape[0]-p.shape[0]),'constant',constant_values=(0,0))
    kl = p * np.log((p+np.finfo(np.float64).eps)/(q+np.finfo(np.float64).eps))
    kl = np.sum(kl)
    return kl


# calculate the RE
def cal_rel(A,B): 
    eps = 0.000000000000001
    A = np.float64(A)
    B = np.float64(B)
    #eps = np.float64(eps)
    res = abs((A-B)/(A+eps))
    return res

# calculate the MSE
def cal_MSE(A,B): 
    res = np.mean((A-B)**2)
    return res

# calculate the MAE
def cal_MAE(A,B,k=None): 
    if k== None:
        res = np.mean(abs(A-B))
    else:
        a = np.array(A[:k])
        b = np.array(B[:k])
        res = np.mean(abs(a-b))
    return res


def write_edge_txt(mat0,mid,file_name):
    a0 = np.where(mat0==1)[0]
    a1 = np.where(mat0==1)[1]
    with open(file_name,'w+') as f:
        for i in range(len(a0)):
            f.write('%d\t%d\n'%(mid[a0[i]],mid[a1[i]]))


class PriorityQueue(object):
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])


def degreeDiscountIC(G, k, p=0.01):

    S = []
    dd = PriorityQueue() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                dd.add_task(v, -priority)
    return S

def runIC (G, S, p = 0.01):

    from copy import deepcopy
    from random import random
    T = deepcopy(S) # copy already selected nodes

    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print (T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T

def find_seed(graph_path,seed_size=20):
    
    # read in graph
    G = nx.Graph()
    with open(graph_path) as f:

        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
        
    
    S = degreeDiscountIC(G, seed_size)
    return S



def cal_spread(graph_path,S_all,p=0.01,seed_size=20,iterations=100):
    
    # read in graph
    G = nx.Graph()
    with open(graph_path) as f:

        for line in f:
            u, v = map(int, line.split())
            # print('u:%s,v:%s'%(u,v))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
           

    #calculate initial set
    
    if seed_size <= len(S_all):
        S = S_all[:seed_size]
    else:
        print('seed_size is too large.')
        S = S_all

    
    avg = 0
    for i in range(iterations):
        T = runIC(G, S, p)
        avg += float(len(T))/iterations

    avg_final = int(round(avg))

    return avg_final