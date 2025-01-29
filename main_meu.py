import community as community_louvain
import networkx as nx
import time
import numpy as np

from numpy.random import laplace
from sklearn import metrics

from utils import *

import os



def main_func(dataset_name='Chamelon',eps=[0.5,1,1.5,2,2.5,3,3.5],exp_num=10,save_csv=False):


    t_begin = time.time()

    data_path = './data/' + dataset_name + '.txt'
    mat0,mid = get_mat(data_path)
    

    cols = ['eps','exper','nmi','evc_overlap','evc_MAE','deg_kl', \
    'cc_rel','mod_rel']
    

    all_data = pd.DataFrame(None,columns=cols)

    # original graph
    mat0_graph = nx.from_numpy_array(mat0,create_using=nx.Graph)

    mat0_edge = mat0_graph.number_of_edges()
    mat0_node = mat0_graph.number_of_nodes()
    print('Dataset:%s'%(dataset_name))
    print('Node number:%d'%(mat0_graph.number_of_nodes()))
    print('Edge number:%d'%(mat0_graph.number_of_edges()))

    t_aux = time.time() #
    mat0_par = community_louvain.best_partition(mat0_graph)
    print('time_louvain',time.time()-t_aux) #

    mat0_degree = np.sum(mat0,0)
    mat0_deg_dist = np.bincount(np.int64(mat0_degree)) # degree distribution

    t_aux = time.time() #
    mat0_evc = nx.eigenvector_centrality(mat0_graph,max_iter=10000)
    mat0_evc_a = dict(sorted(mat0_evc.items(),key = lambda x:x[1],reverse=True))
    mat0_evc_ak = list(mat0_evc_a.keys())
    mat0_evc_val = np.array(list(mat0_evc_a.values()))
    print('time_EVC',time.time()-t_aux) #

    evc_kn = np.int64(0.01*mat0_node)

    # mat0_diam = cal_diam(mat0)
      
    t_aux = time.time() #
    mat0_cc = nx.transitivity(mat0_graph)
    print('time_ClustC',time.time()-t_aux) #

    mat0_mod = community_louvain.modularity(mat0_par,mat0_graph)


    all_deg_kl = []
    all_mod_rel = []
    all_nmi_arr = []
    all_evc_overlap = []
    all_evc_MAE = []
    all_cc_rel = []
    all_diam_rel = []

    ###
    t_aux = time.time() #
    mat0_par_list = list(mat0_par.values())
    block_list = mat0_par_list
    # G = nx_to_gt(mat0_graph)
    # adj_matrix = get_coo_matrix(G, symmetric=True)
    adj_matrix= nx.to_scipy_sparse_array(mat0_graph, format='coo')
    density_matrix = calc_all_densities(adj_matrix, block_list)
    print('time_density',time.time()-t_aux) #
    ###


    for ei in range(len(eps)):
        epsilon = eps[ei]
        ti = time.time()
        
     
        nmi_arr = np.zeros([exp_num])
        deg_kl_arr = np.zeros([exp_num])
        mod_rel_arr = np.zeros([exp_num])
        cc_rel_arr =  np.zeros([exp_num])
        diam_rel_arr = np.zeros([exp_num])
        evc_overlap_arr = np.zeros([exp_num])
        evc_MAE_arr = np.zeros([exp_num])


        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------'%(epsilon,exper+1,exp_num))


            t1 = time.time()
            
            ###

            ###

            MG = add_SBM(adj_matrix, block_list, density_matrix, epsilon)
            mat2 = MG.toarray()
            mat2_graph = coo_matrix_to_networkx(MG)
            print('time_SBM',time.time()-t1) #
            
            # save the graph
            # file_name = './result/' +  'PrivGraph_%s_%.1f_%d.txt' %(dataset_name,epsilon,exper)
            # write_edge_txt(mat2,mid,file_name)

            #evaluate
            mat2_edge = mat2_graph.number_of_edges()
            mat2_node = mat2_graph.number_of_nodes()

            mat2_par = community_louvain.best_partition(mat2_graph)
            mat2_mod = community_louvain.modularity(mat2_par,mat2_graph)

            mat2_cc = nx.transitivity(mat2_graph)

            mat2_degree = np.sum(mat2,0)
            mat2_deg_dist = np.bincount(np.int64(mat2_degree)) # degree distribution
            
            mat2_evc = nx.eigenvector_centrality(mat2_graph,max_iter=10000)
            mat2_evc_a = dict(sorted(mat2_evc.items(),key = lambda x:x[1],reverse=True))
            mat2_evc_ak = list(mat2_evc_a.keys())
            mat2_evc_val = np.array(list(mat2_evc_a.values()))
        

            # mat2_diam = cal_diam(mat2)   # Expensive and useless

            # calculate the metrics
            # clustering coefficent
            cc_rel = cal_rel(mat0_cc,mat2_cc)

            # degree distribution
            deg_kl = cal_kl(mat0_deg_dist,mat2_deg_dist)

            # modularity
            mod_rel = cal_rel(mat0_mod,mat2_mod)
            
        
            # NMI
            labels_true = list(mat0_par.values())
            labels_pred = list(mat2_par.values())
            nmi = metrics.normalized_mutual_info_score(labels_true,labels_pred)


            # Overlap of eigenvalue nodes 
            evc_overlap = cal_overlap(mat0_evc_ak,mat2_evc_ak,np.int64(0.01*mat0_node))

            # MAE of EVC
            evc_MAE = cal_MAE(mat0_evc_val,mat2_evc_val,k=evc_kn)

            # # diameter
            # diam_rel = cal_rel(mat0_diam,mat2_diam)


            nmi_arr[exper] = nmi
            cc_rel_arr[exper] = cc_rel
            deg_kl_arr[exper] = deg_kl
            mod_rel_arr[exper] = mod_rel
            evc_overlap_arr[exper] = evc_overlap
            evc_MAE_arr[exper] = evc_MAE
            # diam_rel_arr[exper] = diam_rel

            print('Nodes=%d,Edges=%d,nmi=%.4f,cc_rel=%.4f,deg_kl=%.4f,mod_rel=%.4f,evc_overlap=%.4f,evc_MAE=%.4f' \
                %(mat2_node,mat2_edge,nmi,cc_rel,deg_kl,mod_rel,evc_overlap,evc_MAE))

     

            data_col = [epsilon,exper,nmi,evc_overlap,evc_MAE,deg_kl, \
                cc_rel,mod_rel]
            col_len = len(data_col)
            data_col = np.array(data_col).reshape(1,col_len)
            data1 = pd.DataFrame(data_col,columns=cols)
            if all_data.empty:
                all_data = data1
            else:
                all_data = pd.concat([all_data, data1], ignore_index=True)

        all_nmi_arr.append(np.mean(nmi_arr))
        all_cc_rel.append(np.mean(cc_rel_arr))
        all_deg_kl.append(np.mean(deg_kl_arr))
        all_mod_rel.append(np.mean(mod_rel_arr))
        all_evc_overlap.append(np.mean(evc_overlap_arr))
        all_evc_MAE.append(np.mean(evc_MAE_arr))
        # all_diam_rel.append(np.mean(diam_rel_arr))

        
        print('all_index=%d/%d Done.%.2fs\n'%(ei+1,len(eps),time.time()-ti))

    res_path = './result'
    save_name = res_path + '/' + '%s_%d.csv' %(dataset_name,exp_num)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    if save_csv == True:
        all_data.to_csv(save_name,index=False,sep=',')

    print('-----------------------------')

    print('dataset:',dataset_name)
    
    print('eps=',eps)
    print('all_nmi_arr=',all_nmi_arr)
    print('all_evc_overlap=',all_evc_overlap)
    print('all_evc_MAE=',all_evc_MAE)
    print('all_deg_kl=',all_deg_kl)
    # print('all_diam_rel=',all_diam_rel)
    print('all_cc_rel=',all_cc_rel)
    print('all_mod_rel=',all_mod_rel)

    print('All time:%.2fs'%(time.time()-t_begin))



if __name__ == '__main__':
    # set the dataset
    # 'Enron'
    dataset_name = 'Chamelon' #'Enron' # 'CA-HepPh' # 'Facebook' # 'Chamelon' # 

    # set the privacy budget, list type
    eps = [0.5,1,1.5,2,2.5,3,3.5]

    exp_num=3

    # run the function
    main_func(dataset_name=dataset_name,eps=eps,exp_num=exp_num,save_csv=True)
   


