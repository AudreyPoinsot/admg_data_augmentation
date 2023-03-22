from .acyclic_graph_generator import AcyclicGraphGenerator
import pandas as pd
import numpy as np
import networkx as nx
import pickle


def introduce_observational_bias(data, bias_type='uniform', bias_params=None):
    """Change data distribution

    Args:
        data (DataFrame): The dataset to bias
        bias_type (str, optional): type of bias to use, 'uniform' or 'gaussian' . Defaults to 'uniform'.
        bias_params (dict, optional): parameters to intriduce bias. Defaults to None.

    Returns:
        DataFrame: Biased dataset
    """
    # check params
    if bias_params['biased_variable'] not in data.columns :
        print('Biased variable not in dataset ... no bias introduced')
        return data
    else:
        # check if it make sens to resample
        intervalle_has_no_effect = False
        if bias_params['bias_frac'] == 1:
            intervalle_has_no_effect = True
        if bias_type == 'uniform':
            if bias_params['intervalle_to_bias_start'] > bias_params['intervalle_to_bias_end']:
                intervalle_has_no_effect = True
            if bias_params['intervalle_to_bias_start'] > data[bias_params['biased_variable']].max():
                intervalle_has_no_effect = True
            if bias_params['intervalle_to_bias_end'] < data[bias_params['biased_variable']].min():
                intervalle_has_no_effect = True
        elif bias_type == 'gaussian':
            min_95_conf_intervalle = bias_params['mean'] - 1.96*bias_params['std']
            max_95_conf_intervalle = bias_params['mean'] + 1.96*bias_params['std']
            if min_95_conf_intervalle > data[bias_params['biased_variable']].max(): # if no points in the 95% confidence intervalle, no bias introduced
                intervalle_has_no_effect = True
            if max_95_conf_intervalle < data[bias_params['biased_variable']].min():
                intervalle_has_no_effect = True
        
        if intervalle_has_no_effect:
            return data
        else:
            # compute weight for resampling
            biased_data = data.copy()
            biased_data['weights'] = 1
            if bias_type == 'uniform':
                if bias_params['sampling_effect'] == 'oversample' :
                    unbiased_subdata = biased_data[(biased_data[bias_params['biased_variable']]>=bias_params['intervalle_to_bias_start'])&(biased_data[bias_params['biased_variable']]<=bias_params['intervalle_to_bias_end'])]
                    biased_subdata = biased_data[(biased_data[bias_params['biased_variable']]<bias_params['intervalle_to_bias_start'])|(biased_data[bias_params['biased_variable']]>bias_params['intervalle_to_bias_end'])]
                    biased_subdata['weights'] = 1/bias_params['bias_frac']
                    size = int(unbiased_subdata.shape[0] + 1/bias_params['bias_frac']*biased_subdata.shape[0])
                else:
                    if bias_params['sampling_effect'] != 'undersample' :
                        print('Sampling effect not recognized .... undersampling by default')
                    biased_subdata = biased_data[(biased_data[bias_params['biased_variable']]>=bias_params['intervalle_to_bias_start'])&(biased_data[bias_params['biased_variable']]<=bias_params['intervalle_to_bias_end'])]
                    unbiased_subdata = biased_data[(biased_data[bias_params['biased_variable']]<bias_params['intervalle_to_bias_start'])|(biased_data[bias_params['biased_variable']]>bias_params['intervalle_to_bias_end'])]
                    biased_subdata['weights'] = bias_params['bias_frac']
                    size = int(unbiased_subdata.shape[0] + bias_params['bias_frac']*biased_subdata.shape[0])
                biased_data = pd.concat((biased_subdata, unbiased_subdata))
            elif bias_type == 'gaussian':
                if bias_params['sampling_effect'] == 'oversample' :
                    biased_data['weights'] = 1/(2*3.14*bias_params['std']**2)*np.exp(-(biased_data[bias_params['biased_variable']] - bias_params['mean'])**2/(2*bias_params['std']**2))
                    num_unbiased_subdata = biased_data[(biased_data[bias_params['biased_variable']]>=min_95_conf_intervalle)&(biased_data[bias_params['biased_variable']]<=max_95_conf_intervalle)].shape[0]
                    num_biased_subdata = biased_data[(biased_data[bias_params['biased_variable']]<min_95_conf_intervalle)|(biased_data[bias_params['biased_variable']]>max_95_conf_intervalle)].shape[0]
                    size = int(num_unbiased_subdata + 1/bias_params['bias_frac']*num_biased_subdata)
                else:
                    if bias_params['sampling_effect'] != 'undersample' :
                        print('Sampling effect not recognized .... undersampling by default')
                    biased_data['weights'] = 1 - 1/(2*3.14*bias_params['std']**2)*np.exp(-(biased_data[bias_params['biased_variable']] - bias_params['mean'])**2/(2*bias_params['std']**2))
                    num_biased_subdata = biased_data[(biased_data[bias_params['biased_variable']]>=min_95_conf_intervalle)&(biased_data[bias_params['biased_variable']]<=max_95_conf_intervalle)].shape[0]
                    num_unbiased_subdata = biased_data[(biased_data[bias_params['biased_variable']]<min_95_conf_intervalle)|(biased_data[bias_params['biased_variable']]>max_95_conf_intervalle)].shape[0]
                    size = int(num_unbiased_subdata + bias_params['bias_frac']*num_biased_subdata)
            # resample according to the weights
            biased_data = biased_data.sample(n=size, weights='weights', random_state=1).drop(columns=["weights"])
            return biased_data

def add_uniform_outliers(data, frac_outlier=0.05):
    """Add outliers generated from a uniform distribution

    Args:
        data (DataFrame): dataset to which add outliers
        frac_outlier (float, optional): fraction of outliers to add. Defaults to 0.05.

    Returns:
        DataFrame: dataset with outliers
    """
    data_out = data.copy()
    outliers_data = []
    for node in data_out.columns:
        outliers_data.append(list(np.random.uniform(low=data_out[node].min(), high=data_out[node].max(), size=int(frac_outlier*data_out.shape[0]))))

    outliers_df = pd.DataFrame(np.transpose(np.array(outliers_data)), columns=data.columns)
    dataset_with_outliers = pd.concat((data_out, outliers_df), ignore_index=True)
    outliers_index_list = [i for i in range(data_out.shape[0], dataset_with_outliers.shape[0])]
    return dataset_with_outliers, outliers_index_list


def add_causal_outliers(data, outlier_causal_model_param, frac_outlier=0.05):
    """Add outliers generated from a random SCM

    Args:
        data (DataFrame): dataset to which add outliers
        outlier_causal_model_param (dict): parameters to use to generate the SCM from which sampling the outliers
        frac_outlier (float, optional): fraction of outliers to add. Defaults to 0.05.

    Returns:
        DataFrame: dataset with outliers
    """
    data_out = data.copy()
    # check params
    if 'causal_mechanism' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['causal_mechanism'] = 'polynomial'
    if 'adjacency_matrix' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['adjacency_matrix'] = None
    if 'noise' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['noise'] = 'gaussian'
    if 'noise_coeff' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['noise_coeff'] = 0.4
    if 'nodes' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['nodes'] = 5
    if 'parents_max' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['parents_max'] = 4
    if 'expected_degree' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['expected_degree'] = 3
    if 'dag_type' not in outlier_causal_model_param.keys():
        outlier_causal_model_param['dag_type'] = 'default'
    # build outlier graph
    generator_out = AcyclicGraphGenerator(
        causal_mechanism=outlier_causal_model_param['causal_mechanism'], 
        adjacency_matrix=outlier_causal_model_param['adjacency_matrix'], 
        noise=outlier_causal_model_param['noise'], 
        noise_coeff=outlier_causal_model_param['noise_coeff'], 
        npoints=int(data.shape[0]*frac_outlier), 
        nodes=outlier_causal_model_param['nodes'], 
        parents_max=outlier_causal_model_param['parents_max'], 
        expected_degree=outlier_causal_model_param['expected_degree'],
        dag_type=outlier_causal_model_param['dag_type']
        )
    # generate outliers
    outliers_df, graph_out = generator_out.generate()
    dataset_with_outliers = pd.concat((data_out, outliers_df), ignore_index=True)
    outliers_index_list = [i for i in range(data_out.shape[0], dataset_with_outliers.shape[0])]
    return dataset_with_outliers, outliers_index_list


def from_adj_mat_to_list_of_edges(graph):
    """Get list of edges

    Args:
        graph (nx.Graph): networkx graph

    Returns:
        list: list of edges
    """
    return list(graph.edges)


def save_graph(graph, path='graph.gpickle'):
    """Save graph as pickle

    Args:
        graph (nx.Graph): networkx graph
        path (str, optional): path to the pickle file to store the graph. Defaults to 'graph.gpickle'.
    """
    with open(path, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


def load_graph(path='graph.gpickle'):
    """Load graph from a pickle file

    Args:
        path (str, optional): path to the pickle file containing the graph. Defaults to 'graph.gpickle'.

    Returns:
        nx.Graph: networkx graph
    """
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def generate_data(causal_mechanism, adjacency_matrix=None, 
                  noise='gaussian', noise_coeff=.4, npoints=500, 
                  nodes=20, parents_max=5, expected_degree=3,
                  dag_type='default',
                  bias_type=None, # 'uniform' or 'gaussian'
                  bias_params=None,
                  outlier_type=None, # 'uniform' or 'random_graph'
                  frac_outlier=0.05,
                  outlier_params=None
                  ):
    """Generate data sampled from a random SCM

    Args:
        causal_mechanism (str): currently implemented mechanisms:
            ['linear', 'polynomial', 'sigmoid_add',
            'sigmoid_mix', 'gp_add', 'gp_mix', 'nn'].
        adjacency_matrix (array, optional): adjacency matrix of the SCM. Defaults to None.
        noise (str, optional): type of noise to use in the generative process
            ('gaussian', 'uniform' or a custom noise function). Defaults to 'gaussian'.
        noise_coeff (float, optional): Proportion of noise in the mechanisms. Defaults to .4.
        npoints (int, optional): dataset size. Defaults to 500.
        nodes (int, optional): number of variables, used if adjacency_matrix=None. Defaults to 20.
        parents_max (int, optional): maximum number of parents per node, used if adjacency_matrix=None. Defaults to 5.
        expected_degree (int, optional): expected degree of the graph, used if adjacency_matrix=None. Defaults to 3.
        dag_type (str, optional): type of graph to generate ('default', 'erdos'), used if adjacency_matrix=None. Defaults to 'default'.
        bias_type (str, optional): type of bias to use. Defaults to None.
        bias_params (dict, optional): parameters for the bias method to use. Defaults to None.
        outlier_type (str, optional): type of outliers to use. Defaults to None.
        outlier_params (dict, optional): parameters for the outliers method to use. Defaults to None.

    Returns:
        DataFrame: generated dataset
    """
    
    # define the causal model generator
    generator = AcyclicGraphGenerator(
        causal_mechanism=causal_mechanism, 
        adjacency_matrix= adjacency_matrix,
        noise=noise, 
        noise_coeff=noise_coeff,
        npoints=npoints, 
        nodes=nodes, 
        parents_max=parents_max, 
        expected_degree=expected_degree,
        dag_type=dag_type
        )
    
    # generate data and graph from the previously defined generator
    data, graph = generator.generate()
    save_graph(graph)
    
    # add biases
    if bias_type in ['uniform', 'gaussian']:
        data = introduce_observational_bias(data, bias_type=bias_type, bias_params=bias_params)
    
    # add outliers
    if outlier_type == 'uniform':
        data, _ = add_uniform_outliers(data, frac_outlier=frac_outlier)
    elif outlier_type == 'causal':
        data, _ = add_causal_outliers(data, outlier_causal_model_param=outlier_params, frac_outlier=frac_outlier)
    
    return data

