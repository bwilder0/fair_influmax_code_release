import numpy as np
import random
from numba import jit
import networkx as nx


def sample_live_icm(g, num_graphs):
    '''
    Returns num_graphs live edge graphs sampled from the ICM on g. Assumes that
    each edge has a propagation probability accessible via g[u][v]['p'].
    '''
    import networkx as nx
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u,v in g.edges():
            if random.random() < g[u][v]['p']:
                h.add_edge(u,v)
        live_edge_graphs.append(h)
    return live_edge_graphs

def f_all_influmax_multlinear(x, Gs, Ps, ws):
    '''
    Objective function for the multilinear extension of a live-edge
    influence maximization problem.
    
    x: continuous decision variables
    
    Gs/Ps/ws: representation of the influence maximization problem as an 
    expectation over a sampled set of probabilistic coverage functions (see below)
    '''
    n = len(Gs)
    sample_weights = 1./n * np.ones(n)
    return objective_live_edge(x, Gs, Ps, ws, sample_weights)

def make_multilinear_objective_samples(live_graphs, target_nodes, selectable_nodes, p_attend):
    '''
    Given a set of sampled live edge graphs, returns an function evaluating the 
    multilinear extension for the corresponding influence maximization problem.
    
    live_graphs: list of networkx graphs containing sampled live edges
    
    target_nodes: nodes that should be counted towards the objective
    
    selectable_nodes: nodes that are eligible to be chosen as seeds
    
    p_attend: probability that each node will be influenced if it is chosen as
    a seed.
    '''
    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)
    def f_all(x):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        return f_all_influmax_multlinear(x_expand, Gs, Ps, ws)
    return f_all


def make_multilinear_objective_samples_group(live_graphs, group_indicator, target_nodes, selectable_nodes, p_attend):
    '''
    Returns a function with stochastically evaluates the value of a point x with 
    respect to the multilinear extensions of all of the group objectives, where
    the indicator for which node belongs to which group is given by 
    group_indicator. 
    '''
    Gs, Ps, start_poses, ws = live_edge_to_edgelist(live_graphs, target_nodes, p_attend)
    def f_all(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        pr_reached = np.zeros(len(x_expand))
        for b in range(batch_size):
            i = random.randint(0, len(Gs)-1)
            pr_reached += (1./batch_size)*marginal_coverage_edgelist(x_expand, Gs[i], Ps[i], start_poses[i], ws[i])
        return pr_reached @ group_indicator
    return f_all


def make_multilinear_gradient_samples(live_graphs, target_nodes, selectable_nodes, p_attend):
    '''
    Given a set of sampled live edge graphs, returns an stochastic gradient 
    oracle for the multilinear extension of the corresponding influence 
    maximization problem. 
    
    live_graphs: list of networkx graphs containing sampled live edges
    
    target_nodes: nodes that should be counted towards the objective
    
    selectable_nodes: nodes that are eligible to be chosen as seeds
    
    p_attend: probability that each node will be influenced if it is chosen as
    a seed.
    '''
    import random
    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)    
    def gradient(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        samples = random.sample(range(len(Gs)), batch_size)
        grad = gradient_live_edge(x_expand, [Gs[i] for i in samples], [Ps[i] for i in samples], [ws[i] for i in samples], 1./batch_size * np.ones(len(Gs)))
        return grad[selectable_nodes]
    return gradient


def make_multilinear_gradient_group(live_graphs, group_indicator, target_nodes, selectable_nodes, p_attend):
    '''
    Given a set of sampled live edge graphs, returns an stochastic gradient 
    oracle for the multilinear extension of the corresponding influence 
    maximization problem, aggregated at the level of groups. The oracle returns a n x m
    matrix, where n is the number of nodes, m is the number of groups, and entry 
    ij is (in expectation), d(objective of group j)/d(xi).
    
    live_graphs: list of networkx graphs containing sampled live edges
    
    target_nodes: nodes that should be counted towards the objective
    
    selectable_nodes: nodes that are eligible to be chosen as seeds
    
    p_attend: probability that each node will be influenced if it is chosen as
    a seed.
    '''
    Gs, Ps, start_poses, ws = live_edge_to_edgelist(live_graphs, target_nodes, p_attend)    
    def gradient(x, batch_size):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        grad = gradient_estimate_all_nodes(x, Gs, Ps, start_poses, ws, batch_size) @ group_indicator
        return grad[selectable_nodes, :]
    return gradient


def live_edge_to_adjlist(live_edge_graphs, target_nodes, p_attend):
    '''
    Takes a list of live edge graphs and converts them to the format used by the functions below. 
    For each live edge graph g, the corresponding entry of Gs is the adjacency list of a bipartite graph,
    with each row representing a connected component of g and the entries of that row giving the nodes in 
    that connected component. Each row is terminated with -1s. 
    
    Each entry of Ps is an array of 1s of the same size as the corresponding entry of Gs. 
    
    Each entry of ws is an array, with each entry giving the size of the corresponding connected
    component.
    '''
    import networkx as nx
    Gs = []
    Ps = []
    ws = []
    target_nodes = set(target_nodes)
    for g in live_edge_graphs:
        cc = list(nx.connected_components(g))
        n = len(cc)
        max_degree = max([len(c) for c in cc])
        G_array = np.zeros((n, max_degree), dtype=np.int)
        P = np.zeros((n, max_degree))
        G_array[:] = -1
        for i in range(n):
            for j, v in enumerate(cc[i]):
                G_array[i, j] = v
                P[i, j] = p_attend[v]
        Gs.append(G_array)
        Ps.append(P)
        w = np.zeros((n))
        for i in range(n):
            w[i] = len(target_nodes.intersection(cc[i]))
        ws.append(w)
    return Gs, Ps, ws

def live_edge_to_edgelist(live_edge_graphs, target_nodes, p_attend):
    '''
    Converts a list of live edge graphs into a list of edge lists, along
    with auxiliary information. Returns:
        
    Gs: a list where each entry is a num_edges*2 numpy array. The first column
    is the starting point of each (directed) edge, and the second is its propagation
    probability, given here by p_attend[starting node]. 
    
    start_poses: a list where each entry allows easy lookup of the incoming edges
    to a given node. The indices [start_pos[v], start_pos[v+1]) give the rows of
    G corresponding to the incoming edges for node v
    
    ws: weight of each node towards the objective in each live graph. Currently, 
    these are all an indicator for target_nodes. 
    '''
    Gs = []
    start_poses = []
    ws = []
    Ps = []
    for h in live_edge_graphs:
        g = nx.DiGraph(h)
        G = np.zeros(g.number_of_edges(), dtype=np.int)
        P = np.zeros(g.number_of_edges())
        w = np.zeros(len(g))
        w[list(target_nodes)] = 1
        start_pos = np.zeros(len(g)+1, dtype=np.int)
        curr_pos = 0
        for v in g:
            start_pos[v] = curr_pos
            for u in g.predecessors(v):
                G[curr_pos] = u
                P[curr_pos] = p_attend[u]
                curr_pos += 1
        start_pos[-1] = g.number_of_edges()
        Gs.append(G)
        Ps.append(P)
        start_poses.append(start_pos)
        ws.append(w)
    return Gs, Ps, start_poses, ws
                
            

@jit
def gradient_live_edge(x, Gs, Ps, ws, weights):
    '''
    Gradient wrt x of the live edge influence maximization model. 
    
    x: current probability of seeding each node
    
    Gs/Ps/ws represent the input graphs, as defined in live_edge_to_adjlist
    '''
    grad = np.zeros((len(x)))
    for i in range(len(Gs)):
        grad += weights[i]*gradient_coverage(x, Gs[i], Ps[i], ws[i])
    grad /= len(x)
    return grad  

@jit
def objective_live_edge(x, Gs, Ps, ws, weights):
    '''
    Objective in the live edge influence maximization model, where nodes are
    seeded with probability in the corresponding entry of x. 
    
    Gs/Ps/ws represent the input graphs, as defined in live_edge_to_adjlist
    
    weights: probability of each graph occurring
    '''
    total = 0
    for i in range(len(Gs)):
        total += weights[i] * objective_coverage(x, Gs[i], Ps[i], ws[i])
    return total        

'''
The following functions compute gradients/objective values for the multilinear relaxation
of a (probabilistic) coverage function. The function is represented by the arrays G and P. 

Each row of G is a set to be covered, with the entries of the row giving the items that will
cover it (terminated with -1s). The corresponding entry of P gives the probability that
the item will cover that set (independently of all others). 

Corresponding to each row of G is an entry in the vector w, which gives the contribution
to the objective from covering that set.
'''

@jit
def gradient_coverage(x, G, P, w):
    '''
    Calculates gradient of the objective at fractional point x.
    
    x: fractional point as a vector. Should be reshapable into a matrix giving 
    probability of choosing copy i of node u.
    
    G: graph (adjacency list)
    
    P: probability on each edge. 
        
    w: weights for nodes in R
    '''
    grad = np.zeros((x.shape[0]))
    #process gradient entries one node at a time
    for v in range(G.shape[0]):
        p_all_fail = 1
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            p_all_fail *= 1 - x[G[v, j]]*P[v, j]
        for j in range(G.shape[1]):
            u = G[v, j]
            if u == -1:
                break
            #0/0 should be 0 here
            if p_all_fail == 0:
                p_others_fail = 0
            else:
                p_others_fail = p_all_fail/(1 - x[u]*P[v, j])
            grad[u] += w[v]*P[v, j]*p_others_fail
    return grad

@jit
def gradient_coverage_single(x, G, P, w, v):
    '''
    Calculates gradient of the objective function for a single node v (i.e., 
    the gradient of Pr[v is reached] * w[v])
    
    x: fractional point as a vector. Should be reshapable into a matrix giving 
    probability of choosing copy i of node u.
    
    G: graph (adjacency list)
    
    P: probability on each edge. 
        
    w: weights for nodes in R
    
    v: node to compute gradients for
    '''
    grad = np.zeros((x.shape[0]))
    #process gradient entries one node at a time
    p_all_fail = 1
    for j in range(G.shape[1]):
        if G[v, j] == -1:
            break
        p_all_fail *= 1 - x[G[v, j]]*P[v, j]
    for j in range(G.shape[1]):
        u = G[v, j]
        if u == -1:
            break
        #0/0 should be 0 here
        if p_all_fail == 0:
            p_others_fail = 0
        else:
            p_others_fail = p_all_fail/(1 - x[u]*P[v, j])
        grad[u] += w[v]*P[v, j]*p_others_fail
    return grad

@jit
def gradient_coverage_single_edgelist(x, G, P, start_pos, w, v):
    '''
    Calculates gradient of the objective function for a single node v (i.e., 
    the gradient of Pr[v is reached] * w[v])
    
    x: fractional point as a vector. Should be reshapable into a matrix giving 
    probability of choosing copy i of node u.
    
    G: edge list for the graph. Column 0: start point of the edge, Column 1: 
    propagation probability
    
    start_pos: start_pos[v] gives the row in G where the edges corresponding to 
    node v begin. start_pos[n] = # edges. 
        
    w: weights for nodes in R
    
    v: node to compute gradients for
    '''
    grad = np.zeros((x.shape[0]))
    #process gradient entries one node at a time
    p_all_fail = 1
    for j in range(start_pos[v], start_pos[v+1]):
        p_all_fail *= 1 - x[G[j]]*P[j]
    p_all_fail *= 1 - x[v]
    for j in range(start_pos[v], start_pos[v+1]):
        #0/0 should be 0 here
        if p_all_fail == 0:
            p_others_fail = 0
        else:
            p_others_fail = p_all_fail/(1 - x[G[j]]*P[j])
        grad[G[j]] += w[v]*P[j]*p_others_fail
    if p_all_fail > 0:
        grad[v] += w[v]*p_all_fail/(1 - x[v])
    if(np.any(np.isnan(grad))):
        print(x.min(), x.max())
        raise Exception()
    return grad

@jit
def gradient_estimate_all_nodes(x, Gs, Ps, start_poses, ws, B):
    '''
    Returns a stochastic estimate of the gradient of the probability that 
    every node is influenced wrt every selectable node. 
    '''
    n = start_poses[0].shape[0] - 1
    grad_est = np.zeros((x.shape[0], n))
    for b in range(B):
        v = random.randint(0, n-1)
        idx = random.randint(0, len(Gs)-1)
        grad_est[:, v] += (n/float(B)) * gradient_coverage_single_edgelist(x, Gs[idx], Ps[idx], start_poses[idx], ws[idx], v)
    return grad_est


@jit
def marginal_coverage(x, G, P, w):
    '''
    Returns marginal probability that each RHS vertex is reached.
    '''
    probs = np.ones((G.shape[0]))
    for v in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[v, j] == -1:
                break
            u = G[v, j]
            probs[v] *= 1 - x[u]*P[v, j]
    probs = 1 - probs
    return probs

@jit
def marginal_coverage_edgelist(x, G, P, start_pos, w):
    '''
    Returns marginal probability that each RHS vertex is reached.
    '''
    probs = np.ones(x.shape[0])
    for v in range(x.shape[0]):
        for j in range(start_pos[v], start_pos[v+1]):
            probs[v] *= 1 - x[G[j]]*P[j]
    probs *= 1 - x
    probs = 1 - probs
    return probs


@jit
def objective_coverage(x, G, P, w):
    '''
    Weighted objective value: the expected weight of the RHS nodes that are reached.
    '''
    return np.dot(w, marginal_coverage(x, G, P, w))
