import numpy as np
from utils import greedy

def indicator(S, n):
    x = np.zeros(n)
    x[list(S)] = 1
    return x

def multi_to_set(f, n):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def make_weighted(f, weights, *args):
    def weighted(x):
        return np.dot(weights, f(x, *args))
    return weighted

def make_normalized(f, targets):
    def normalized(x, *args):
        return f(x, *args)@np.diag(1./targets)
    return normalized

def make_contracted_function(f, S):
    '''
    For any function defined on the multilinear extension which takes x as its
    first argument, return a function which conditions on all items in S being
    chosen. 
    '''
    def contracted(x, *args):
        x = x.copy()
        x[list(S)] = 1
        return f(x, *args)
    return contracted


def mirror_sp(x, grad_oracle, k, group_indicator, group_targets, num_iter, step_size = 1, batch_size = 200, verbose=False):
    '''
    Uses stochastic saddle point mirror descent to solve the inner maxmin linear
    optimization problem.
    '''
    #random initialization
    startstep = step_size
    m = group_indicator.shape[1]
    v = x + 0.01*np.random.rand(*x.shape)
    v[v > 1] = 1
    v = k*v/v.sum()
    y = (1./m) * np.ones(m)
    group_weights = 1./group_targets
    group_weights[group_targets <= 0 ] = 0
    #historical mixed strategy for defender
    for t in range(num_iter):
        step_size = startstep/np.sqrt(t+1)
        #get a stochastic estimate of the gradient for each player
        g = grad_oracle(x, batch_size)
        group_grad = g @ np.diag(group_weights)
        grad_v = group_grad@y
        grad_y = v@group_grad
        #gradient step
        v = v * np.exp(step_size*grad_v)
        y = y*np.exp(-step_size*grad_y)
        #bregman projection
        v[v > 1] = 1
        v = k*v/v.sum()
        y = y/y.sum()
    return v

def lp_minmax(x, grad_oracle, k, group_indicator, group_targets):
    import gurobipy as gp
    m = gp.Model()
    m.setParam( 'OutputFlag', False )
    v = m.addVars(range(len(x)), lb = 0, ub = 1)
    obj = m.addVar()
    m.update()
    m.addConstr(gp.quicksum(v) <= k)
    g = grad_oracle(x, 5000)
    for i in range(len(group_targets)):
        m.addConstr(obj <= gp.quicksum(v[j]*g[j, i] for j in range(len(v)))/group_targets[i])
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return np.array([v[i].x for i in range(len(v))])


def rounding(x):
    '''
    Rounding algorithm that does not require decomposition of x into bases
    of the matroid polytope
    '''
    import random
    i = 0
    j = 1
    x = x.copy()
    for t in range(len(x)-1):
        if x[i] == 0 and x[j] == 0:
            i = i + 1
        if x[i] + x[j] < 1:
            if random.random() < x[i]/(x[i] + x[j]):
                x[i] = x[i] + x[j]
                x[j] = 0
                j = max((i,j)) + 1
            else:
                x[j] = x[i] + x[j]
                x[i] = 0
                i = max((i,j)) + 1
        else:
            if random.random() < (1 - x[j])/(2 - x[i] - x[j]):
                x[j] = x[i] + x[j] - 1
                x[i] = 1
                i = max((i,j)) + 1

            else:
                x[i] = x[i] + x[j] - 1
                x[j] = 1
                j = max((i,j)) + 1
    return x

def multiobjective_fw(grad_oracle, val_oracle, k, group_indicator, group_targets, num_iter, solver = 'md'):
    '''
    Uses FW updates to find a fractional point meeting a threshold value for each 
    of the objectives
    '''
#    startstep = stepsize
    x = np.zeros((num_iter, group_indicator.shape[0]))
    for t in range(num_iter-1):
        #how far each group currently is from the target
        iter_targets = group_targets - val_oracle((1/num_iter)*x[0:t].sum(axis=0), 5000) 
        if np.all(iter_targets < 0):
#            print('all targets met')
            iter_targets = np.ones(len(group_targets))
#        iter_targets = group_targets - val_oracle(x[t], 1000) 
#        print(iter_targets)
#        iter_targets[iter_targets < 0] = 0
#        print(iter_targets.min(), iter_targets.max())
        #Frank-Wolfe update
#        stepsize = startstep/np.sqrt(2*t+1)
        if solver == 'md':
            x[t+1] = mirror_sp((1./(t+1))*x[0:(t+1)].sum(axis=0), grad_oracle, k, group_indicator, iter_targets, num_iter=1000)
        elif solver == 'gurobi':
            x[t+1] = lp_minmax((1./(t+1))*x[0:(t+1)].sum(axis=0), grad_oracle, k, group_indicator, iter_targets)
        else:
            raise Exception('solver must be either md or gurobi')
    return x


def greedy_top_k(grad, elements, budget):
    '''
    Greedily select budget number of elements with highest weight according to
    grad
    ''' 
    import numpy as np
    inx = np.argpartition(grad, -budget)[-budget:]
    indicator = np.zeros(len(grad))
    indicator[inx[:budget]] = 1
    return indicator


def fw(grad_oracle, val_oracle, threshold, k, group_indicator, group_targets, num_iter, stepsize= 0.3):
    '''
    Run the normal frank-wolf algorithm to maximize a single submodular function
    '''
    x = np.zeros((num_iter, group_indicator.shape[0]))
    for t in range(num_iter-1):
        grad = grad_oracle((1./num_iter)*x[0:t].sum(axis=0), 1000).sum(axis=1)
        x[t+1] = greedy_top_k(grad, None, k)

    return x


def threshold_include(n_items, val_oracle, threshold):
    '''
    Makes a single pass through the items and returns a set including every
    item whose singleton value is at least threshold for some objective
    '''
    to_include = []
    x = np.zeros((n_items))
    for i in range(n_items):
        x[i] = 1
        vals = val_oracle(x, 1000)
        if vals.max() >= threshold:
            to_include.append(i)
        x[i] = 0
    return to_include


def algo(grad_oracle, val_oracle, threshold, k, group_indicator, group_targets, num_iter, solver):
    '''
    Combine algorithm that runs the first thresholding stage and then FW
    '''
    S = threshold_include(group_indicator.shape[0], val_oracle, threshold)
    grad_oracle = make_contracted_function(grad_oracle, S)
    val_oracle = make_contracted_function(val_oracle, S)
    x = multiobjective_fw(grad_oracle, val_oracle, k - len(S), group_indicator, group_targets, num_iter, solver)
    x[:, list(S)] = 1
    return x

def maxmin_algo(grad_oracle, val_oracle, threshold, k, group_indicator, num_iter, num_bin_iter, eps, solver):
    S = threshold_include(group_indicator.shape[0], val_oracle, threshold)
    grad_oracle = make_contracted_function(grad_oracle, S)
    val_oracle = make_contracted_function(val_oracle, S)
    ub = 1.
    lb = 0
    iternum = 0 
    while ub - lb > eps and iternum < num_bin_iter:
        target = (ub + lb)/2
#        print(target)
        group_targets = np.zeros(group_indicator.shape[1])
        group_targets[:] = target
        x = algo(grad_oracle, val_oracle, threshold, k, group_indicator, group_targets, num_iter, solver)
        vals = val_oracle(x.mean(axis=0), 1000)
        if vals.min() > target:
            lb = target
        else:
            ub = target
        iternum += 1
    return x

