import numpy as np 
from random import sample
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
# from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import gurobipy as gp 

'''get odds ratio bounds on estimated propensities (est_Q) given sensitivity level Gamma
'''
def get_bnds(est_Q,Gamma): 
    n = len(est_Q)
    p_hi = np.multiply(np.exp(Gamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(Gamma), est_Q ))
    p_lo = np.multiply(np.exp(-Gamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(-Gamma), est_Q ))
    assert (p_lo < p_hi).all()
    a_bnd = 1/p_hi; 
    b_bnd = 1/p_lo
    return [ a_bnd, b_bnd ]


''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind, 
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_short(Y, a_, b_, sub_ind=[]): 
    if len(sub_ind)>0: 
        print sub_ind
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    sort_inds = np.argsort(Y); a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds] 
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = sum(np.multiply(b_, Y)) /sum(b_) 
    while (val > prev_val) and (k < n_plus+1): 
        denom = 0; num = 0; prev_val = val; num = 1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:]))
        denom = sum(a_[0:k])+sum(b_[k:]); val = num / denom; k+=1; 
    lda_opt = prev_val; k_star = k-1
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]    
    return [lda_opt, weights, sum(weights)]

''' explicit: include plots of all values 
'''
def find_opt_weights_plot(Y,a_,b_,sub_ind=[]):
    if len(sub_ind)>0: 
        print sub_ind
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    sort_inds = np.argsort(Y); a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds] 
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = sum(np.multiply(b_, Y)) /sum(b_) 
    vals = [ (1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:])))/(sum(a_[0:k])+sum(b_[k:])) for k in range(n_plus) ]
    lda_opt = np.max(vals); k_star = np.argmax(vals)-1
    plt.figure()
    plt.plot(vals)
    plt.figure()
    plt.plot(np.diff(vals))
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]    
    return [lda_opt, weights, sum(weights)]

def rnd_k_val(k_,a_,b_,Y): 
    k= int(math.floor(k_)) # floor or round? 
    return (1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:])))/(sum(a_[0:k])+sum(b_[k:]))

''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind, 
use ternary search to find the optimal value. 
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_shorter(Y, a_, b_, sub_ind=[]): 
    if len(sub_ind)>0: 
        print sub_ind
        a_=a_[sub_ind]; b_=b_[sub_ind]; Y = Y[sub_ind]
    sort_inds = np.argsort(Y); a_=a_[sort_inds]; Y = Y[sort_inds]; b_=b_[sort_inds] 
    n_plus = len(Y); weights = np.zeros(n_plus); prev_val = -np.inf; k=1; val = sum(np.multiply(b_, Y)) /sum(b_) 
    left = 0; right = n_plus-1;keepgoing=True
    while keepgoing:
        #left and right are the current bounds; the maximum is between them
#         print (left,right)
#         print (rnd_k_val(leftThird,a_,b_,Y), rnd_k_val(rightThird,a_,b_,Y))
        if abs(right - left) < 2.1:
            k = np.floor((left + right)/2)
            keepgoing=False
        leftThird = left + (right - left)/3
        rightThird = right - (right - left)/3
        if rnd_k_val(leftThird,a_,b_,Y) < rnd_k_val(rightThird,a_,b_,Y):
            left = leftThird
        else:
            right = rightThird
    k_star=int(k); k=int(k)
    lda_opt = (1.0*sum(np.multiply(a_[0:k], Y[0:k])) + sum(np.multiply(b_[k:], Y[k:])))/(sum(a_[0:k])+sum(b_[k:]))
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_[0:k_star]; weights[sort_inds_b] = b_[k_star:]    
    return [lda_opt, weights, sum(weights)]

# Glue code from alternative API specification
'''Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind, 
Lambda (min problem value), weights, sum(weights)
'''
def find_opt_robust_ipw_val(Y, a_, b_,shorter=False):
    if shorter: 
        [lda_opt, weights, s_wghts] = find_opt_weights_shorter(Y, a_, b_)
    else: 
        [lda_opt, weights, s_wghts] = find_opt_weights_short(Y, a_, b_)
    return lda_opt

'''Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind, 
Lambda (max problem value), weights, sum(weights)
'''
def find_opt_robust_ipw_val_max(Y, a_,b_,shorter=False):
    if shorter: 
        [lda_opt, weights, s_wghts] = find_opt_weights_shorter(-Y, a_, b_)
    else: 
        [lda_opt, weights, s_wghts] = find_opt_weights_short(-Y, a_, b_)
    return -lda_opt

def find_opt_weights_short_val_mult(a_,b_,Y,p):
    return find_opt_weights_shorter(Y, a_*p, b_*p)



'''
Helper functions for estimating propensities 
'''
def estimate_prop(x, T, predict_x, predict_T): 
    clf_dropped = LogisticRegression()
    clf_dropped.fit(x, T)
    est_prop = clf_dropped.predict_proba(predict_x)
    est_Q = np.asarray( [est_prop[k,1] if predict_T[k] == 1 else est_prop[k,0] for k in range(len(predict_T))] )
    return [est_Q, clf] 
def get_prop(clf, x, T):
    est_prop = clf_dropped.predict_proba(x)
    est_Q = np.asarray( [est_prop[k,1] if T[k] == 1 else est_prop[k,0] for k in range(len(T))] )
    return est_Q


# get indicator vector from signed vector 
def get_0_1_sgn(vec): 
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else 0 for i in range(n) ]).flatten()
# get signed vector from indicator vector
def get_sgn_0_1(vec): 
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else -1 for i in range(n) ]).flatten()
'''
performs policy match indicator function; returns 1 or 0
input: signed treatments, signed policy assignment
'''
def pol_match(T_sgned, pol): 
    sgn_match = np.multiply(T_sgned, pol )
    return get_0_1_sgn(sgn_match)

''' return Pr[ \pi(x)=T ]
'''
def logistic_pol_match_obs(T_sgned, theta, x): 
    n = len(T_sgned); pol_match = np.multiply(T_sgned, np.dot(x, theta).flatten())
    LOGIT_TERM_POS = np.ones(n) / ( np.ones(n) + np.exp( -pol_match ))
    return LOGIT_TERM_POS
''' return Pr[ \pi(x)=1 ]
'''
def logistic_pol_asgn(theta, x): 
    theta = theta.flatten()
    n = x.shape[0]
    if len(theta) == 1: 
        logit = np.multiply(x, theta).flatten()
    else: 
        logit = np.dot(x, theta).flatten()
    LOGIT_TERM_POS = np.ones(n) / ( np.ones(n) + np.exp( -logit ))
    return LOGIT_TERM_POS


''' minimize wrt TV bound
'''
def get_general_interval_wghts_algo_uncentered_smoothed_f_divergence_TV(incumbent_pol, quiet=True, **params): 
    T_obs = params['T'].astype(int) ; Y = params['Y'].flatten(); n = params['n']; x = params['x']; 
    n = params['n']; a = params['a']; b = params['b']
    gamma = params['gamma']; wm = 1/params['q']
    if params['subind'] == True: 
        subinds = params['subinds']
        Y = Y[subinds]; n = len(subinds); T_obs = T_obs[subinds]; x = x[subinds]; a= a[subinds]; b= b[subinds]; wm = wm[subinds]
    # assume estimated propensities are probs of observing T_i
    y = Y; weights = np.zeros(n)
     # nominal propensities
    # smoothing probabilities
    T_sgned=get_sgn_0_1(T_obs)
    probs_pi_T = params['pi_pol'](T_sgned, incumbent_pol, x)
    a_mod = np.multiply(a, probs_pi_T); b_mod = np.multiply(b, probs_pi_T)
    wm = np.multiply(wm, probs_pi_T)
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_mod[i] * t)
        m.addConstr(w[i] >= a_mod[i] * t)
        m.addConstr(d[i] >=   w[i] - t*wm[i])
        m.addConstr(d[i] >= - w[i] + t*wm[i])
    m.optimize()
    return -m.ObjVal

''' maximize wrt TV bound
'''
def get_general_interval_wghts_algo_uncentered_smoothed_f_divergence_TV_max(incumbent_pol, quiet=True, **params): 
    T_obs = params['T'].astype(int) ; Y = params['Y'].flatten(); n = params['n']; x = params['x']; 
    n = params['n']; weights = np.zeros(n); a = params['a']; b = params['b']
    gamma = params['gamma']
    # assume estimated propensities are probs of observing T_i
    y = -Y
    wm = 1/params['q'] # nominal propensities
    # smoothing probabilities
    T_sgned=get_sgn_0_1(T_obs)
    probs_pi_T = params['pi_pol'](T_sgned, incumbent_pol, x)
    a_mod = np.multiply(a, probs_pi_T); b_mod = np.multiply(b, probs_pi_T)
    wm = np.multiply(wm, probs_pi_T)
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_mod[i] * t)
        m.addConstr(w[i] >= a_mod[i] * t)
        m.addConstr(d[i] >=   w[i] - t*wm[i])
        m.addConstr(d[i] >= - w[i] + t*wm[i])
    m.optimize()
    return m.ObjVal

