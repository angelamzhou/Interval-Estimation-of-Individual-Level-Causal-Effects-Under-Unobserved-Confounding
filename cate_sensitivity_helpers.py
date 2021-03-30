"""
@author: Angela Zhou
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# import cvxpy as cvx
from scipy.stats import norm
from scipy.spatial.distance import cdist, pdist, squareform
from copy import deepcopy
import math
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression


# imports

''' *** normalscale for Gamma; gamma >= 1
'''
def get_bnds_normalscale(est_Q,Gamma):
    n = len(est_Q)
    p_hi = np.multiply(Gamma, est_Q ) / (np.ones(n) - est_Q + np.multiply(Gamma, est_Q ))
    p_lo = np.multiply(-Gamma, est_Q ) / (np.ones(n) - est_Q + np.multiply(-Gamma, est_Q ))
    assert (p_lo < p_hi).all()
    a_bnd = 1/p_hi;
    b_bnd = 1/p_lo
    return [ a_bnd, b_bnd ]

''' *** DEPRECATED FOR HISTORICAL COMPATIBILITY
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
    return [est_Q, clf_dropped]
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

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def logistic_pol_asgn(theta, x):
    n = x.shape[0]
    theta = theta.flatten()
    if len(theta) == 1:
        logit = np.multiply(x, theta).flatten()
    else:
        logit = np.dot(x, theta).flatten()
    LOGIT_TERM_POS = np.ones(n) * 1.0 / (np.ones(n) + np.exp(-logit))
    return LOGIT_TERM_POS


def beta_w(th, data):
    X = data['x']
    t01 = data['t01']
    Y = data['y']
    a_ = data['a_']
    b_ = data['b_']
    W = np.diag(th)
    return np.linalg.inv(X.T * W * X) * X.T * W * Y

##############################################################################
# helpers for generating data
##############################################################################
''' generate data
X MVN
T probit conditional on X with random coefficient
Y is a random linear model of X
u is randomly generated
'''
def generate_data_random_mvn(n, p):
    x_sdp = np.random.normal(size=[n, p])
    # beta_x = [1,1]
    beta_x = np.random.normal(size=[1, p])
    X = np.mat(x_sdp)
    u = (np.random.rand(n) > 0.5).astype(
        int).flatten()  # binary unobserved confounder
    # propensities and generate T
    beta_T_u = np.random.normal(size=[1, p + 1])*0.3
    X_u = np.hstack([X, u.reshape([n, 1])])
    # probit model
    prob_T_1_u = np.clip(norm.cdf(np.dot(beta_T_u, X_u.T)).flatten(),
                         0.0001, 0.999)  # probit model
    prob_T_1 = np.clip(norm.cdf(np.dot(beta_T_u[0][:-1], X.T)).flatten(),
                       0.0001, 0.999)  # probit model
    T = np.array(np.random.uniform(size=n) < prob_T_1_u).astype(int).flatten()
    prob_T_1_u_obs = np.asarray(
        [prob_T_1_u[i] if T[i] == 1 else 1 - prob_T_1_u[i] for i in range(n)])
    prob_T_1_obs = np.asarray(
        [prob_T_1[i] if T[i] == 1 else 1 - prob_T_1[i] for i in range(n)])

    y_sdp = np.dot(beta_x, x_sdp.T) + T * np.dot(beta_x, x_sdp.T) + T * u * \
        2 + np.random.normal(size=n) * X[:, 0] * 0.4  # heteroskedastic noise
    Y = np.mat(y_sdp).reshape([n, 1])
    X = np.mat(x_sdp)
    return [X, T, Y, u, prob_T_1_obs, prob_T_1_u_obs]


def generate_data_random_mvn_intercept(n, p):
    x_sdp = np.random.normal(size=[n, p])
    x_sdp[:,0] = np.ones(n) #beta0 is intercept term
    beta_x = np.random.normal(size=[1, p])
    X = np.mat(x_sdp)
    u = (np.random.rand(n) > 0.5).astype(
        int).flatten()  # binary unobserved confounder
    # propensities and generate T
    beta_T_u = np.random.normal(size=[1, p + 1])*0.3
    X_u = np.hstack([X, u.reshape([n, 1])])
    # probit model
    prob_T_1_u = np.clip(norm.cdf(np.dot(beta_T_u, X_u.T)).flatten(),
                         0.0001, 0.999)  # probit model
    prob_T_1 = np.clip(norm.cdf(np.dot(beta_T_u[0][:-1], X.T)).flatten(),
                       0.0001, 0.999)  # probit model
    T = np.array(np.random.uniform(size=n) < prob_T_1_u).astype(int).flatten()
    prob_T_1_u_obs = np.asarray(
        [prob_T_1_u[i] if T[i] == 1 else 1 - prob_T_1_u[i] for i in range(n)])
    prob_T_1_obs = np.asarray(
        [prob_T_1[i] if T[i] == 1 else 1 - prob_T_1[i] for i in range(n)])

    y_sdp = np.dot(beta_x, x_sdp.T) + T * np.dot(beta_x, x_sdp.T) + T * u * \
        2 + np.random.normal(size=n) * X[:, 0] * 0.4  # heteroskedastic noise
    Y = np.mat(y_sdp).reshape([n, 1])
    X = np.mat(x_sdp)
    return [X, T, Y, u, prob_T_1_obs, prob_T_1_u_obs]



''' Sample from weight space
'''
def sample_uniformly(data):
    X_0_vec = data['x0']
    a_sdp = data['a_']
    b_sdp = data['b_']
    X = data['x']
    Y = data['y']; n = len(Y)
    N_SAMP = 10000
    W_samp = np.random.uniform(low=a_sdp, high=b_sdp, size = (N_SAMP,n))
    # Let's go through and randomly fix one coordinate of W_samp to be on the exterior (either a_sdp or b_sdp)
    for i in range(N_SAMP):
        ind = np.random.choice(n,size=5)
        W_samp[i, ind ] = (np.random.uniform()<0.5)*(b_sdp[ind]-a_sdp[ind])+a_sdp[ind]

    def beta(W):
        return np.linalg.inv(X.T*W*X)*X.T*W*Y
    betas = np.zeros([N_SAMP,p])
    for i in range(N_SAMP):
        betas[i,:] = beta(np.diag(W_samp[i,:])).flatten()

    plt.scatter(betas[:,0],betas[:,1] ,s=.4)

''' Helpers for Nonparametric case
'''
##### Code for nonparametric case
def find_opt_weights_short_val_mult(a_,b_,Y,p):
    [lda_opt, weights, s_wghts] = find_opt_weights_shorter(Y, a_*p, b_*p)
    return lda_opt


''' Takes in a set of points to evaluate on
s is bandwidth
(Doesn't differentiate between treatment or otherwise)
'''
def kernel_1d_bounds(xs, data, s):
    a_bnd = data['a_']
    b_bnd = data['b_']
    x = data['x']
    y = data['y']; n = len(y)
    q0 = data['q0']
    ps = data['true_Q'] # true probability of observed treatment

    dists = cdist(xs.reshape(-1,1), x.reshape(-1,1), 'sqeuclidean')
    K = np.exp(-dists / s**2)
    upper=[find_opt_weights_short_val_mult(a_bnd,b_bnd,y,kk) for kk in K]
    lower=[-find_opt_weights_short_val_mult(a_bnd,b_bnd,-1*y,kk) for kk in K]

    if plott:
        plt.plot(xs,(K/q0).dot(y)/(K/q0).sum(1),label='confounded kernel regression')
        plt.plot(xs,(K/ps).dot(y)/(K/ps).sum(1),label='IPW kernel regression for +1')
        plt.plot(xs,upper,'r--',label='robust kernel regression for +1 (upper)')
        plt.plot(xs,lower,'r-.',label='robust kernel regression for +1 (lower)')
    return [upper,lower]

# return the probability mass
def gaussian_kernel_int_cdf( h ):
    return exp(-1 * (np.linalg.norm(x1-x2)) / (2*variance))

 # alpha = kernel_int((t_lo-clip_tau[i])/h, 1)

''' Takes in a set of points to evaluate on
s is bandwidth
(Doesn't differentiate between treatment or otherwise)
mu_Xi: predicted at each xi
mu_xs: predicted at each x0 that we evaluate the bound at
'''
def kernel_1d_bounds_centered(xs, data, s, mu_xi, mu_xs, plott=True):
    a_bnd = data['a_']; b_bnd = data['b_']; x = data['x']
    y = data['y'] - mu_xi ; # center outcomes
    n = len(y); q0 = data['q0']
    ps = data['true_Q'] # true probability of observed treatment

    dists = cdist(xs.reshape(-1,1), x.reshape(-1,1), 'sqeuclidean')
    K = np.exp(-dists / s**2)
    upper=np.asarray([find_opt_weights_short_val_mult(a_bnd,b_bnd,y,kk) for kk in K])
    lower=np.asarray([-find_opt_weights_short_val_mult(a_bnd,b_bnd,-1*y,kk) for kk in K])

    if plott:
        plt.plot(xs,(K/q0).dot(y)/(K/q0).sum(1),label='confounded kernel regression, centered')
        plt.plot(xs,(K/ps).dot(y)/(K/ps).sum(1),label='IPW kernel regression, centered')
        plt.plot(xs,mu_xs + upper,'r--',label='robust kernel regression (upper), centered')
        plt.plot(xs,mu_xs + lower,'r-.',label='robust kernel regression (lower), centered')
        plt.legend()

    return [mu_xs+upper,mu_xs+lower]

'''
'''
def plot_cate_diff_gams(xs, data, GAMS, s = 0.5):
    data = data.copy()
    x = data['x']
    y = data['y']; n = len(y)
    t = data['tsgn']; # assuming signed
    q0 = data['q0']
    ps = data['true_Q'] # true probability of observed treatment
    uppers_T1 = [None] * len(GAMS); lowers_T1 = [None] * len(GAMS)
    uppers_T0 = [None] * len(GAMS); lowers_T0 = [None] * len(GAMS)

    for ind,gam in enumerate(GAMS):
        [a_bnd, b_bnd] = get_bnds(q0, gam)
        data['a_'] = a_bnd; data['b_'] = b_bnd
        if data['weight_type'] == "marginal":
            est_Q_1_xs = data['est_Q_1_xs']
            [a_xT1s, b_xT1s] = get_bnds(est_Q_1_xs, gam)
            [a_xT0s, b_xT0s] = get_bnds( np.ones(len(xs)) - est_Q_1_xs, gam)
            data['a_xT0s'] = a_xT0s; data['b_xT0s'] = b_xT0s
            data['a_xT1s'] = a_xT1s; data['b_xT1s'] = b_xT1s

        res = kernel_1d_cate_bounds(xs, data, s)
        conf_KR_T1 = res['conf_KR_T1']
        conf_KR_T0 = res['conf_KR_T0']
        uppers_T1[ind] = res['upperT1']; uppers_T0[ind] = res['upperT0']
        lowers_T1[ind] = res['lowerT1']; lowers_T0[ind] = res['lowerT0']
    plt.figure()
    [plt.plot(xs,  uppers_T1[i] - lowers_T0[i], label=r'lower, $\Gamma$='+str(gam) ) for i,gam in enumerate(GAMS) ]
    [plt.plot(xs, lowers_T1[i] - uppers_T0[i], label=r'upper, $\Gamma$='+str(gam) ) for i,gam in enumerate(GAMS) ]
    return [uppers_T1,lowers_T1,uppers_T0,lowers_T0]

def plot_mult_gams(xs, diff_gams_res, GAMS, colors):
    [uppers_T1,lowers_T1,uppers_T0,lowers_T0] = diff_gams_res
    plt.figure()
    [plt.plot(xs,  uppers_T1[i] - lowers_T0[i], color=colors[i] ) for i,gam in enumerate(GAMS) ]
    [plt.plot(xs, lowers_T1[i] - uppers_T0[i], label=r'$\Gamma$='+str(np.round(gam,2)), color=colors[i] ) for i,gam in enumerate(GAMS) ]
    return

# unconfounded
def plot_cate_unconf(xs, X, T, Y, s):
    dists1 = cdist(xs.reshape(-1,1), X[T==1].reshape(-1,1), 'sqeuclidean')
    dists0 = cdist(xs.reshape(-1,1), X[T==0].reshape(-1,1), 'sqeuclidean')
    K = np.exp(-dists1 / s**2)
    K0 = np.exp(-dists0 / s**2)
    Y1 = K.dot(Y[T==1])/K.sum(1)
    Y0 = K0.dot(Y[T==0])/K0.sum(1)
    plt.plot(xs,Y1 - Y0,label='unconf kernel regression',color='black')




''' Takes in a dict with data, confounded and 'true' propensities
a_xT0s, b_xT0s: bounds on the propensity of T=0 [ computed for the candidate points xs ]
a_xT1s, b_xT1s: bounds on the propensity of T=1 [ computed for the candidate points xs ]
'''
def kernel_1d_cate_bounds(xs, data, s= 0.5):
    a_bnd = data['a_']
    b_bnd = data['b_']
    x = data['x']
    y = data['y']; n = len(y)
    t = data['tsgn']; # assuming signed
    q0 = data['q0']
    ps = data['true_Q'] # true probability of observed treatment
    X_A = data['X_lo']
    X_B = data['X_hi']
    WGHT_FN = data['weight_type'] # toggle between scale_invariant and marginal
    if (WGHT_FN =='marginal'):
        a_xT0s = data['a_xT0s']; b_xT0s = data['b_xT0s']; n_ldas = data['n_ldas'] # get the upper,lower bound on feasible propensities
        a_xT1s = data['a_xT1s']; b_xT1s = data['b_xT1s']

    plt.style.use('ggplot')
    area_outside_bndy = 1 - np.asarray([np.maximum(norm.cdf( X_A, loc = x0, scale = s**2*0.5 ),1-norm.cdf( X_B, loc = x0, scale = s**2*0.5 ) ) for x0 in xs ]).flatten()

    alphas_1 = np.dot(np.diag(area_outside_bndy), np.ones([len(xs), len(y[t==1])]))
    alphas_0 = np.dot(np.diag(area_outside_bndy), np.ones([len(xs), len(y[t==-1])]))
    dists0 = cdist(xs.reshape(-1,1), x[t==-1].reshape(-1,1), 'sqeuclidean')
    dists1 = cdist(xs.reshape(-1,1), x[t==1].reshape(-1,1), 'sqeuclidean')
    K = np.divide(np.exp(-dists1 / s**2), alphas_1)
    K0 = np.divide(np.exp(-dists0 / s**2), alphas_0)
    dists_all = cdist(xs.reshape(-1,1), x.reshape(-1,1), 'sqeuclidean')
    alphas_all = np.dot(np.diag(area_outside_bndy), np.ones([len(xs), len(y)]))
    K_all = np.divide(np.exp(-dists_all / s**2), alphas_all)
    # K = np.exp(-dists / s**2)
    # K0 = np.exp(-dists0 / s**2)

    conf_KR_T1 = (K/q0[t==1]).dot(y[t==1])/(K/q0[t==1]).sum(1)
    true_ipw_KR_T1 = (K/ps[t==1]).dot(y[t==1])/(K/ps[t==1]).sum(1)
    conf_KR_T0 = (K0/q0[t==-1]).dot(y[t==-1])/(K0/q0[t==-1]).sum(1)
    true_ipw_KR_T0 = (K0/(ps[t==-1])).dot(y[t==-1])/(K0/(ps[t==-1])).sum(1)
    ## T=1
    print 'Computing bounds for T=1,'
    if (WGHT_FN =='scale_invariant'):
        upper=[find_opt_weights_short_val_mult(a_bnd[t==1],b_bnd[t==1],y[t==1],kk) for kk in K]
        lower=[-find_opt_weights_short_val_mult(a_bnd[t==1],b_bnd[t==1],-1*y[t==1],kk) for kk in K]
    else:
        upper=[find_opt_weights_short_marginal_val(y[t==1],kk/K_all[i].sum(), a_xT1s[i],b_xT1s[i],n_ldas) for i, kk in enumerate(K)]
        lower=[-find_opt_weights_short_marginal_val(-1*y[t==1],kk/K_all[i].sum(), a_xT1s[i],b_xT1s[i],n_ldas) for i, kk in enumerate(K)]
    plt.plot(xs,upper,'r--',label='robust kernel regression for +1 (upper)')
    plt.plot(xs,lower,'r-.',label='robust kernel regression for +1 (lower)')
    plt.plot(xs, conf_KR_T1,'r',label='confounded kernel regression for +1')
    # plt.plot(xs,(K/est_Q[t==1]).dot(y[t==1])/(K/est_Q[t==1]).sum(1),'m',label='IPW-corrected kernel regression for +1')
    plt.plot(xs, true_ipw_KR_T1 ,'b',label='true IPW-corrected kernel regression for +1')
    plt.legend(loc='upper left'); plt.title('T=1: confounded, robust bounds')
    ## T=0
    if (WGHT_FN =='scale_invariant'):
        upper0=[find_opt_weights_short_val_mult(a_bnd[t==-1],b_bnd[t==-1],y[t==-1],kk) for kk in K0]
        lower0=[-find_opt_weights_short_val_mult(a_bnd[t==-1],b_bnd[t==-1],-y[t==-1],kk) for kk in K0]
    else:
        upper0=[find_opt_weights_short_marginal_val(y[t==-1],kk/K_all[i].sum(), a_xT0s[i],b_xT0s[i],n_ldas) for i, kk in enumerate(K0)]
        lower0=[-find_opt_weights_short_marginal_val(-1*y[t==-1],kk/K_all[i].sum(),a_xT0s[i], b_xT0s[i], n_ldas) for i, kk in enumerate(K0)]
    plt.figure(); plt.plot(xs,upper0,'r--',label='robust kernel regression for -1 (upper)')
    plt.plot(xs,lower0,'r-.',label='robust kernel regression for -1 (lower)')
    plt.plot(xs, conf_KR_T0,'r',label='confounded kernel regression for -1')
    plt.plot(xs, true_ipw_KR_T0,'b',label='IPW-corrected kernel regression for -1')
    plt.legend(loc='upper left')

    plt.figure() # Plot T=1, T=0 together
    plt.plot(xs,upper0,linestyle='--',color='r',label='T==-1, upper')
    plt.plot(xs,lower0,linestyle=':',color='r',label='T==-1, lower')
    plt.plot(xs, conf_KR_T0,'r',label='T==-1, confounded kr')
    plt.plot(xs,upper,linestyle='--',color='b',label='T==1, upper')
    plt.plot(xs,lower,linestyle=':',color='b',label='T==1, lower')
    plt.plot(xs, conf_KR_T1,'b',label='T==1, confounded kr')

    plt.plot(xs,true_ipw_KR_T0,'purple',label='T==-1, ipw-corrected')
    plt.plot(xs,true_ipw_KR_T1,'purple',label='T==1, ipw-corrected')
    plt.legend(loc='upper left',bbox_to_anchor=(1.05,1)); plt.title('all regression functions and bounds')
    plt.figure() # plot CATE function
    plt.plot(xs,np.asarray(upper) - np.asarray(lower0),linestyle='--',color='r',label='CATE upper bound')
    plt.plot(xs,np.asarray(lower) - np.asarray(upper0),linestyle=':',color='b',label='CATE lower bound')
# In general, plotting the "true CATE" will depend ...
    # plt.plot(xs,-2*xs,linestyle='-.',color='b',label=' \'true\' CATE function')
    # add naive CATE estimation
    plt.plot(xs, conf_KR_T1 - conf_KR_T0 , linestyle=':' ,color='purple',label=' confounded CATE')
    plt.axhline(y=0,color='black',alpha=0.5); plt.title('CATE')
    res = { 'upperT1':np.asarray(upper), 'lowerT1':np.asarray(lower),
            'upperT0':np.asarray(upper0), 'lowerT0':np.asarray(lower0),
            'conf_KR_T1':conf_KR_T1, 'true_ipw_KR_T1':true_ipw_KR_T1, 'conf_KR_T0':conf_KR_T0, 'true_ipw_KR_T0':true_ipw_KR_T0 }
    return res
### Helpful testing code


def plt_conf(xs, data, s= 0.5):
    a_bnd = data['a_']
    b_bnd = data['b_']
    x = data['x']
    y = data['y']; n = len(y)
    t = data['tsgn']; # assuming signed
    q0 = data['q0']
    ps = data['true_Q'] # true probability of observed treatment

    dists = cdist(xs.reshape(-1,1), x[t==1].reshape(-1,1), 'sqeuclidean')
    K = np.exp(-dists / s**2)
    dists0 = cdist(xs.reshape(-1,1), x[t==-1].reshape(-1,1), 'sqeuclidean')
    K0 = np.exp(-dists0 / s**2)

    conf_KR_T1 = (K/q0[t==1]).dot(y[t==1])/(K/q0[t==1]).sum(1)
    conf_KR_T0 = (K0/q0[t==-1]).dot(y[t==-1])/(K0/q0[t==-1]).sum(1)
    return [conf_KR_T1, conf_KR_T0]


def plot_cate_unconf(xs, X, T, Y, s, X_A, X_B):
    dists1 = cdist(xs.reshape(-1,1), X[T==1].reshape(-1,1), 'sqeuclidean')
    dists0 = cdist(xs.reshape(-1,1), X[T==0].reshape(-1,1), 'sqeuclidean')
    area_outside_bndy = 1 - np.asarray([np.maximum(norm.cdf( X_A, loc = x0, scale = s**2*0.5 ),1-norm.cdf( X_B, loc = x0, scale = s**2*0.5 ) ) for x0 in xs ]).flatten()

    alphas_1 = np.dot(np.diag(area_outside_bndy), np.ones([len(xs), len(Y[T==1])]))
    alphas_0 = np.dot(np.diag(area_outside_bndy), np.ones([len(xs), len(Y[T==0])]))

    K = np.divide(np.exp(-dists1 / s**2), alphas_1)
    K0 = np.divide(np.exp(-dists0 / s**2), alphas_0)
    Y1 = K.dot(Y[T==1])/K.sum(1)
    Y0 = K0.dot(Y[T==0])/K0.sum(1)
    plt.plot(xs,Y1 - Y0,label='unconf kernel regression',color='black')


def bootstrap_resample(X, y, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)

    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    Y_resample = y[resample_i]
    return [X_resample, Y_resample]

def conf_KR_mu(xs, X, Y, q0, s, X_A, X_B):
    area_outside_bndy = 1 - np.asarray([np.maximum(norm.cdf( X_A, loc = x0, scale = s**2*0.5 ),1-norm.cdf( X_B, loc = x0, scale = s**2*0.5 ) ) for x0 in xs ]).flatten()
    alphas = np.dot(np.diag(area_outside_bndy), np.ones([len(xs), len(Y)]))
    dists = cdist(xs.reshape(-1,1), X.reshape(-1,1), 'sqeuclidean')
    K = np.divide(np.exp(-dists / s**2), alphas)
    mu = (K/q0).dot(Y)/(K/q0).sum(1)
    return mu

""" # number of samples
# alpha in [0,100]
"""
def bootstrap_KR(N_btstrp, xs, X, Y, q0, alpha, s, X_A, X_B):
    samps = np.zeros([N_btstrp, len(xs)])
    for i in range(N_btstrp):
        X_samp, Y_samp, q0_samp = resample( X, Y, q0 )
        samps[i,:] = conf_KR_mu(xs, X_samp, Y_samp, q0_samp, s, X_A, X_B)
    lower = np.percentile(samps, alpha*0.5, axis=0)
    upper = np.percentile(samps, 100-alpha*0.5, axis=0)
    return [lower, upper]

####
# New sensitivity model

''' Given  Y (unsorted), lower bound 'a_', upper bound 'b_' on weights, and possible index list sub_ind,
return Lambda (min problem value), weights, sum(weights)
'''
def find_opt_weights_short_marginal(Y, r, a_x0, b_x0, n_ldas, quiet=True):
    sort_inds = np.argsort(Y); Y = Y[sort_inds]; r = r[sort_inds]
    sort_Y = Y.sort()
    weights = np.zeros(len(Y));
    ldas = np.linspace(np.min(Y), np.max(Y), n_ldas)
    n = len(Y)
    vals = [ldas[i] + b_x0*np.sum(np.maximum(np.multiply(r,(Y-ldas[i])), np.zeros(len(Y)) )) \
    - a_x0 * np.sum(np.maximum(np.multiply(r,ldas[i] - Y) , np.zeros(len(Y)) )) for i in range(len(ldas)) ]
    lda_opt = np.min(vals); k_star = np.argmin(vals)
    if not quiet:
        plt.plot(ldas,vals)
    sort_inds_a = sort_inds[0:k_star]; sort_inds_b = sort_inds[k_star:]
    weights[sort_inds_a] = a_x0;
    weights[sort_inds_b] = b_x0

    return [lda_opt, weights, sum(weights)]
#####

def find_opt_weights_short_marginal_val(Y, r, a_x0, b_x0, n_ldas, quiet=True):
    [lda_opt, weights, sweights] = find_opt_weights_short_marginal(Y, r, a_x0, b_x0, n_ldas, quiet=True)
    return lda_opt

def get_policy_diff_gams(xs, diff_gams_res, GAMS, colors):
    def plot_mult_gams(xs, diff_gams_res, GAMS, colors):
        [uppers_T1,lowers_T1,uppers_T0,lowers_T0] = diff_gams_res
        plt.figure()
        [plt.plot(xs,  uppers_T1[i] - lowers_T0[i], color=colors[i] ) for i,gam in enumerate(GAMS) ]
        [plt.plot(xs, lowers_T1[i] - uppers_T0[i], label=r'$\Gamma$='+str(np.round(gam,2)), color=colors[i] ) for i,gam in enumerate(GAMS) ]
        return

# Percentile bootstrap on bounds
# e.g. kernel regression



### test
# fineness = 1
# xeval = np.asarray(xeval).flatten()
# for i in log_progress(range(fineness),every=1):
#     Xeval_eps = xeval;
#     Xeval_eps[dim_ind] = xeval[dim_ind] + range_end*i*1.0/fineness # replace with a general affine perturbation
#     data['x0'] = Xeval_eps.reshape([1,p]);  theta0 = np.random.uniform(X.shape[0])*(data['b_']-data['a_']) + data['a_']
#     data['sense'] = 'min';
#     [prob, W_sdp] = min_SDP_data(data, vbs=True)
#     W_sdp = np.asarray(W_sdp.value).flatten()
#     val = np.dot(beta_w(W_sdp, data).T, Xeval_eps.reshape([p,1]))
#     # check membership
#     [prob, W_feas, Z_feas] = check_membership_beta(beta_w(W_sdp, data), data)
#     print val
