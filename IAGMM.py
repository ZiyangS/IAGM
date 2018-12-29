import time
import copy
import numpy as np
from numpy.linalg import inv, det, slogdet
from utils import *
import mpmath
from scipy.stats import norm

class Sample:
    """Class for defining a single sample"""
    def __init__(self, mu, s_l, s_r, pi, lam, r, beta_l, beta_r, w_l, w_r, alpha, M):
        self.mu = mu
        self.s_l = s_l
        self.s_r = s_r
        self.pi = np.reshape(pi, (1, -1))
        self.lam = lam
        self.r = r
        self.beta_l = beta_l
        self.beta_r = beta_r
        self.w_l = w_l
        self.w_r = w_r
        self.M = M
        self.alpha = alpha

class Samples:
    """Class for generating a collection of samples"""
    def __init__(self, N, D):
        self.sample = []
        self.N = N
        self.D = D

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self, S):
        return self.sample.append(S)

def infinte_mixutre_model(X, Nsamples=1000, Nint=50, anneal=False):
    """
    infinite asymmetric gaussian distribution(AGD) mixture model
    using Gibbs sampling
    input:
        Y : the input datasets
        Nsamples : the number of Gibbs samples
        Nint : the samples used for evaluating the tricky integral
        anneal : perform simple siumulated annealing
    output:
        Samp : the output samples
        Y : the input datasets
    """
    # compute some data derived quantities, N is observations number, D is dimensionality number
    N, D = X.shape
    muy = np.mean(X, axis=0)
    vary = np.zeros(D)
    for i in range(D):
        vary[i] = np.var(X[:, i])

    # initialise a single sample
    Samp = Samples(Nsamples, D)

    c = np.zeros(N)            # initialise the stochastic indicators
    pi = np.zeros(1)           # initialise the weights
    mu = np.zeros((1, D))      # initialise the means
    s_l = np.zeros((1, D))    # initialise the precisions
    s_r = np.zeros((1, D))    # initialise the precisions
    n = np.zeros(1)            # initialise the occupation numbers

    # set first mu to the mean of all data
    mu[0,:] = muy
    # set first pi to 1, because only one component initially
    pi[0] = 1.0

    # draw beta from prior
    # (beta)^(-1) is subject to Rasmussen's gamma(1,1), eq 7 (Rasmussen 2000)
    # gamma distribution (Rasmussen 2000) use scale and mean, which is different
    # alpha need to change to 1/2*alpha
    # theta parameter change to mean, theta*(1/alpha). So second parameter of Rasmussen's should be 2*theta/a
    beta_l = np.array([np.squeeze(draw_invgamma(0.5, 2)) for d in range(D)])
    beta_r = np.array([np.squeeze(draw_invgamma(0.5, 2)) for d in range(D)])
    # beta_l = np.array([np.squeeze(draw_gamma(0.5, 2)) for d in range(D)])
    # beta_r = np.array([np.squeeze(draw_gamma(0.5, 2)) for d in range(D)])

    # draw w from prior
    # w is subject ot Rasmussen's gamma(1, vary) , eq 7 (Rasmussen 2000)
    # which means its subject to standard gamma(0.5, 2*vary)
    w_l = np.array([np.squeeze(draw_gamma(0.5, 2*vary[k])) for k in range(D)])
    w_r = np.array([np.squeeze(draw_gamma(0.5, 2*vary[k])) for k in range(D)])

    # draw s_l, s_r from prior
    # S_ljk, S_rjk is subject to gamma(beta, (w)^(-1)), eq 8 (Rasmussen 2006)
    # which means its subject to standard gamma(0.5*beta, 2/(beta*w))
    # initially, there is only one component, j=1, so the index will set to 0
    s_l[0, :] = np.array([np.squeeze(draw_gamma(beta_l[k]/2, 2/(beta_l[k]*w_l[k]))) for k in range(D)])
    s_r[0, :] = np.array([np.squeeze(draw_gamma(beta_r[k]/2, 2/(beta_r[k]*w_r[k]))) for k in range(D)])
    print(s_l)
    time.sleep(100)

    # initially, all samples are in the only component
    n[0] = N

    # draw lambda from prior
    # lambda is subject to Guassian(muy, vary), eq 3 (Rasmussen 2000)
    lam = np.array([np.squeeze(draw_normal(muy[k], vary[k])) for k in range(D)])

    # draw r from prior
    # precision r is subject to gamma(1, (vary)^(-1)), eq 3 (Rasmussen 2000)
    # which means its subject to standard gamma(0.5, 2/vary)
    r = np.array([np.squeeze(draw_gamma(0.5, 2/vary[k])) for k in range(D)])

    # draw alpha from prior
    # (alpha)^(-1) is subject to Rasmussen's paper's gamma distribution, scale is 1, mean is 1, eq 14 (Rasmussen 2006)
    # so (alphs)^(-1) is sujbect to gamma distribution, scale is 1/2, and theta is 2
    alpha = 1.0/draw_gamma(0.5, 2.0)
    # set only 1 component, m is the component number
    M = 1
    # define the sample
    S = Sample(mu, s_l, s_r, pi, lam, r, beta_l, beta_r, w_l, w_r, alpha, M)

    Samp.addsample(S)                           # add the sample
    print('{}: initialised parameters'.format(time.asctime()))

    # loop over samples
    z = 1
    oldpcnt = 0
    while z < Nsamples:
        print("start")
        print(z)
        # define simulated annealing temperature
        G = max(1.0, float(0.5*Nsamples)/float(z + 1)) if anneal else 1.0

        # recompute muy and covy
        muy = np.mean(X, axis=0)
        for k in range(D):
            vary[k] = np.var(X[:, k])
        precisiony= 1/vary

        # the observations belonged to class j
        Xj = [X[np.where(c==j), :] for j, nj in enumerate(n)][0]
        mu_cache = mu
        mu = np.zeros((M, D))
        j = 0
        # draw muj from posterior (depends on sj, c, lambda, r), eq 4 (Rasmussen 2000)
        for x, nj, s_lj, s_rj in zip(Xj, n, s_l, s_r):
            # for every dimensionality, compute the posterior distribution of mu_jk
            for k in range(D):
                x_k = x[:, k]
                # p represent the number of x_ik < mu_jk
                p = x_k[x_k < mu_cache[j][k]].shape[0]
                # q represent the number of x_ik >= mu_jk, q = n - p
                q = nj - p
                # x_l_sum represents the sum from i to n of x_ik, which x_ik < mu_jk
                x_l_sum = np.sum(x_k[x_k < mu_cache[j][k]])
                # x_r_sum represents the sum from i to n of x_ik, which x_ik >= mu_jk
                x_r_sum = np.sum(x_k[x_k >= mu_cache[j][k]])
                r_n = r[k] + p * s_lj[k] + q * s_rj[k]
                mu_n = (s_lj[k] * x_l_sum + s_rj[k] * x_r_sum + r[k] * lam[k])/r_n
                mu[j, k] = norm.rvs(mu_n, 1/r_n)
            j += 1

        # draw lambda from posterior (depends on mu, M, and r), eq 5 (Rasmussen 2000)
        mu_sum = np.sum(mu, axis=0)
        for k in range(D):
            scale = 1/(precisiony[k] + M * r[k])
            loc = scale * (muy[k]*precisiony[k] + r[k]* mu_sum[k])
            lam[k] = draw_normal(loc=loc, scale=scale)

        # draw r from posterior (depnds on M, mu, and lambda), eq 5 (Rasmussen 2000)
        temp_para_sum = np.zeros(D)
        for k in range(D):
            for muj in mu:
                temp_para_sum[k] += np.outer((muj[k] - lam[k]), np.transpose(muj[k] - lam[k]))
        r = np.array([np.squeeze(draw_gamma((M+1)/2, 2/(vary[k] + temp_para_sum[k]))) for k in range(D)])

        # draw alpha from posterior (depends on k, N), eq 15 (Rasmussen 2000)
        # Because its not standard form, using ARS to sampling
        alpha = draw_alpha(k, N)

        # draw sj from posterior (depends on mu, c, beta, w), eq 8 (Rasmussen 2000)
        for j, nj in enumerate(n):
            Xj = X[np.where(c == j), :][0]
            # for every dimensionality, compute the posterior distribution of s_ljk, s_rjk
            for k in range(D):
                x_k = Xj[:, k]
                # # p represent the number of x_ik < mu_jk
                # p = x_k[x_k < mu[j][k]].shape[0]
                # # q represent the number of x_ik >= mu_jk, q = n - p
                # q = x_k[x_k >= mu[j][k]].shape[0]
                # x_l represents the data from i to n of x_ik, which x_ik < mu_jk
                x_l = x_k[x_k < mu[j][k]]
                # x_r represents the data from i to n of x_ik, which x_ik >= mu_jk
                x_r = x_k[x_k >= mu[j][k]]
                cumculative_sum_left_equation = np.squeeze(np.outer((x_l[k] - mu[j][k]), np.transpose(x_l[k] - mu[j][k])))
                cumculative_sum_right_equation = np.squeeze(np.outer((x_r[k] - mu[j][k]), np.transpose(x_r[k] - mu[j][k])))

                # def Metropolis_Hastings_Sampling_posterior_sljk(s_ljk, s_rjk, nj, beta, w, sum):
                s_l[j][k] = Metropolis_Hastings_Sampling_posterior_sljk(s_ljk=s_l[j][k], s_rjk=s_r[j][k],
                                                nj=nj, beta=beta_l[k], w=w_l[k], sum=cumculative_sum_left_equation)
                s_r[j][k] = Metropolis_Hastings_Sampling_posterior_srjk(s_ljk=s_l[j][k], s_rjk=s_r[j][k],
                                                nj=nj, beta=beta_r[k], w=w_r[k], sum=cumculative_sum_right_equation)

        # compute the unrepresented probability - apply simulated annealing, eq 17 (Rasmussen 2000)
        p_unrep = (alpha / (N - 1.0 + alpha)) * integral_approx(X, lam, r, beta_l, beta_r, w_l, w_r, G, size=Nint)
        p_indicators_prior = np.outer(np.ones(k + 1), p_unrep)
        # print(p_unrep)
        # time.sleep(100)


        # for the represented components, eq 17 (Rasmussen 2000)
        for j in range(M):
            # n-i,j : the number of oberservations, excluding Xi, that are associated with component j
            nij = n[j] - (c == j).astype(int)
            # print(nij)
            idx = np.argwhere(nij > 0)
            idx = idx.reshape(idx.shape[0])
            likelihood_for_associated_data = np.ones(len(idx))
            for i in idx:
                # print(i)
                for k in range(D):
                    # print(k)
                    if X[i][k] < mu[j][k]:
                        likelihood_for_associated_data[i] *= 1 / (np.power(s_l[j][k], -0.5) + np.power(s_r[j][k], -0.5))* \
                        np.exp(- 0.5 * s_l[j][k] * (X[i][k] - mu[j][k]) ** 2)
                    else:
                        likelihood_for_associated_data[i] *= 1 / (np.power(s_l[j][k], -0.5) + np.power(s_r[j][k], -0.5))* \
                        np.exp(- 0.5 * s_r[j][k] * (X[i][k] - mu[j][k]) ** 2)
                    # print(likelihood_for_associated_data[i])
            p_indicators_prior[j, idx] = nij[idx]/(N - 1.0 + alpha)*likelihood_for_associated_data
        print(p_indicators_prior)
        # stochastic indicator (we could have a new component)
        c = np.hstack(draw_indicator(p_indicators_prior))
        # print(c)


        # draw w from posterior (depends on k, beta, D, sj), eq 9 (Rasmussen 2000)
        w_l = np.array([np.squeeze(draw_gamma(0.5 *(M*beta_l[k]+1),\
                        2/(vary[k] + beta_l[k] * np.sum(s_l, axis=0)[k])))\
                        for k in range(D)])
        w_r = np.array([np.squeeze(draw_gamma(0.5 *(M*beta_r[k]+1),\
                        2/(vary[k] + beta_r[k] * np.sum(s_r, axis=0)[k])))\
                        for k in range(D)])

        # draw beta from posterior (depends on k, s, w), eq 9 (Rasmussen 2000)
        # Because its not standard form, using ARS to sampling.
        beta_l = np.array([draw_beta_ars(w_l, s_l, M, k)[0] for k in range(D)])
        beta_r = np.array([draw_beta_ars(w_l, s_l, M, k)[0] for k in range(D)])
        # beta_l = np.array([draw_beta(beta_l, w_l, s_l, M, k) for k in range(D)])
        # beta_r = np.array([draw_beta(beta_r, w_r, s_r, M, k) for k in range(D)])

        # sort out based on new stochastic indicators
        nij = np.sum(c == M)        # see if the *new* component has occupancy
        if nij > 0:
            # draw from priors and increment M
            newmu = np.array([np.squeeze(norm.rvs(loc=lam[k], scale=1 / r[k], size=1)) for k in range(D)])
            news_l = np.array([np.squeeze(draw_gamma(beta_l[k] / 2, 2 / (beta_l[k] * w_l[k]))) for k in range(D)])
            news_r = np.array([np.squeeze(draw_gamma(beta_r[k] / 2, 2 / (beta_r[k] * w_r[k]))) for k in range(D)])
            mu = np.concatenate((mu, np.reshape(newmu, (1, D))))
            s_l = np.concatenate((s_l, np.reshape(news_l, (1, D))))
            s_r = np.concatenate((s_r, np.reshape(news_r, (1, D))))
            M = M + 1
        # find the associated number for every components
        n = np.array([np.sum(c == j) for j in range(M)])

        # find unrepresented components
        badidx = np.argwhere(n == 0)
        Nbad = len(badidx)

        # remove unrepresented components
        if Nbad > 0:
            mu = np.delete(mu, badidx, axis=0)
            s_l = np.delete(s_l, badidx, axis=0)
            s_r = np.delete(s_r, badidx, axis=0)
            # if the unrepresented compont removed is in the middle, make the sequential component indicators change
            for cnt, i in enumerate(badidx):
                idx = np.argwhere(c >= (i - cnt))
                c[idx] = c[idx] - 1
            M -= Nbad        # update component number

        # recompute n
        n = np.array([np.sum(c == j) for j in range(M)])

        # recompute pi
        pi = n.astype(float)/np.sum(n)

        pcnt = int(100.0 * z / float(Nsamples))
        if pcnt > oldpcnt:
            print('{}: %--- {}% complete ----------------------%'.format(time.asctime(), pcnt))
            oldpcnt = pcnt

        # add sample
        S = Sample(mu, s_l, s_r, pi, lam, r, beta_l, beta_r, w_l, w_r, alpha, M)
        newS = copy.deepcopy(S)
        Samp.addsample(newS)
        z += 1
        print(M)
        print(n)
        # time.sleep(10)
        # print(c)

    return Samp, X, c, n

