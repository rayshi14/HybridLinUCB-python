#!/usr/bin/env python2.7

import numpy as np
import math
import sys
from scipy import linalg

class HybridUCB:
    def __init__(self):
        self.article_features = {}
    
        # upper bound coefficient
        self.alpha = 3 #1 + np.sqrt(np.log(2/delta)/2)
        self.r1 = 0.5
        self.r0 = -20
        # dimension of user features = d
        self.d = 5
        # dimension of article features = k
        self.k = self.d*self.d
        # A0 : matrix to compute hybrid part, k*k
        self.A0 = np.identity(self.k)
        self.A0I = np.identity(self.k)
        # b0 : vector to compute hybrid part, k
        self.b0 = np.zeros((self.k, 1))
        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = {}
        # AaI : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}
        # Ba : collection of matrix to compute hybrid part, d*k
        self.Ba = {}
        # BaT : collection of matrix to compute hybrid part, d*k
        self.BaT = {}
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = {}
        
        
        # other dicts to spped up computation
        self.AaIba = {}
        self.AaIBa = {}
        self.BaTAaI = {}
        self.theta = {}

        self.beta = np.zeros((self.k, 1))
        
        self.index = {}
        
        self.a_max = 0
        
        self.z = None
        self.zT = None
        self.xaT = None
        self.xa = None

    def get_Aa(self):
        return self.Aa

    def set_Aa(self, Aa):
        self.Aa = Aa
    
    def get_ba(self):
        return self.ba
    
    def set_ba(self, ba):
        self.ba = ba
    
    def get_AaI(self):
        return self.AaI
    
    def set_AaI(self, AaI):
        self.AaI = AaI
    
    def get_theta(self):
        return self.theta

    def set_theta(self, theta):
        self.theta = theta

    # Evaluator will call this function and pass the article features.
    # Check evaluator.py description for details.
    def set_articles(self, art):
        # init collection of matrix/vector Aa, Ba, ba
        i = 0
        art_len = len(art)
        self.article_features = np.zeros((art_len, 1, self.d))
        self.Aa = np.zeros((art_len, self.d, self.d))
        self.AaI = np.zeros((art_len, self.d, self.d))
        self.Ba = np.zeros((art_len, self.d, self.k))
        self.BaT = np.zeros((art_len, self.k, self.d))
        self.ba = np.zeros((art_len, self.d, 1))
        self.AaIba = np.zeros((art_len, self.d, 1))
        self.AaIBa = np.zeros((art_len, self.d, self.k))
        self.BaTAaI = np.zeros((art_len, self.k, self.d))
        self.theta = np.zeros((art_len, self.d, 1))
        for key in art:
            self.index[key] = i
            self.article_features[i] = art[key][1:]
            self.Aa[i] = np.identity(self.d)
            self.AaI[i] = np.identity(self.d)
            self.Ba[i] = np.zeros((self.d, self.k))
            self.BaT[i] = np.zeros((self.k, self.d))
            self.ba[i] = np.zeros((self.d, 1))
            self.AaIba[i] = np.zeros((self.d, 1))
            self.AaIBa[i] = np.zeros((self.d, self.k))
            self.BaTAaI[i] = np.zeros((self.k, self.d))
            self.theta[i] = np.zeros((self.d, 1))
            i += 1
    

    # This function will be called by the evaluator.
    # Check task description for details.
    def update(self, reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0
            
            self.A0 += np.dot(self.BaTAaI[self.a_max], self.Ba[self.a_max])
            self.b0 += np.dot(self.BaTAaI[self.a_max], self.ba[self.a_max])
            self.Aa[self.a_max] += np.dot(self.xa, self.xaT)
            self.AaI[self.a_max] = linalg.inv(self.Aa[self.a_max])
            self.Ba[self.a_max] += np.dot(self.xa, self.zT)
            self.BaT[self.a_max] = np.transpose(self.Ba[self.a_max])
            self.ba[self.a_max] += r * self.xa
            self.AaIba[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
            self.AaIBa[self.a_max] = np.dot(self.AaI[self.a_max], self.Ba[self.a_max])
            self.BaTAaI[self.a_max] = np.dot(self.BaT[self.a_max], self.AaI[self.a_max])
            
            self.A0 += np.dot(self.z, self.zT) - np.dot(self.BaTAaI[self.a_max], self.Ba[self.a_max])
            self.b0 += r * self.z - np.dot(self.BaT[self.a_max], np.dot(self.AaI[self.a_max], self.ba[self.a_max]))
            self.A0I = linalg.inv(self.A0)
            self.beta = np.dot(self.A0I, self.b0)
            self.theta = self.AaIba - np.dot(self.AaIBa, self.beta)#self.AaI[article].dot(self.ba[article] - self.Ba[article].dot(self.beta))
                
        else:
        # error
            pass
    
    # This function will be called by the evaluator.
    # Check task description for details.
	# Use vectorized code to increase speed
    def reccomend(self, timestamp, user_features, articles):
        article_len = len(articles)
        # za : feature of current user/article combination, k*1
        self.xaT = np.array([user_features[1:]])
        self.xa = np.transpose(self.xaT)
        # recommend using hybrid ucb
        # fast vectorized for loops
        
        index = [self.index[article] for article in articles]
        
        article_features_tmp = self.article_features[index]

        zaT_tmp = np.einsum('i,j', article_features_tmp.reshape(-1), user_features[1:]).reshape(article_len, 1, self.k)
        za_tmp = np.transpose(zaT_tmp, (0,2,1))#np.transpose(zaT_tmp,(0,2,1))
    
        #np.dot(self.A0I, np.dot(BaTAaI_tmp, self.xa)) (20, 36, 1)
        A0IBaTAaIxa_tmp = np.transpose(np.dot(np.transpose(np.dot(self.BaTAaI[index], self.xa), (0,2,1)), np.transpose(self.A0I)), (0,2,1))
        
        A0Iza_tmp = np.transpose(np.dot(zaT_tmp, np.transpose(self.A0I)), (0,2,1)) # (20, 36, 1)
        A0Iza_diff_2A0IBaTAaIxa_tmp = A0Iza_tmp - 2*A0IBaTAaIxa_tmp
        
        # np.dot(zaT_tmp, A0Iza_diff_2A0IBaTAaIxa_tmp), (20, 1, 1)
        sa_1_tmp = np.sum(za_tmp.reshape(article_len,self.k,1,1)*A0Iza_diff_2A0IBaTAaIxa_tmp.reshape(article_len, self.k,1,1),-3)
        
        # np.dot(AaIBa_tmp, A0IBaTAaIxa_tmp)
        AaIxa_add_AaIBaA0IBaTAaIxa_tmp = np.dot(self.AaI[index], self.xa) + np.sum(np.transpose(self.AaIBa[index], (0,2,1)).reshape(article_len, self.k,self.d,1)*A0IBaTAaIxa_tmp.reshape(article_len,self.k,1,1),-3)
        sa_2_tmp = np.transpose(np.dot(np.transpose(AaIxa_add_AaIBaA0IBaTAaIxa_tmp,(0,2,1)),self.xa),(0,2,1))
        sa_tmp = sa_1_tmp + sa_2_tmp
        # np.dot(self.xaT, self.theta[article])
        xaTtheta_tmp = np.transpose(np.dot(np.transpose(self.theta[index],(0,2,1)),self.xa),(0,2,1))
        
        max_index = np.argmax(np.dot(zaT_tmp, self.beta) + xaTtheta_tmp + self.alpha * np.sqrt(sa_tmp))
        
        self.z = za_tmp[max_index]
        self.zT = zaT_tmp[max_index]
        art_max = index[max_index]
        
        # article index with largest UCB
        # global a_max, entries
        self.a_max = art_max
        
        # entries += 1
        
        # if entries % 100 == 0:
        #     print entries, evaluated, clicked, clicked / evaluated 
        
        return articles[max_index]

# lin UCB
class LinUCB:
    def __init__(self):
        # upper bound coefficient
        self.alpha = 0.25 # if worse -> 2.9, 2.8 1 + np.sqrt(np.log(2/delta)/2)
        self.r1 = 1 # if worse -> 0.7, 0.8
        self.r0 = 0 # if worse, -19, -21
        # dimension of user features = d
        self.d = 6
        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = {}
        # AaI : store the inverse of all Aa matrix
        self.AaI = {}
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = {}
        
        self.a_max = 0
        
        self.theta = {}
        
        self.x = None
        self.xT = None
        # linUCB

    def get_Aa(self):
        return self.Aa

    def set_Aa(self, Aa):
        self.Aa = Aa
    
    def get_ba(self):
        return self.ba
    
    def set_ba(self, ba):
        self.ba = ba
    
    def get_AaI(self):
        return self.AaI
    
    def set_AaI(self, AaI):
        self.AaI = AaI
    
    def get_theta(self):
        return self.theta

    def set_theta(self, theta):
        self.theta = theta

    def set_articles(self, art):
        # init collection of matrix/vector Aa, Ba, ba
        for key in art:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d, 1))
            
    def update(self, reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0
            self.Aa[self.a_max] += np.dot(self.x, self.xT)
            self.ba[self.a_max] += r * self.x
            self.AaI[self.a_max] = linalg.solve(self.Aa[self.a_max], np.identity(self.d))
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
        else:
        # error
            pass
    
    def reccomend(self, timestamp, user_features, articles):
        xaT = np.array([user_features])
        xa = np.transpose(xaT)
        art_max = -1
        old_pa = 0
        
        AaI_tmp = np.array([self.AaI[article] for article in articles])
        theta_tmp = np.array([self.theta[article] for article in articles])
        art_max = articles[np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]

        self.x = xa
        self.xT = xaT
        # article index with largest UCB
        self.a_max = art_max
        
        return self.a_max  

LinUCBObj = None
HybridUCBObj = None

t = 0
break_point = 0

def set_articles(art):
    global HybridUCBObj, LinUCBObj
    LinUCBObj = LinUCB()
    HybridUCBObj = HybridUCB()
    LinUCBObj.set_articles(art)
    HybridUCBObj.set_articles(art)
    #UCB2Obj = UCB2()
    #UCB2Obj.set_articles(art)

def update(reward):
    if t < break_point:
        return LinUCBObj.update(reward)
    else:
        return HybridUCBObj.update(reward)

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global t
    t+=1

    if t==break_point:
        HybridUCBObj.set_Aa(LinUCBObj.get_Aa())
        HybridUCBObj.set_ba(LinUCBObj.get_ba())
        HybridUCBObj.set_AaI(LinUCBObj.get_AaI())
        HybridUCBObj.set_theta(LinUCBObj.get_theta())

    if t < break_point:
        return LinUCBObj.reccomend(timestamp, user_features, articles)
    else:
        return HybridUCBObj.reccomend(timestamp, user_features, articles)
        
