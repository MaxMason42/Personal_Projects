
from Distributions import Gaussian, Binomial, Multinomial
import numpy as np
import math

class naiveBayesClassifier():
    def __init__(self):        
        self.X_train = np.array
        self.y_train = np.array
        self.features = {}
        self.priors = {}
        self.mode = ""
        self.conditional_probability = {}
        self.alpha = 1.0
        
    def fit(self, X, y, type = 'gaussian'):
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.mode = type
        
        for value in np.unique(y):
            self.priors[value] = np.count_nonzero(self.y_train == value) / len(self.y_train)
        
        
        if type == 'gaussian':
            self.mode = 'gaussian'
            self.gaussian_conditional_probabilities()
        elif type == 'bernoulli':
            self.mode = 'bernoulli'
            self.bernoulli_conditional_probabilities()
        elif type == 'multinomial':
            self.mode = 'multinomial'
            self.multinomial_conditional_probabilities()
        else:
            raise Exception("Non verified type. Values can either be 'gaussian', 'bernoulli', or 'multinomial'")
            
    def gaussian_conditional_probabilities(self):
        for feature in self.features:
            for value in np.unique(self.y_train):
                sigma = self.X_train[feature][np.where(self.y_train==value)[0]].std()
                mu = self.X_train[feature][np.where(self.y_train==value)[0]].mean()
                self.conditional_probability[(value, feature)] = Gaussian(mu, sigma)     
        
    def bernoulli_conditional_probabilities(self, alpha=1.0):
        self.alpha = alpha
        for feature in self.features:
            for value in np.unique(self.y_train):
                feature_count = np.count_nonzero(self.X_train[feature][np.where(self.y_train==value)[0]])
                p = (feature_count + self.alpha) / (np.count_nonzero(self.y_train == value) + self.alpha * 2)
                self.conditional_probability[(value, feature)] = Binomial(1, p)
    
    def multinomial_conditional_probabilities(self, alpha=1.0):
        self.alpha = alpha
        for feature in self.features:
            for value in np.unique(self.y_train):
                outcomes_counts = self.X_train[feature][np.where(self.y_train==value)[0]].value_counts()
                total_count = outcomes_counts.sum()
                outcomes_probs = ((outcomes_counts + self.alpha) / (total_count + self.alpha*len(outcomes_counts))).to_dict()
                self.conditional_probability[(value, feature)] = Multinomial(outcomes_probs)
    
    
    def predict(self, X):
        results = []
        X = np.array(X)
        
        for data in X:
            prob_outcomes = {}
            
            for outcome in np.unique(self.y_train):
                prior = self.priors[outcome]
                product = 1
                for i, feature in enumerate(self.features):
                    if data[i] not in self.X_train[feature][np.where(self.y_train==outcome)[0]].unique() and self.mode == 'multinomial':
                        outcome_probs = self.alpha / (self.X_train[feature][np.where(self.y_train==outcome)[0]].value_counts().sum() + 
                                                      self.alpha*len(self.X_train[feature][np.where(self.y_train==outcome)[0]].unique()))
                    elif data[i] not in self.X_train[feature][np.where(self.y_train==outcome)[0]].unique() and self.mode == 'binomial':
                        outcome_probs = self.alpha / (np.count_nonzero(self.y_train == value) + self.alpha * 2)
                    else:
                        outcome_probs = self.conditional_probability[(outcome, feature)].prob(data[i])
                    product *= outcome_probs
                
                posterior = prior * product
                prob_outcomes[outcome] = posterior
            results.append(max(prob_outcomes, key=prob_outcomes.get))
        return results
