from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(O_t = x_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

        # print(self.pi)
        # print(self.A)
        # print(self.B)
        # print(self.obs_dict)
        # print(self.state_dict)

        # print(self.B.shape)
    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(O_t = x_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - delta: (num_state*L) A numpy array delta[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################

        O = [self.obs_dict[obs] for obs in Osequence]
        # print("Osequence:" + str(Osequence))
        # print("O:" + str(O))
        # print("self.pi:" + str(self.pi))
        # print("self.B:" + str(self.B))
        delta[:,0] = self.pi * self.B[:,O[0]]

        for t in range(1, L):
            delta[:, t] = self.B[:, O[t]] * np.dot(np.transpose(self.A),delta[:,t-1])
        return delta

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(O_t = x_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array gamma[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        gamma = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        O = [self.obs_dict[obs] for obs in Osequence]

        gamma[:, L - 1] = 1

        for t in reversed(range(0, L - 1)):
            gamma[:, t] = np.dot(self.A, np.multiply(gamma[:, t + 1], self.B[:, O[t + 1]]))

        return gamma

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        O = [self.obs_dict[obs] for obs in Osequence]
        beta = self.backward(Osequence)
        prob = sum([beta[i, 0] * self.pi[i] * self.B[i, O[0]] for i in range(len(self.pi))])
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i | O, λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        O = [self.obs_dict[obs] for obs in Osequence]

        delta = self.forward(Osequence)
        gamma = self.backward(Osequence)
        beta = delta * gamma

        prob = beta / np.sum(delta[:, len(O) - 1])
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Edit here
        ###################################################
        O = []
        for obs in Osequence:
            if obs not in self.obs_dict:
                self.obs_dict[obs] = len(self.obs_dict)

                O.append(self.obs_dict[obs])
                new_col = np.array([0.000001]*len(self.state_dict)).T
                new_col = np.expand_dims(new_col, axis=1)
                self.B = np.hstack((self.B, new_col))
            else:
                O.append(self.obs_dict[obs])

        S = len(self.pi)
        N = len(O)
        delta = np.zeros([S, N])
        paths = np.zeros([S,N], dtype="int")
        tmpPath = np.zeros([N], dtype="int")
        for j in range(S):
            delta[j, 0] = self.pi[j] * self.B[j, O[0]]

        for t in range(1, N):
            for j in range(S):
                deltas = [delta[i, t - 1] * self.A[i, j] for i in range(S)]
                delta[j, t] = max(deltas) * self.B[j, O[t]]
                paths[j, t] = np.argmax(deltas)

        tmpPath[N-1] = np.argmax(delta[:,N-1])
        for t in reversed(range(1,N)):
            tmpPath[t-1] = paths[tmpPath[t],t]
        path = tmpPath.tolist()
        j = 0
        for i in path:
            for key, value in self.state_dict.items():
                if value == i:
                    path[j] = key
            j+=1

        return path
