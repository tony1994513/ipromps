#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import scipy.linalg
import scipy.stats as stats
from scipy.stats import multivariate_normal as mvn


class ProMP(object):
    """
    Uni-dimensional probabilistic MP
    """
    def __init__(self, num_basis=31, sigma_basis=0.05, num_samples=101, sigmay=0.0):
        self.x = np.linspace(0.0, 1.0, num_samples)     # the time stamp
        self.num_samples = num_samples      # num of samples
        self.num_basis = num_basis          # num of basis func
        self.sigma_basis = sigma_basis      # the sigma of basis func
        self.C = np.arange(0,num_basis)/(num_basis-1.0)     # the mean of basis func
        self.sigmaSignal = float('inf')     # the zero-mean noise, including modelling error and the system noise
        self.Phi = np.exp(-.5*(np.array(map(lambda x: x-self.C, np.tile(self.x, (self.num_basis, 1)).T)).T**2 / (self.sigma_basis**2)))
        self.viapoints = []     # the via point list
        self.sigmay = sigmay    # the measurement noise cov mat for updated distribution
        self.W = np.array([])   # the weight for each demo
        self.Y = np.empty((0, self.num_samples), float)     # the demon traj array

        # the unit promp prior W distribution
        self.meanW = None
        self.sigmaW = None

        # the unit promp updated W distribution
        self.meanW_uUpdated = None
        self.sigmaW_uUpdated = None

        # the n-dimension promps updated W distribution
        self.meanW_nUpdated = None
        self.sigmaW_nUpdated = None
        # the n-dimension promps fit alpha
        self.alpha_fit = None

    def add_demonstration(self, demonstration):
        """
        add demonstration to train promp
        :param demonstration:
        :return:
        """
        interpolate = interp1d(np.linspace(0, 1, len(demonstration)), demonstration, kind='cubic')
        stretched_demo = interpolate(self.x)
        self.Y = np.vstack((self.Y, stretched_demo))
        self.W = np.dot(np.linalg.inv(np.dot(self.Phi, self.Phi.T)), np.dot(self.Phi, self.Y.T)).T
        self.meanW = np.mean(self.W, 0) if self.num_demos>1 else None
        self.sigmaW = np.cov(self.W.T) if self.num_demos>1 else None
        self.sigmaSignal = np.sum(np.sum((np.dot(self.W, self.Phi) - self.Y)**2)) / (self.num_demos*self.num_samples)

    @property
    def noise(self):
        """
        compute the regression noise
        :return:
        """
        return self.sigmaSignal

    @property
    def num_demos(self):
        """
        the number of demonstration
        :return:
        """
        return self.Y.shape[0]

    @property
    def num_points(self):
        return self.Y.shape[1]

    @property
    def num_viapoints(self):
        return len(self.viapoints)

    @property
    def goal_bounds(self):
        """
        Joint boundaries of the last point
        :return: (lower boundary, upper boundary)
        """
        return self._get_bounds(-1)

    @property
    def goal_mean(self):
        """
        Mean of the last point
        :return: scalar
        """
        return self._get_mean(-1)

    def get_bounds(self, t):
        """
        Return the bounds at time t
        :param t: 0 <= t <= 1
        :return: (lower boundary, upper boundary)
        """
        return self._get_bounds(int(self.num_points*t))

    def get_mean(self, t):
        """
        Return the mean at time t
        :param t: 0 <= t <= 1
        :return: scalar
        """
        return self._get_mean(int(self.num_points*t))

    def get_std(self):
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
        return std

    def _get_mean(self, t_index):
        mean = np.dot(self.Phi.T, self.meanW)
        return mean[t_index]

    def _get_bounds(self, t_index):
        mean = self._get_mean(t_index)
        std = self.get_std()
        return mean-std, mean+std

    def clear_viapoints(self):
        del self.viapoints[:]

    def add_viapoint(self, t, obsy):
        """
        Add a viapoint to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param obsy: observed value at time t
        :return:
        """
        self.viapoints.append({'t': t, 'obsy': obsy})

    def param_updata(self, randomness=1e-10):
        """
        :param randomness:
        :return:
        """
        newMean = self.meanW
        newSigma = self.sigmaW
        # conditioning
        for viapoint in self.viapoints:
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(viapoint['t'], (self.num_basis, 1)).T)).T ** 2 / (self.sigma_basis ** 2)))
            aux = self.sigmay + np.dot(np.dot(PhiT.T, newSigma), PhiT)
            K = np.dot(newSigma, PhiT) * 1 / aux
            newMean = newMean + np.dot(K, (viapoint['obsy'] - np.dot(PhiT.T, newMean)))
            newSigma = newSigma - np.dot(K, np.dot(PhiT.T, newSigma))
        # save the updated distribution
        self.meanW_uUpdated = newMean
        self.sigmaW_uUpdated = newSigma

    def gen_uTrajectory(self, randomness=1e-10):
        """
        Outputs a trajectory from unit promp
        :param randomness: float between 0. (output will be the mean of gaussians) and 1. (fully randomized inside the variance)
        :return: a 1-D vector of the generated points
        """
        newMean = self.meanW
        newSigma = self.sigmaW
        # conditioning
        for viapoint in self.viapoints:
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(viapoint['t'], (self.num_basis, 1)).T)).T ** 2 / (self.sigma_basis ** 2)))
            aux = self.sigmay + np.dot(np.dot(PhiT.T, newSigma), PhiT)
            K = np.dot(newSigma, PhiT) * 1 / aux
            newMean = newMean + np.dot(K, (viapoint['obsy'] - np.dot(PhiT.T, newMean)))
            newSigma = newSigma - np.dot(K, np.dot(PhiT.T, newSigma))
        # save the updated distribution
        self.meanW_uUpdated = newMean
        self.sigmaW_uUpdated = newSigma
        sampW = np.random.multivariate_normal(newMean, randomness*newSigma, 1).T
        return np.dot(self.Phi.T, sampW)

    def plot_prior(self,  b_distribution=True, color='b', alpha_std=0.4, linewidth_mean=2,
                   b_regression=False, b_dataset=False, linewidth_data=2):
        """
        plot the prior distribution from training sets
        :param legend: the figure legend
        :param b_distribution: the 1 std envelope and mean
        :param color: the color of envelope and mean
        :param alpha_std: the transparency of envelope
        :param linewidth_mean:
        :param b_regression:
        :param b_dataset: the dataset to train to model
        :param linewidth_data:
        :return:
        """
        x = self.x
        # the probability distribution
        if b_distribution:
            mean = np.dot(self.Phi.T, self.meanW)
            std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
            plt.fill_between(x, mean-std, mean+std, color=color, alpha=alpha_std,label="Initial distribution")
            plt.plot(x, mean, color=color, label="Initial distribution mean", linewidth=linewidth_mean)
        # the regression result
        if b_regression:
            for w in self.W:
                reg = np.dot(self.Phi.T, w)
                plt.plot(x, reg, color='black', label=legend, linewidth=linewidth_data, alpha=0.5)
        # the dataset to train to model
        if b_dataset:
            for y in self.Y:
                plt.plot(x, y, color='gray', label=legend, linewidth=linewidth_data, alpha=0.5)
    def plot_nUpdated(self, legend='', color='g', via_show=True, alpha_std=0.4, mean_line_width=3):
        """
        plot the n-dimension updated distribution, valid from NDProMP or IProMP
        """
        if self.meanW_nUpdated is None:
            print "there is no updated distribution from NDProMP or IProMP"
            return
        x = self.x
        mean0 = np.dot(self.Phi.T, self.meanW_nUpdated)
        std0 = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW_nUpdated, self.Phi))))
        plt.plot(x, mean0, linestyle='--', color=color, label=legend, linewidth=mean_line_width)
        plt.fill_between(x, mean0-std0, mean0+std0, color=color, alpha=alpha_std)
        # option to show the via point
        if via_show:
            for viapoint_id, viapoint in enumerate(self.viapoints):
                plt.plot(viapoint['t'], viapoint['obsy'], marker="o", markersize=10, color=color)
                plt.errorbar(viapoint['t'], viapoint['obsy'], yerr=self.sigmay, fmt="o")

    def plot_uUpdated(self, legend='', color='g'):
        """
        :param legend:
        :param color:
        :param via_show:
        :return:
        """
        x = self.x
        mean = np.dot(self.Phi.T, self.meanW_uUpdated)
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW_uUpdated, self.Phi))))
        plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
        plt.plot(x, mean, '--',color=color, label=legend, linewidth=5)

    def plot_uViapoints(self):
        '''
        :return:
        '''
        for viapoint_id, viapoint in enumerate(self.viapoints):
            if viapoint_id == 0:
                plt.plot(viapoint['t'], viapoint['obsy'], marker="o", markersize=15, color='red', label='via points')
            else:
                plt.plot(viapoint['t'], viapoint['obsy'], marker="o", markersize=15, color='red')
            # plt.errorbar(viapoint['t'], viapoint['obsy'], yerr=self.sigmay, fmt='o', markersize=15)




class NDProMP(object):
    """
    n-dimensional ProMP
    """
    def __init__(self, num_joints, num_basis=11, sigma_basis=0.05, num_samples=101, sigmay=None):
        """
        :param num_joints:
        :param num_basis:
        :param sigma_basis:
        :param num_samples:
        :param sigmay:
        """
        if num_joints < 1:
            raise ValueError("You must declare at least 1 joint in a NDProMP")
        self.num_joints = num_joints
        self.promps = [ProMP(num_basis, sigma_basis, num_samples, sigmay[idx_joint,idx_joint])
                       for idx_joint in range(num_joints)]

        self.W_full = np.array([])  # the weight for each demonstration
        self.meanW_full = np.array([])
        self.covW_full = np.array([])

        self.meanW_full_updated = np.array([])  # the updated weight distribution
        self.covW_full_updated = np.array([])

        self.viapoints = []
        self.sigmay = sigmay  # the measurement noise cov mat

    def obs_mat(self, t):
        """
        Get the obs mat with missing observations
        :param t: the specific time
        :return: the observation mat
        """
        bf = np.exp(-.5 * (np.array(map(lambda x: x - self.C,
                                       np.tile(t, (self.num_basis, 1)).T)).T**2 / (self.sigma_basis**2)))
        bf_full = np.array([]).reshape(0,0)
        # construct the obs mat
        for idx_obs in range(self.num_joints):
            bf_full = scipy.linalg.block_diag(bf_full, bf.T)
        return bf_full

    def add_demonstration(self, demonstration):
        """
        Add a new N-joints demonstration[time][joint] and update the model
        :param demonstration: List of "num_joints" demonstrations
        :return:
        """
        demonstration = np.array(demonstration).T  # Revert the representation for each time for each joint, for each joint for each time
        if len(demonstration) != self.num_joints:
            raise ValueError("The given demonstration has {} joints while num_joints={}".format(len(demonstration), self.num_joints))
        # add demo for each promp
        for joint_demo_idx, joint_demo in enumerate(demonstration):
            self.promps[joint_demo_idx].add_demonstration(joint_demo)
        # construct the weight full samples
        self.W_full = np.array([]).reshape(self.num_demos,0)
        for idx_promp in range(self.num_joints):
            self.W_full = np.hstack([self.W_full, self.promps[idx_promp].W])
        # save the prior distribution
        self.meanW_full = np.mean(self.W_full,0)
        self.meanW_full = self.meanW_full.reshape([self.num_joints*self.num_basis, 1])
        self.covW_full = np.cov(self.W_full.T) if self.num_demos > 1 else None
        self.covW_full = self.covW_full.reshape([self.num_joints*self.num_basis, self.num_joints*self.num_basis]) if self.num_demos > 1 else None

    @property
    def x(self):
        return self.promps[0].x

    @property
    def num_samples(self):
        return self.promps[0].num_samples

    @property
    def num_basis(self):
        return self.promps[0].num_basis

    @property
    def num_demos(self):
        return self.promps[0].num_demos

    @property
    def num_points(self):
        return self.promps[0].num_points

    @property
    def num_viapoints(self):
        return len(self.viapoints)

    @property
    def C(self):
        return self.promps[0].C

    @property
    def sigma_basis(self):
        return self.promps[0].sigma_basis

    @property
    def Phi(self):
        return self.promps[0].Phi

    @property
    def goal_bounds(self):
        return [joint.goal_bounds for joint in self.promps]

    @property
    def goal_means(self):
        return [joint.goal_mean for joint in self.promps]

    def get_bounds(self, t):
        """
        Return the bounds of all joints at time t
        :param t: 0 <= t <= 1
        :return: [(lower boundary joints 0, upper boundary joints 0), (lower boundary joint 1), upper)...]
        """
        return [joint.get_bounds(t) for joint in self.promps]

    def get_means(self, t):
        """
        Return the mean of all joints at time t
        :param t: 0 <= t <= 1
        :return: [mean joint 1, mean joint 2, ...]
        """
        return [joint.get_mean(t) for joint in self.promps]

    def get_stds(self):
        """
        Return the standard deviation of all joints
        :param t: 0 <= t <= 1
        :return: [std joint 1, std joint 2, ...]
        """
        return [joint.get_std() for joint in self.promps]

    def clear_viapoints(self):
        for promp in self.promps:
            promp.clear_viapoints()

    def add_viapoint(self, t, obsys):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: time stamp of observation
        :param obsys: list of observations obys[joint] for each joint
        :return:
        """
        if len(obsys) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(obsys), self.num_joints))
        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo])
        self.viapoints.append({'t': t, 'obsy': obsys})

    def param_update(self, unit_update):
        """
        :param unit_update:
        :return:
        """
        new_meanW_full = self.meanW_full
        new_covW_full = self.covW_full
        for viapoint in self.viapoints:
            bf_full = self.obs_mat(viapoint['t'])
            # the observation of specific time
            y_observed = viapoint['obsy'].reshape([self.num_joints, 1])
            # update the distribution
            aux = self.sigmay + np.dot(bf_full, np.dot(new_covW_full, bf_full.T))
            K = np.dot(np.dot(new_covW_full, bf_full.T), np.linalg.inv(aux))
            new_meanW_full = new_meanW_full + np.dot(K, y_observed - np.dot(bf_full, new_meanW_full))
            new_covW_full = new_covW_full - np.dot(K, np.dot(bf_full, new_covW_full))

        # save the updated distribution for ndpromp
        self.meanW_full_updated = new_meanW_full
        self.covW_full_updated = new_covW_full

        # save the updated distribution for each promp
        if unit_update:
            for i in range(self.num_joints):
                self.promps[i].meanW_nUpdated = new_meanW_full.reshape([self.num_joints,self.num_basis]).T[:,i]
                self.promps[i].sigmaW_nUpdated = new_covW_full[i*self.num_basis:(1+i)*self.num_basis, i*self.num_basis:(i+1)*self.num_basis]

    def gen_nTrajectory(self, randomness=1e-10):
        """
        :param randomness:
        :return:
        """
        new_meanW_full = self.meanW_full_updated
        trajectory = np.dot(self.Phi.T, new_meanW_full.reshape([self.num_joints, self.num_basis]).T)
        return trajectory


class IProMP(NDProMP):
    """
    (n)-dimensional Interaction ProMP, derived from NDProMP
    """
    def __init__(self, num_joints=28, num_obs_joints=None, num_basis=11, sigma_basis=0.05,
                 num_samples=101, sigmay=None, min_max_scaler=None, num_alpha_candidate=10,method="promp"):
        """
        construct function, call NDProMP construct function and define the member variables
        :param num_joints:
        :param num_obs_joints:
        :param num_basis:
        :param sigma_basis:
        :param num_samples:
        :param sigmay:
        :param min_max_scaler:
        :param num_alpha_candidate:
        """
        # compute the obs noise after preprocessing
        noise_cov_full = sigmay
        NDProMP.__init__(self, num_joints=num_joints, num_basis=num_basis,
                         sigma_basis=sigma_basis, num_samples=num_samples, sigmay=noise_cov_full)
        
        self.method = method
        self.viapoints = []
        self.num_obs_joints = num_obs_joints    # the observed joints number

        # the scaling factor alpha
        self.alpha = []
        self.mean_alpha = []
        self.std_alpha = []

        # the preprocessing
        self.min_max_scaler = min_max_scaler

        # the fit alpha
        self.alpha_fit = None

        # num_alpha_candidate
        self.num_alpha_candidate = num_alpha_candidate

    def set_alpha(self, alpha):
        """
        set the alpha for this model
        :param alpha:
        :return:
        """
        self.alpha_fit = alpha
        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].alpha_fit = alpha

    def obs_mat(self, t):
        """
        Get the obs mat with missing observations
        :param t: the specific time
        :return: the observation mat
        """
        bf = np.exp(-.5 * (np.array(map(lambda x: x - self.C,
                      np.tile(t, (self.num_basis, 1)).T)).T**2 / (self.sigma_basis ** 2)))
        zero_entry = np.zeros([1, self.num_basis])
        bf_full = np.array([]).reshape(0,0)
        if self.method == "ipromp":
            # construct the obs mat
            for idx_obs in range(self.num_obs_joints):
                bf_full = scipy.linalg.block_diag(bf_full, bf.T)
            for idx_non_obs in range(self.num_joints - self.num_obs_joints):
                bf_full = scipy.linalg.block_diag(bf_full, zero_entry)
        elif self.method == "promp":
            # construct the obs mat
            for idx_obs in range(self.num_obs_joints):
                bf_full = scipy.linalg.block_diag(bf_full, bf.T)
        return bf_full

    def add_viapoint(self, t, obsys):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: List of observations obys[joint] for each joint
        :return:
        """
        if len(obsys) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(obsys), self.num_joints))

        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo])
        self.viapoints.append({'t': t, 'obsy': obsys})

    def param_update(self, unit_update,dynamic_update=None):
        """
        updated the mean and sigma of w by the via points
        :param unit_update: the bool option for
        :return:
        """
        if dynamic_update == None:
            new_meanW_full = self.meanW_full
            new_covW_full = self.covW_full
        else:
            new_meanW_full = self.meanW_full_updated
            new_covW_full = self.covW_full_updated
            
        for viapoint in self.viapoints:
            bf_full = self.obs_mat(viapoint['t'])
            # the observation of specific time
            y_observed = viapoint['obsy'].reshape([self.num_joints, 1])
            # update the distribution
            aux = self.sigmay + np.dot(bf_full, np.dot(new_covW_full, bf_full.T))
            K = np.dot(np.dot(new_covW_full, bf_full.T), np.linalg.inv(aux))
            new_meanW_full = new_meanW_full + np.dot(K, y_observed - np.dot(bf_full, new_meanW_full))
            new_covW_full = new_covW_full - np.dot(K, np.dot(bf_full, new_covW_full))

        # save the updated distribution for ipromp
        self.meanW_full_updated = new_meanW_full
        self.covW_full_updated = new_covW_full

        # save the updated distribution for each promp
        if unit_update:
            for i in range(self.num_joints):
                self.promps[i].meanW_nUpdated = new_meanW_full.reshape([self.num_joints, self.num_basis]).T[:, i]
                self.promps[i].sigmaW_nUpdated = new_covW_full[i * self.num_basis:(1 + i) * self.num_basis,
                                             i * self.num_basis:(i + 1) * self.num_basis]


    def gen_nTrajectory(self, randomness=1e-10):
        """
        Add a viapoint i.e. an observation at a specific time
        :param randomness: the measurement noise
        :return: the mean of predictive distribution
        """
        new_meanW_full = self.meanW_full_updated
        trajectory = np.dot(self.Phi.T, new_meanW_full.reshape([self.num_joints, self.num_basis]).T)
        return trajectory

    def prob_obs(self):
        """
        compute the pdf of observation sets
        :return: the total joint probability
        """
        prob_full = 0.0
        for viapoint in self.viapoints:
            bf_full = self.obs_mat(viapoint['t'])
            # the y mean and cov
            mean_t = np.dot(bf_full, self.meanW_full)[:,0]
            cov_t = np.dot(bf_full, np.dot(self.covW_full, bf_full.T)) + self.sigmay
            prob = mvn.pdf(viapoint['obsy'], mean_t, cov_t, allow_singular=True)
            log_prob = math.log(prob) if prob != 0.0 else -np.inf
            prob_full = prob_full + log_prob
        return prob_full

    def add_alpha(self, alpha):
        """
        Add a phase to the trajectory
        :param alpha:
        :return:
        """
        self.alpha.append(alpha)
        self.mean_alpha = np.mean(self.alpha)
        self.std_alpha = np.std(self.alpha)

    def alpha_candidate(self, num=None):
        """
        compute the alpha candidate by unit sampling
        :param num: the num of alpha candidate
        :return: the list of alpha candidate
        """
        if num is None:
            num = self.num_alpha_candidate
        self.num_alpha_candidate = num

        # alpha_candidate = np.linspace(self.mean_alpha-2*self.std_alpha, self.mean_alpha+2*self.std_alpha, num)
        alpha_candidate = np.random.normal(self.mean_alpha, self.std_alpha, num)
        candidate_pdf = stats.norm.pdf(alpha_candidate, self.mean_alpha, self.std_alpha)
        alpha_gen = []
        for idx_candidate in range(self.num_alpha_candidate):
            alpha_gen.append({'candidate': alpha_candidate[idx_candidate], 'prob': candidate_pdf[idx_candidate]})
        return alpha_gen

    def log_ll_alpha(self, alpha_candidate, obs, time):
        """
        compute the alpha candidate log likelihood
        :param alpha_candidate: the alpha candidate
        :param obs: the observations
        :param time: the timestamp
        :return: the alpha candidate log likelihood
        """
        prob_full = 0.0
        for obs_idx in range(len(time)):
            bf_full = self.obs_mat(time[obs_idx]/alpha_candidate)
            mean_t = np.dot(bf_full, self.meanW_full)[:, 0]
            cov_t = np.dot(np.dot(bf_full, self.covW_full), bf_full.T) + self.sigmay
            cov_t = np.dot(np.dot(bf_full, self.covW_full), bf_full.T)

            prob = mvn.pdf(obs[obs_idx], mean_t, cov_t, allow_singular=True)
            log_prob = math.log(prob) if prob != 0.0 else -np.inf
            prob_full = prob_full + log_prob

            # new_mean_t = mean_t[0: self.num_obs_joints]
            # new_cov_t = cov_t[0:self.num_obs_joints, 0:self.num_obs_joints]

            # prob = mvn.pdf(obs[obs_idx][0: self.num_obs_joints], new_mean_t, new_cov_t)
            # log_prob = math.log(prob) if prob != 0.0 else -np.inf
            # prob_full = prob_full + log_prob

        return prob_full

    def estimate_alpha(self, alpha_candidates, obs, times):
        """
        compute the MAP
        :param alpha_candidates: the alpha candidate
        :param obs: the observations
        :param times: the timestamp
        :return: MAP for alpha
        """
        pp_list = []
        for alpha_candidate in alpha_candidates:
            lh = self.log_ll_alpha(alpha_candidate['candidate'], obs, times)
            pp = math.log(alpha_candidate['prob']) + lh
            pp_list.append(pp)
        id_max = np.argmax(pp_list)
        return id_max

    def gen_real_traj(self, alpha):
        """
        generate the predicted traj, which resume the real length
        :param alpha: the best fit alpha
        :return: time_traj, traj
        """
        traj = self.gen_nTrajectory()
        time_traj = np.linspace(0.0, alpha, self.num_points)
        return time_traj, traj
