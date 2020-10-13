import pickle

class Mahalanobis(object):

    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        if len(pose.shape) == 1:
            return (pose[self.prefix:]-self.mean).reshape(1, -1).dot(self.prec)
        else:
            return (pose[:, self.prefix:]-self.mean).dot(self.prec)


prior_filename = 'priors/smil_pose_prior.pkl'
prior = pickle.load(open(prior_filename, 'rb'), encoding='latin1')
print('prior')            
print(prior.mean)
