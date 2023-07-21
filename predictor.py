import torch.cuda
"""
abstract class for performance predictor
reference: https://github.com/automl/NASLib/
"""

class Predictor:
    def __init__(self, labels=None, device=None, ss_type=None, encoding_type=None):
        self.ss_type = ss_type
        self.encoding_type = encoding_type
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.labels = labels.float().to(self.device)
        
    def set_ss_type(self, ss_type):
        """
        Set search space type(e.g. NAS-Bench-201)
        :param ss_type:
        :return:
        """
        self.ss_type = ss_type

    def pre_process(self):
        """
        This is called at the start of the NAS algorithm,
        before any architectures have been queried
        """
        pass

    def pre_compute(self, xtrain, xtest, unlabeled=None):
        """
        This method is used to make batch predictions
        more efficient. Perform a computation on the train/test
        set once (e.g., calculate the Jacobian covariance)
        and then use it for all train_sizes.
        """
        pass

    def pre_train(self, xtrain, ytrain, info=None):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass

    def fit(self, xtrain, ytrain, info=None):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass

    def query(self, xtest, info):
        """
        This can be called any number of times during the NAS algorithm.
        inputs: list of architectures,
                info about the architectures (e.g., training data up to 20 epochs)
        output: predictions for the architectures
        """
        pass

    def proxy_fit(self, xtrain, ytrain, info=None):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass