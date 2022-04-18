from abc import ABC, abstractmethod


class IClassifier(ABC):

    @abstractmethod
    def fit(self, training_x, training_y):
        pass

    @abstractmethod
    def predict(self, instance):
        pass
