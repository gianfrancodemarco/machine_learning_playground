from custom.classification.interfaces.IClassifier import IClassifier
from custom.math.MathUtils import gini, error, entropy, get_distribution
from logging import getLogger
from BinaryTree import BinaryTree, NodeType
import numpy as np

logger = getLogger(__name__)


def _get_criterion(criterion):
    criterions = {
        'entropy': entropy,
        'gini': gini,
        'error': error
    }

    try:
        return criterions[criterion]
    except KeyError:
        logger.error(f'Invalid criterion: "{criterion}"')
        raise


class DecisionTree(IClassifier):

    def __init__(self, criterion, min_examples_per_split=10):
        self.min_examples_per_split = min_examples_per_split
        self.criterion = _get_criterion(criterion)
        self._tree = BinaryTree()

    def fit(self, training_x, training_y):
        self.learn_tree(training_x, training_y, self._tree)

    def less_examples_than_min(self, examples):
        return len(examples) < self.min_examples_per_split

    def pure_node(self, labels):
        return len(labels) == 1

    def learn_tree(self, training_x, training_y, current_node: BinaryTree):

        labels, current_class_distribution = get_distribution(training_y)

        if self.less_examples_than_min(training_x):
            current_node.set_value(labels[np.argmax(current_class_distribution)])
            current_node.set_node_type(NodeType.leaf)
            return

        if self.pure_node(labels) == 1:
            # pure class
            current_node.set_value(labels[0])
            current_node.set_node_type(NodeType.leaf)
            return

        current_criterion_value = self.criterion(current_class_distribution)
        current_information_gain = 0
        current_best_split_feature_index = None
        current_best_split_feature_value = None
        current_left_examples_x = []
        current_left_examples_y = []
        current_right_examples_x = []
        current_right_examples_y = []

        for feature_idx, feature_values in enumerate(training_x.T):
            unique_values = np.unique(feature_values)

            if len(unique_values) == 1:
                # nothing to split here
                continue

            for value in unique_values:
                total_examples = len(training_x)

                left_examples = [(training_x, training_y) for (training_x, training_y) in zip(training_x, training_y) if
                                 training_x[feature_idx] == value]
                right_examples = [(training_x, training_y) for (training_x, training_y) in zip(training_x, training_y)
                                  if training_x[feature_idx] != value]

                # left
                left_x = [training_x for (training_x, _) in left_examples]
                left_y = [training_y for (_, training_y) in left_examples]
                _, left_class_distribution = get_distribution(left_y)
                left_class_criterion_value = self.criterion(left_class_distribution)
                left_class_weight = len(left_x) / total_examples

                # right
                right_x = [training_x for (training_x, _) in right_examples]
                right_y = [training_y for (_, training_y) in right_examples]
                _, right_class_distribution = get_distribution(right_y)
                right_class_criterion_value = self.criterion(right_class_distribution)
                right_class_weight = len(right_x) / total_examples

                information_gain = current_criterion_value - left_class_weight * left_class_criterion_value - right_class_weight * right_class_criterion_value

                if information_gain > current_information_gain:
                    current_information_gain = information_gain
                    current_best_split_feature_index = feature_idx
                    current_best_split_feature_value = value
                    current_left_examples_x = left_x
                    current_left_examples_y = left_y
                    current_right_examples_x = right_x
                    current_right_examples_y = right_y

        current_node.set_value((current_best_split_feature_index, current_best_split_feature_value))
        current_node.set_node_type(NodeType.split)
        self.learn_tree(np.array(current_left_examples_x), np.array(current_left_examples_y),
                        current_node.set_left(None))
        self.learn_tree(np.array(current_right_examples_x), np.array(current_right_examples_y),
                        current_node.set_right(None))

    def predict(self, instance):
        pass
