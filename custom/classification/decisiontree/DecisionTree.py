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

    def __init__(self, criterion_name, min_examples_per_split=200):
        self.min_examples_per_split = min_examples_per_split
        self.criterion = _get_criterion(criterion_name)
        self._criterion_name = criterion_name
        self._tree = BinaryTree()

    def fit(self, training_x, training_y):
        self._learn_tree(training_x, training_y, self._tree)

    def _less_examples_than_min(self, examples):
        return len(examples) < self.min_examples_per_split

    def _pure_node(self, labels):
        return len(labels) == 1

    def _learn_tree(self, training_x, training_y, current_node: BinaryTree):
        """
        Learns the decision tree wrt the current examples
        This methods build a tree finding at each step the best feature and feature value to maximize the chosen criteria
        The tree that it builds is binary, so that for each split, the examples that match the condition are brought to the left child and those
        who not are brought to the right child

        It generates a leaf node instead of a split if:
        - there are less examples than the minimum
        - the examples are already all of the same class (pure node)



        :param training_x:
        :param training_y:
        :param current_node:
        :return:
        """

        labels, current_class_distribution = get_distribution(training_y)
        current_node.samples = len(training_x)
        current_node.samples_distribution = current_class_distribution

        if self._less_examples_than_min(training_x):
            current_node.set_value(labels[np.argmax(current_class_distribution)])
            current_node.set_node_type(NodeType.leaf)
            return

        if self._pure_node(labels) == 1:
            # pure class
            current_node.set_value(labels[0])
            current_node.set_node_type(NodeType.leaf)
            return

        best_information_gain, best_split_feature_index, best_split_feature_value = self.find_best_split_feature(
            training_x, training_y
        )

        left_x, left_y, right_x, right_y = self._split_examples(
            training_x, training_y, best_split_feature_index, best_split_feature_value
        )

        current_node.set_value((best_split_feature_index, best_split_feature_value))
        current_node.set_node_type(NodeType.split)
        current_node.information_gain = best_information_gain

        left_subtree = current_node.set_left(None)
        left_subtree.node_index = (current_node.node_index + 1)
        self._learn_tree(np.array(left_x), np.array(left_y), left_subtree)

        right_subtree = current_node.set_right(None)
        right_subtree.node_index = (left_subtree.node_index + left_subtree.children_number() + 1)
        self._learn_tree(np.array(right_x), np.array(right_y), right_subtree)


    def find_best_split_feature(self, training_x, training_y):
        """
        Find the best split to generate wrt the criterion

        :param training_x:
        :param training_y:
        :return: information gain, feature index and feature value for the best split
        """

        best_information_gain = 0
        best_split_feature_index = None
        best_split_feature_value = None

        for feature_idx, feature_values in enumerate(training_x.T):
            unique_values = np.unique(feature_values)

            if len(unique_values) == 1:
                # nothing to split here
                continue

            for value in unique_values:
                split_information_gain = self.compute_get_information_gain_for_split_on_value(training_x, training_y,
                                                                                              feature_idx, value)

                if split_information_gain > best_information_gain:
                    best_information_gain = split_information_gain
                    best_split_feature_index = feature_idx
                    best_split_feature_value = value

        return best_information_gain, best_split_feature_index, best_split_feature_value

    def compute_get_information_gain_for_split_on_value(self, training_x, training_y, feature_idx, value):
        """
        Computes the information gain for a specific split (on a pair feature index/feature value)

        :param training_x:
        :param training_y:
        :param feature_idx:
        :param value:
        :return: the information gain
        """

        total_examples = len(training_x)

        left_x, left_y, right_x, right_y = self._split_examples(training_x, training_y, feature_idx, value)

        # left
        _, left_class_distribution = get_distribution(left_y)
        left_class_criterion_value = self.criterion(left_class_distribution)
        left_class_weight = len(left_x) / total_examples

        # right
        _, right_class_distribution = get_distribution(right_y)
        right_class_criterion_value = self.criterion(right_class_distribution)
        right_class_weight = len(right_x) / total_examples

        labels, current_class_distribution = get_distribution(training_y)
        current_criterion_value = self.criterion(current_class_distribution)

        information_gain = current_criterion_value - \
                           (left_class_weight * left_class_criterion_value) - \
                           (right_class_weight * right_class_criterion_value)

        return information_gain

    def _split_examples(self, training_x, training_y, feature_idx, value):
        left_x = []
        left_y = []

        right_x = []
        right_y = []

        for example in zip(training_x, training_y):
            if example[0][feature_idx] == value:
                left_x.append(example[0])
                left_y.append(example[1])
            else:
                right_x.append(example[0])
                right_y.append(example[1])

        return left_x, left_y, right_x, right_y

    def predict(self, instance):

        current_node = self._tree

        while current_node._node_type == NodeType.split:
            feature_index = current_node._value[0]
            feature_value = current_node._value[1]

            if instance[feature_index] == feature_value:
                current_node = current_node._left
            else:
                current_node = current_node._right


        return current_node._value