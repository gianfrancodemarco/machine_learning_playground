from __future__ import annotations

from enum import Enum


class NodeType(Enum):
    split = "split"
    leaf = "leaf"


class BinaryTree:

    def __init__(self):
        self._left = None
        self._right = None
        self._value = None
        self._node_type : NodeType | None = None

    def set_left(self, value):
        """
        Set a value for the left child.
        If it doesn't exist, create a BinaryTree into the left child and then sets its value

        :param value: The value to be set into the left child
        :return: The left child
        """
        if not self._left:
            self._left = BinaryTree()

        self._left.set_value(value)
        return self._left

    def set_right(self, value):
        """
        Set a value for the right child.
        If it doesn't exist, create a BinaryTree into the right child and then sets its value

        :param value: The value to be set into the right child
        :return: The right child
        """

        if not self._right:
            self._right = BinaryTree()

        self._right.set_value(value)
        return self._right

    def set_value(self, value):
        self._value = value

    def set_node_type(self, node_type: NodeType):
        self._node_type = node_type
