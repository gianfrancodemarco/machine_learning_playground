from numpy.distutils.lib2def import output_def

from custom.classification.decisiontree.BinaryTree import BinaryTree, NodeType

class BinaryTreeToDotFormatConverter:

    output_template = """
digraph Tree {
node [shape=box, fontname="helvetica"] ;
edge [fontname="helvetica"] ;
    """

    node_template = """{number} [label="{label}\\n{criterion} = {criterion_value}\\nsamples = {samples}\\ndistribution = {distribution}"]; """

    edge_template = """{number_from} -> {number_to} [labeldistance = 2.5, labelangle = 45, headlabel = {headlabel}];"""

    @classmethod
    def dump_tree(cls, tree: BinaryTree):

        output = cls.output_template
        output = cls._dump_tree(output, tree)
        output += '\n}'

        with open("output.dot", "w") as f:
            f.write(output)

    @classmethod
    def _dump_tree(cls, output, tree):
        output = cls.add_node(output, tree)

        if tree._left:
            output = cls._dump_tree(output, tree._left)
            output = cls.add_edge(output, tree.node_index, tree._left, "True")

        if tree._right:
            output = cls._dump_tree(output, tree._right)
            output = cls.add_edge(output, tree.node_index, tree._right, "False")

        return output

    @classmethod
    def add_node(cls, output, tree):

        node = cls.node_template

        node = node.replace("{samples}", str(tree.samples))
        node = node.replace("{distribution}", str([round(el, 2) for el in tree.samples_distribution]))


        if tree.information_gain:
            node = node.replace("{criterion}", "Information gain")
            node = node.replace("{criterion_value}", str(round(tree.information_gain, 2)))
        else:
            node = node.replace("\\n{criterion} = {criterion_value}", "")

        if tree._node_type == NodeType.leaf:
            node = node.replace("{label}", f"Predicted label: {tree._value}")

        elif tree._node_type == NodeType.split:
            node = node.replace("{label}", f"Feature n.{tree._value[0]} = {tree._value[1]}")

        node = node.replace("{number}", str(tree.node_index))
        output += f"\n{node}"

        return output

    @classmethod
    def add_edge(cls, output, from_index, to_node: BinaryTree, headlabel):

        edge = cls.edge_template

        edge = edge.replace("{headlabel}", headlabel)
        edge = edge.replace("{number_from}", str(from_index))
        edge = edge.replace("{number_to}", str(to_node.node_index))

        output += f"\n{edge}"

        return output


