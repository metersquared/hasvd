# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


def random_kary_tree(n, k):
    nxG = nx.Graph()
    node_count = 1
    nxG.add_node(r"$\rho_T$")
    parent_count = 0
    leaf_nodes = [r"$\rho_T$"]
    while parent_count < n:
        for leaf_node in leaf_nodes:
            child_count = rng.choice(np.append(0, np.arange(2, k + 1)))
            for i in range(child_count):
                node_count = node_count + 1
                nxG.add_node(node_count)
                nxG.add_edge(node_count, leaf_node)
                leaf_nodes.append(node_count)
            if child_count > 0:
                parent_count = parent_count + 1
                leaf_nodes.remove(leaf_node)
            if parent_count == n:
                break
    return nxG, leaf_nodes


# %%
kary_tree, leaf_nodes = random_kary_tree(3, 5)
# Generate the layout using graphviz_layout
pos = nx.nx_agraph.graphviz_layout(
    kary_tree,
    prog="dot",
    args=f"-Groot={1}",
)

# options
optionsLeaf = {"node_color": "white", "edgecolors": "black"}
optionsParent = {
    "node_color": "black",
    "edgecolors": "black",
}

root_node = [r"$\rho_T$"]


plt.axis("off")
plt.rcParams["text.usetex"] = True
# nx.draw(kary_tree, pos)
nx.draw_networkx_nodes(kary_tree, pos, nodelist=leaf_nodes, **optionsLeaf)
nx.draw_networkx_nodes(
    kary_tree,
    pos,
    nodelist=[node for node in kary_tree.nodes() if node not in leaf_nodes],
    **optionsParent,
)
nx.draw_networkx_labels(
    kary_tree,
    pos,
    {n: n for n in root_node if n in pos},
    font_color="white",
)
extendNode = rng.choice(leaf_nodes)
nx.draw_networkx_labels(
    kary_tree,
    pos,
    {extendNode: r"$\alpha$"},
    font_color="white",
)
nx.draw_networkx_edges(kary_tree, pos, edgelist=kary_tree.edges())
plt.savefig("subtree-t.png", transparent=True)
# %%
leaf_nodes.remove(extendNode)
node_count = len(kary_tree.nodes())
child_count = 4
new_edges = []
child_extend = []
for i in range(child_count):
    node_count = node_count + 1
    child_extend.append(node_count)
    kary_tree.add_node(node_count)
    kary_tree.add_edge(node_count, extendNode)
    new_edges.append((extendNode, node_count))
    leaf_nodes.append(node_count)

child_choice = rng.choice(child_extend)
leaf_nodes.remove(child_choice)
for i in range(2):
    node_count = node_count + 1
    kary_tree.add_node(node_count)
    kary_tree.add_edge(node_count, child_choice)
    new_edges.append((child_choice, node_count))
    leaf_nodes.append(node_count)

pos = nx.nx_agraph.graphviz_layout(
    kary_tree,
    prog="dot",
    args=f"-Groot={1}",
)
# %%
optionsExtend = {
    "node_color": "black",
    "edgecolors": "black",
}

plt.axis("off")
nx.draw_networkx_nodes(kary_tree, pos, nodelist=leaf_nodes, **optionsLeaf)
nx.draw_networkx_nodes(kary_tree, pos, nodelist=[extendNode], **optionsExtend)
nx.draw_networkx_nodes(
    kary_tree,
    pos,
    nodelist=[
        node
        for node in kary_tree.nodes()
        if node not in leaf_nodes and node != extendNode
    ],
    **optionsParent,
)
nx.draw_networkx_labels(
    kary_tree,
    pos,
    {n: n for n in root_node if n in pos},
    font_color="white",
)
nx.draw_networkx_labels(
    kary_tree,
    pos,
    {extendNode: r"$\alpha$"},
    font_color="white",
)
print(new_edges)
print([edge for edge in kary_tree.edges() if edge not in new_edges])
nx.draw_networkx_edges(
    kary_tree,
    pos,
    edgelist=[edge for edge in kary_tree.edges() if edge not in new_edges],
)
nx.draw_networkx_edges(kary_tree, pos, edgelist=new_edges)
plt.savefig("tree-example.png", transparent=True)
# %%
