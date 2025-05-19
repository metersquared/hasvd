# %%
import utils
import matplotlib.pyplot as plt
import networkx as nx

# %%
nxG, pos, leaf_nodes = utils.random_kary_tree(4, 6)

# options
optionsLeaf = {"node_color": "white", "edgecolors": "black"}
optionsParent = {
    "node_color": "black",
    "edgecolors": "black",
}

root_node = [r"$\rho$"]


plt.axis("off")
plt.rcParams["text.usetex"] = True
# nx.draw(nxG, pos)
nx.draw_networkx_nodes(nxG, pos, nodelist=leaf_nodes, **optionsLeaf)
nx.draw_networkx_nodes(
    nxG,
    pos,
    nodelist=[node for node in nxG.nodes() if node not in leaf_nodes],
    **optionsParent,
)
nx.draw_networkx_labels(
    nxG,
    pos,
    {n: n for n in root_node if n in pos},
    font_color="white",
)
nx.draw_networkx_edges(nxG, pos, edgelist=nxG.edges())

# %%
