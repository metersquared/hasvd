# %%
import matplotlib.pyplot as plt
from utils import hasvd_Node, draw_nxgraph

tree2 = hasvd_Node(2, 14)

node13 = tree2.add_child(direction=1, tag=13)
node13.add_child(tag=1)
node12 = node13.add_child(direction=2, tag=12)
node12.add_child(tag=2)
node12.add_child(tag=3)
node11 = node12.add_child(direction=1, tag=11)
node11.add_child(tag=4)
node11.add_child(tag=5)
node9 = tree2.add_child(direction=2, tag=9)
node9.add_child(tag=6)
node10 = node9.add_child(direction=1, tag=10)
node10.add_child(tag=8)
node10.add_child(tag=7)

draw_nxgraph(node9)
plt.savefig("submatrix-graph-1.png", transparent=True)

# %%
draw_nxgraph(tree2)
plt.savefig("submatrix-graph-2.png", transparent=True)

# %%
