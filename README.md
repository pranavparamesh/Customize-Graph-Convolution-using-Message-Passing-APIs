# Customize-Graph-Convolution-using-Message-Passing-APIs

In previous sessions, we have learned using the built-in graph convolution modules to build a multi-layer graph neural network. However, sometimes one desires to invent a new way of aggregating neighbor information. DGL's message passing APIs are designed for this scenario.

In this tutorial, you will learn:

What is under the hood of the nn.SAGEConv module in DGL?
DGL's message passing APIs.
Design a new graph convolution module.



Message passing and GNNs
DGL follows the message passing paradigm inspired by the Message Passing Neural Network proposed by Gilmer et al. Essentially, they found many GNN models can fit into the following framework:

$$
m_{u\sim v}^{(l)} = M^{(l)}\left(h_v^{(l-1)}, h_u^{(l-1)}, e_{u\sim v}^{(l-1)}\right)
$$$$
m_{v}^{(l)} = \sum_{u\in\mathcal{N}(v)}m_{u\sim v}^{(l)}
$$$$
h_v^{(l)} = U^{(l)}\left(h_v^{(l-1)}, m_v^{(l)}\right)
$$
where DGL calls $M^{(l)}$ the message function and $\sum$ the reduce function. Note that $\sum$ here can represent any function and is not necessarily a summation.

For example, the GraphSAGE model has the following mathematical form:

$$
h_{\mathcal{N}(v)}^k\leftarrow \text{Average}\{h_u^{k-1},\forall u\in\mathcal{N}(v)\}
$$$$
h_v^k\leftarrow \text{ReLU}\left(W^k\cdot \text{CONCAT}(h_v^{k-1}, h_{\mathcal{N}(v)}^k) \right)
$$
You can see that message passing is directional: the message sent from one node $u$ to other node $v$ is not necessarily the same as the other message sent from node $v$ to node $u$ in the opposite direction.

DGL graphs provide two members srcdata and dstdata for the purpose of message passing. You first put the input node features in srcdata. After you perform message passing, you can retrieve the result of message passing from dstdata.
