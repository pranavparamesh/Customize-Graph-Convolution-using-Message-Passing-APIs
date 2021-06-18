# Customize-Graph-Convolution-using-Message-Passing-APIs

In previous sessions, we have learned using the built-in graph convolution modules to build a multi-layer graph neural network. However, sometimes one desires to invent a new way of aggregating neighbor information. DGL's message passing APIs are designed for this scenario.

In this tutorial, you will learn:

What is under the hood of the nn.SAGEConv module in DGL?
DGL's message passing APIs.
Design a new graph convolution module.



# Message passing and GNNs

DGL follows the message passing paradigm inspired by the Message Passing Neural Network proposed by Gilmer et al. Essentially, they found many GNN models can fit into the framework where DGL calls the message function and the reduce function. Note that here can represent any function and is not necessarily a summation.
For example, the GraphSAGE model has the following mathematical forms
You can see that message passing is directional: the message sent from one node to other node is not necessarily the same as the other message sent from node to node in the opposite direction.

DGL graphs provide two members srcdata and dstdata for the purpose of message passing. You first put the input node features in srcdata. After you perform message passing, you can retrieve the result of message passing from dstdata.
