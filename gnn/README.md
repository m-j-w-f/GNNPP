# 🧪 Graph Neural Network Experiments

Files in this folder are used to perform experiments with GNNs to evaluate which architectures are promising.

## 🚧 Current TODOs

- [ ] How to construct the Graph: Test different Methods
  - [ ] Use a per node model that operates on a small subgraph of nodes and predicts temperatures for one model only
- [ ] What about Stations that are far outside (isolated Nodes)?
- [x] Modularize -> use multiple torch.modules in a Sequential model
- [x] Early stopping should save model checkpoint
- [ ] gin config
- [ ] [Neural Tangent Kernel](https://en.wikipedia.org/wiki/Neural_tangent_kernel)
- [ ] Attention weights mitteln und anschauen (Kanten in Graph einfärben)
- [ ] W&B hinzufügen
- [ ] Check out GNN explainer
- [ ] Hyperparameter Optimization
  - [ ] Test different Layers and Architectures
  - [x] Embed Node ID (wie in paper von Sebastian)
  - [x] ResNet (Recidual Layer)
  - [x] Softplus for sigma
  - [ ] Edge Weights (Distances)
