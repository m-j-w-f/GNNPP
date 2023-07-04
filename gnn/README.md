# 🧪 Graph Neural Network Experiments

Files in this folder are used to perform experiments with GNNs to evaluate which architectures are promising.

## ⚡️Current Best Hyperparameters
  
| Parameter           | ResGnn   | GraphSAGE    | GGNGI        |
|---------------------|----------|--------------|--------------|
| graph generation    | max_dist | nearest_k in | nearest_k in |
| k/max_dist          | 50       | 5            | 5            |
| Embedding Dimension | 10       | 10           | 10           |
| Layers              | 2        | 2            | 2            |
| Hidden Channels     | 512      | 512          | 512          |
| Heads               | 1        | -            | -            |
| Learning Rate       | 0.0026   | 0.002        | 0.002        |
| Batch size          | 8        | 8            | 8            |
| Validation Score    |          | 0.792        | 0.806        |


## 🚧 TODOs

- [ ] Use Learning Rate Finder to find optimal learning rate [Link](https://github.com/davidtvs/pytorch-lr-finder)
- [ ] How to construct the Graph: Test different Methods
  - [x] k_nearest
  - [x] max_dist
  - [ ] Delauny triangulation
- [ ] What about Stations that are far outside (isolated Nodes)?
- [x] Modularize -> use multiple torch.modules in a Sequential model
- [x] Early stopping should save model checkpoint
- [x] Attention weights mitteln und anschauen (Kanten in Graph einfärben)
- [x] W&B hinzufügen
- [ ] Check out GNN explainer
- [ ] Hyperparameter Optimization
  - [ ] Test different Layers and Architectures
  - [x] Embed Node ID (wie in paper von Sebastian)
  - [x] ResNet (Recidual Layer)
  - [x] Softplus for sigma
  - [x] Edge Weights (Distances)

