# A PyTorch Implementation of Graph Convolutional Network for Edge Prediction

## Summary
This repository provides a PyTorch implementation of Stanford's tutorial on GCN for edge prediction found at: [Graph Convolutional Prediction of Protein Interactions in Yeast](http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html)

## Requirements
<ul>
<li>numpy~=1.20.2</li>
<li>scipy~=1.6.3</li>
<li>networkx~=2.5.1</li>
<li>torch~=1.8.1</li>
<li>sklearn~=0.0</li>
<li>scikit-learn~=0.24.2</li>
</ul>

## Usage
The train.py script trains the model for a specified number of epochs and performs inference at the end of training on the test set. The dataset used for this experiment is provided in the /data folder due its manageable size. 

To run the script use the following command in the /src directory:

```sh
python train.py --input_path <path to dataset> --epochs <number of training epochs> --learning_rate <well, LR :)> --model_path <path to save the trained model to> --hidden_dim <hidden layer dimension> --output_dim <graph embedding dimension>
```
