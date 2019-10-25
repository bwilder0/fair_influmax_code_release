# Overview
This repository contains code for the paper:

Alan Tsang*, Bryan Wilder*, Eric Rice, Milind Tambe, Yair Zick. Group-Fairness in Influence Maximization. IJCAI 2019. [[arXiv]](https://arxiv.org/abs/1903.00967). 
```
@inproceedings{tsang2019group,
  title={Group-Fairness in Influence Maximization},
  author={Tsang, Alan and Wilder, Bryan and Rice, Eric and Tambe, Milind and Zick, Yair},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2019}
}
```

Run check_fairness.py in order to compare the performance of the three algorithms in the paper on the synthetic Antelope Valley networks (included in the networks folder). check_fairness_intersectional.py runs the experiment where nodes have multiple group memberships.

# Dependencies
* Optionally, you can use [Gurobi](http://www.gurobi.com/) to solve the inner maxmin LP instead of the mirror descent algorithm discussed in the paper. This may be faster and require less tuning for small problems.
