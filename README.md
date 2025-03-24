# DENAS
Diabetic retinopathy is one of the leading causes of preventable blindness worldwide, and early automatic prediction of diabetic retinopathy has drawn sustained attention. Despite preliminary results made by deep methods and potential progress yielded by neural architecture search, there still exists some limitations including insufficient ability to identify different lesions, susceptibility to local optima and unfair competition between candidate architectures. To this end, this paper proposes differential evolution neural architecture search for automatic prediction of diabetic retinopathy. Firstly, we firstly pre-define search space with multiple operations to solve the challenge of lesion recognition at different scales. Then, we introduce differential gradient evolution strategy to alleviate the dilemma of differentiable neural architecture search getting stuck in local optima. Within it, a series of theoretical analysis is provided to demonstrate the robustness and global convergence. In addition, we design a novel fitness function to reduce unfair competition among candidate architectures. Extensive experiments performed on benchmark datasets demonstrate that our proposed method has achieved average improvements of 26.7%, 24.9%, 30.8%, 45.1%, and 30.5% on Accuracy, Cohen’s Kappa, ROC-AUC, IBA, and F1-score, respectively.. For a detailed description of technical details and experimental results, please refer to our paper:

DENAS: Differential Evolution Neural Architecture Search for Prediction of Diabetic Retinopathy 

## DATASETS

IDRiD, MESSIDOR, EYEPACS, DDR require manual downloading, with corresponding paper [IDRiD](https://doi.org/10.3390/data3030025), [MESSIDOR](https://doi.org/10.5566/IAS.1155), [EYEPACS](https://doi.org/10.1177/193229680900300315), [DDR](https://doi.org/10.1016/j.ins.2019.06.011).

## Prerequisite for server

```
Python = 3.8, PyTorch == 2.0.1, torchvision == 0.15.2
```

## Architecture search

To carry out architecture search using 2nd-order, run 

```
python train_search.py
```

## Architecture evaluation

To evaluate our best cells by training from scratch, run 

```
python train.py 
```

Customized architectures are supported through the ***--arch*** flag once specified in ***genotypes.py***.

## Visualization

Package [graphviz](https://graphviz.readthedocs.io/en/stable/index.html) is required to visualize the learned cells

```
python visualize.py 
```

where SODAS can be replaced by any customized architectures in ***genotypes.py***.
