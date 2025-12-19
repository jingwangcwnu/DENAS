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
