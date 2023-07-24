# SHTAU

## Environment
* python=3.7.0
* tensorflow=1.15.0
* numpy=1.21.6
* scipy=1.7.3


## Usage
Please unzip the datasets first. Also you need to create the `Models/` directory. The following command lines start training and testing on the three datasets, respectively, which also specify the hyperparameter settings for the reported results in the paper. Training and testing logs for trained models are contained in the `History/` directory.
```
python .\labcode_hop.py --data yelp --reg 1e-2 --ssl_reg 1e-5 --mult 1e2 --edgeSampRate 0.1 --gamma 1.5
```
