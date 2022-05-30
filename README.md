# Block_NAS

NAS as a combination of blocks


## Connecting to a GPU and loading Pytorch: 

- Connect to Pandora 
```
$ ssh pandora
```
- Load Pytorch 
```
$ module load anaconda
$ conda activate pytorch-1.10.1
```
- Choose 1 available GPU

```
$ nvidia-smi
$ export CUDA_VISIBLE_DEVICES=id_of_available_gpu
```

## Requirements: 

after loading pytorch

```
$ pip install thop
```
 
## Test the blackbox


```
$ python pytorch_bb.py CIFAR10 x.txt 
```

## NOMAD


```
$ module load gcc
$ nomad -i
$ nohup nomad parameter_file.txt &
```
