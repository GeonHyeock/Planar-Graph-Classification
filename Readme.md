# Planar Graph Classification : Graph Neural Network for Graphs

## Poster
- KSIAM에서 주관하는 2024 춘계학술대회 포스터 내용입니다.

<img width="100%" src="./Poster.jpg"/>

## Command

### ⭐️Data Sampling
~~~
1. data폴더에 planar_embedding1000000.pg 저장
   (http://www.inf.udec.cl/~jfuentess/datasets/graphs.php)

2. python src/DataSampling.py --min_node {m} --max_node {M} --N {N} --LargePlanarGraph {PG}
~~~

### ⭐️Model Train
- [model list](lightning-hydra-template/configs/model)
~~~
1. cd lightning-hydra-template
2. python src/train.py model={model name}
   (option) logger=wandb logger.wandb.name={model log name}
~~~

## Inference Time Test
~~~
1. cd lightning-hydra-template
2. python src/test.py model={model name}
~~~
