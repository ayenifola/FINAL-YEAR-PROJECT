Anomaly Detection in IOT
----

## Setup
We use [Pixi](https://pixi.sh/dev/) for environment management. Head over to their documentation to install it, if you haven't.

Then install the python environment by running the following project in the project's root directory.
```shell
pixi install
```

## Running Experiments (Examples)

### Feature Selection using Genetic Algorithm
Refer to the function `feature_selection` in [main.py](main.py) to see relevant arguments.

To perform feature selection for the model `ModernMLP` and the `MQTTSet` dataset, use the following command as an example.
```shell
pixi run python ./main.py feature-selection --model-type=mlp --dataset-type=mqttset
```

### Training Model

To train a model, you can do something like this:
```shell
 pixi run python ./main.py train --model-type=mlp --dataset-type=mqttset 
```

### Evaluating Model

If you have trained a `ModernMLP` model on the `MQTTSet` dataset, have a saved checkpoint at `./checkpoints/mlp-epoch=05-val_f1=0.8358.pt`, u can evaluate the model on test dataset split like so.
```shell

 pixi run python ./main.py evaluate ./checkpoints/mlp-epoch=05-val_f1=0.8358.pt --model-type=mlp --dataset-type=mqttset 
```