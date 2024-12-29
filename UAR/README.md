# Universal Authorship Representation of Literary Characters

This folder contain all the code and data necessary to train UAR models on DramaCV. It is based on the official [LUAR repository](https://github.com/LLNL/LUAR), that we slightly modified to match DramaCV data structure. 

## HuggingFace
Our trained UAR model variants are now available on HuggingFace, find the Scene version [here](https://huggingface.co/gasmichel/UAR_scene) and the Play version [here](https://huggingface.co/gasmichel/UAR_Play).

Also find DramaCV directly on huggingface [here](https://huggingface.co/datasets/gasmichel/DramaCV).

## Training

### UAR Play
```bash
python src/main.py --dataset_name drama --data_path data/full_data/ --model_name sentence-transformers/all-distilroberta-v1 --do_learn --validate --evaluate --gpus 1 --experiment_id uar_play --validate_every 1 --batch_size 1 --episode_length 8 --token_max_length 64 --embedding_dim 512
```

### UAR Scene
```bash
python src/main.py --dataset_name drama --data_path data/scene_data/ --model_name sentence-transformers/all-distilroberta-v1 --do_learn --validate --evaluate --gpus 1 --experiment_id uar_scene --validate_every 1 --batch_size 8 --episode_length 8 --token_max_length 64 --embedding_dim 512
```

## Evaluation 

The notebook `evaluate.ipynb` displays some evaluations that we ran for the paper using the trained variants.
