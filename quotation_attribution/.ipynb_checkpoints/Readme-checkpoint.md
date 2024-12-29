Code and data used to run experiments on quotation attribution on PDNC.

  Note that we use checkpoints for $Uar_{scene}$ and $UAR_{play}$, that you can download [here](https://www.file.io/H2Wm/download/sJSrMQnO2rD7).
  We will release a public huggingface model for both models in a near future.

  All our results are stored in `training/*results*/`, and we showcase results of Table 3 to 5 in the jupyter notebook `process_results.ipynb`.

### Training 

To train a model on PDNC, you can run the following command:

```
python train_speaker.py --dataPath=data/pdnc-no-minor_w100_full --sourceData=../data/pdnc_source/ --base_model=SpanBERT/spanbert-large-cased --savePath=results_vanilla_span_w100_full --model_mode=vanilla
```

and if using our UAR embeddings, use:
```
python train_speaker.py --dataPath=data/pdnc-no-minor_w100_full --sourceData=../data/pdnc_source/ --base_model=SpanBERT/spanbert-large-cased --savePath=results2_scene_drama_span_w100_full --model_mode=drama_luar --path_to_ckpt=scene_drama_lar.ckpt
```

the `model_mode` argument accepts: [`vanilla`, `semantics`, `luar`, `drama_luar`]. If given argument is `drama_luar`, you need to specify a path to a model checkpoint to load UAR.

### Inference

To run inference on the 6 novels that are part of the second release of PDNC, run:

```
python test_speaker.py --dataPath=data/test-pdnc-no-minor-w100_full --sourceData=../data/test_pdnc_source/ --base_model=SpanBERT/spanbert-large-cased --savePath=test_results_vanilla_span_w100_full --model_mode=vanilla --model_path=results_vanilla_span_w100_full/split_0/best_model.model
```