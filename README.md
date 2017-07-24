
**Overview**

Kaggle competition https://www.kaggle.com/c/planet-understanding-the-amazon-from-space
Pytorch code by Edward Tyantov.

**Project Structure**

 * planet/ - python2.7 module
 * planet/pretrained/ - pretrained models, a lot of intermediate ones
 * planet/model.py - different CNN models (one function - one idependent model, imported explicitly in train.py)
   * planet/generic_models/ - boilerplate for resnets, ...
 * planet/boilerplate.py - routines + lr scheduler 
 * planet/transform_rules.py - rules to transform input data for training (normalization, augmentation, ...)
 * results/ - runtime results (automatically created)
 * data/ - in this folder the data should be placed as follows
 ```
 ls data
sample_submission.csv  test-jpg  test-tif-v2  train.csv  train-jpg  train-tif-v2
 ```
 
 **Hardware/Software requirements**
 
 Mandatory gpu + cuda. I used TitanX for training (4-8Gb memory was enough).
 
 My setup: 
  * Ubuntu 16.04.1 LTS
  * CUDA
  * Anaconda2 4.2.0 (sckit-learn, ...)
     * `wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh; chmod +X Anaconda2-4.4.0-Linux-x86_64.sh; ./Anaconda2-4.4.0-Linux-x86_64.sh`
  * Pytorch 0.1.12_2
     *  conda install pytorch torchvision -c soumith
  
 Additional packages:
  * visdom for graphs: pip install visdom
    * for start service: sudo python -m visdom.server -p 80
  * cv2: conda install opencv
  
 
 **Training**
 
 Train.py: train a model + generate a submission file.
 
 ```
Usage: train.py [options]

Options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size=BATCH_SIZE
  --test_batch_size=TEST_BATCH_SIZE
  -e EPOCH, --epoch=EPOCH
  -r WORKERS, --workers=WORKERS
  -l LR, --learning-rate=LR
  -m MOMENTUM, --momentum=MOMENTUM
  -w WEIGHT_DECAY, --weight_decay=WEIGHT_DECAY
  -o OPTIMIZER, --optimizer=OPTIMIZER
                        sgd|adam|yf
  -c FROM_CHECKPOINT, --from_checkpoint=FROM_CHECKPOINT
                        resums training from a specific epoch
  --train_percent=TRAIN_PERCENT
                        train/val split percantage
  --decrease_rate=DECREASE_RATE
                        For lr schedule, on plateau how much to descrease lr
  --early_stop_n=EARLY_STOP_N
                        Early stopping on a specific number of degrading
                        epochs
  --folds=FOLDS         Number of folds, for ensemble training only
  --lr_schedule=LR_SCHEDULE
                        possible: adaptive, adaptive_best, decreasing or
                        frozen.                        adaptive_best is the
                        same as plateau scheduler
  --model=MODEL         Which model to use, check model.py for names
  --transform=TRANSFORM
                        Specify a transformation rule. Check
                        transform_rules.py for names
  --weigh_loss=WEIGH_LOSS
                        weigh loss function according to class occurence or
                        not
  --warm_up_epochs=WARM_UP_EPOCHS
                        warm_up_epochs number if model has it
  --max_stops=MAX_STOPS
                        max_stops for plateau/adaptive lr schedule
  --weigh_sample=WEIGH_SAMPLE
                        weigh sample according to class occurrence or not
  --data_type=DATA_TYPE
                        Data type. Possible values: jpg|tif|tif-
                        index|mix|jpg_numpy
  --run_type=RUN_TYPE   train|eval
  --seed=SEED           
  --shard=SHARD         Postfix for results folder, where the results will be
                        saved, <results+shard>/
  --holdout=HOLDOUT     if eq. 1, then small 10% holdout set is not used for
                        training, but for blending later
  --blacklist=BLACKLIST
                        Use blacklist file of garbage images or labels, *Used
                        for ensembles only*
  ```
 
 Best single model was trained:
  
 `python2.7 train.py -l 0.3 -b 96 --model mix_net_v6 --transform mix_index_nozoom_256 --lr_schedule adaptive_best --data_type mix --max_stops 3 `


Ensemble:

`python2.7 train.py -l 0.3 -b 96 --model mix_net_v6 --transform mix_index_nozoom_256 --lr_schedule adaptive_best --data_type mix --max_stops 3 --run_type ens --folds 5 --shard _5_mixnet6_hvb`


Train.py automatically generate submission.csv, with ensemble option on - submission{0,1..}.csv + detailed_submission*.csv (with probabilities), which you should average or blend or stack after.

**Models**

My best single model (mixnetv6) solution consist of following tricks:
 * 6-channel input (3-jpg channel, NIR-chanell, NDWI-index, SAVI-index)
 * model: resnet18 on jpg, resnet 18 on nir+indexes, concat -> 256 embedding FC + final FC
   * jpg branch lr modifier 0.05 to base LR, for nir branch - layer{3,4} - 1.0, layer{2,3} - 0.1, FC - 1.0 
 * plateau scheduller on val loss (cross-entropy), patience=3
 * early stopping: 6 epochs
 * train time augmentation: shift, flip, scale, rotate, transpose
 * test time augmentation, 6x: as-is, rotate 90*{1,2,3}, flip x, flip y
 * standart for this challenge searching thresholds for F2 (firstly I implement per class search - it is more consistent, but default on scale is better)
 
Best Ensemble:
 * trained various models on jpg, mix channels
   * models: 
     * densenet{121,169} on jpg, 5 folds
     * mixnetv6, mixnetv3 (different LRs) 5,6,7 folds
     * wideresnet on 6-channel (for mix branch unpretrained WideResNet), 7 folds
     * resnet18 + embedding FC on 8 folds
 * best submit was based on weighting predictions using holdout F2 score, weight=((score - min_score)/max_score)**0.5

**Blending + stacking**

_Note that this scripts were written in a hurry and are just bunch of ad-hoc code. use it on your own risk ;)_


Using blend.py and stack.py one can ensemble predictions.
 
 For stacking the code uses out-of-fold prediction, which automatically are done.
 For blending (learning classifier on a holdout dataset) while training use --holdout to hold 10% of the train dataset (data/holdout.txt)

For stack.py one more prerequisite: 

`pip install stacked_generalization`

Usage:

'python2.7 stack.py --folder stacks/stack4'
```
$:ls stacks/stack4
5_densenet121_hv  5_dn169_hv  5_mixnet6_lowLR_hv  5_mixnetv3_hv  5_selu_hv  6_jpg_hv  6_mixnet6_lowLR_hv  7_mixnet6_lowLR_hv  7_wide_hv  8_jpg_hv
```

**Results**

 * Best single model: mixnet_v6: public: 0.92905, private: 0.93071
 * Best blended ensemble: public: 0.93015, private: 0.93217 
 * Best submit during competition: 0.93023: 0.93168 (also ensemble)

