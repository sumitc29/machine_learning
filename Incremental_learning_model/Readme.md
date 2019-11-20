# Increamental models
These are the model which can be re-trained with the new data provided,updating the weights while carrying forward with tle old trained weights. In this case while retraining weights are not updated from scratch, these are updated in continuation with the last training.

Incremental models can be found here [here](https://scikit-learn.org/stable/modules/computing.html#incremental-learning)

Following are the attributes which gives model utility of re-train
* partial_fit()
* warm_state 
