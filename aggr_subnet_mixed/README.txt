(finished by Lei) run pre_process_tr_*.py and pre_process_te_*.py
 - you only need to change the pooling choice, either 'avg' or 'max'

(finished by Lei) run mixed_aggr.py
 - this is used to aggregate the features from different kinds of subsequences (e.g., 24-frame, 32-frame, 48-frame, and 64-frame)

then run main_avg.py or main_max.py
 - you may need to change the input information, e.g., different frame numbers for different subsequences (this is only for training each kind of subsequences separately, and for this you do not need to run mixed_aggr.py), you also need to change the name of the model file to be saved

To add more layers, update AggrNet.py

Lei
