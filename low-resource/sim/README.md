transfer from low sim to high sim

train (logp04):
/tigress/fdamani/mol-edit-data/data/logp04/src_train.csv
/tigress/fdamani/mol-edit-data/data/logp04/tgt_train.csv

test:
/tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/jin_test.csv

model:
/tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt



TODO
- train models on logp06
	- rnn with attention
	- transformer
- can we generalize to logp075?
	- train a model on logp075 initialized with weights from logp06
	- perform inference using logp06 model
	- how do these compare?