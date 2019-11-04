python ../../OpenNMT-py/preprocess.py -train_src /tigress/fdamani/mol-edit-data/data/logp08/src_train.csv \
									  -train_tgt /tigress/fdamani/mol-edit-data/data/logp08/tgt_train.csv \
									  -valid_src /tigress/fdamani/mol-edit-data/data/logp08/src_valid.csv \
									  -valid_tgt /tigress/fdamani/mol-edit-data/data/logp08/tgt_valid.csv \
									  -share_vocab \
									  -save_data /tigress/fdamani/mol-edit-data/data/logp08/onmt-data