python ../../OpenNMT-py/preprocess.py -train_src /tigress/fdamani/mol-edit-data/data/qed/src_train.csv \
									  -train_tgt /tigress/fdamani/mol-edit-data/data/qed/tgt_train.csv \
									  -valid_src /tigress/fdamani/mol-edit-data/data/qed/src_valid.csv \
									  -valid_tgt /tigress/fdamani/mol-edit-data/data/qed/tgt_valid.csv \
									  -save_data /tigress/fdamani/mol-edit-data/data/qed/onmt-copy \
									  -dynamic_dict \
									  -share_vocab