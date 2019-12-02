python ../../OpenNMT-py/preprocess.py -train_src /tigress/fdamani/mol-edit-data/data/drd2500/drd2500-qed/src_train.csv \
									  -train_tgt /tigress/fdamani/mol-edit-data/data/drd2500/drd2500-qed/tgt_train.csv \
									  -valid_src /tigress/fdamani/mol-edit-data/data/drd2500/drd2500-qed/src_valid.csv \
									  -valid_tgt /tigress/fdamani/mol-edit-data/data/drd2500/drd2500-qed/tgt_valid.csv \
									  -share_vocab \
									  -save_data /tigress/fdamani/mol-edit-data/data/drd2500/drd2500-qed/onmt-data