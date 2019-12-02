python ../../OpenNMT-py/preprocess.py -train_src /tigress/fdamani/mol-edit-data/data/qed/src_train.csv \
									  -train_tgt /tigress/fdamani/mol-edit-data/data/qed/tgt_train.csv \
									  -valid_src /tigress/fdamani/mol-edit-data/data/qed/src_valid.csv \
									  -valid_tgt /tigress/fdamani/mol-edit-data/data/qed/tgt_valid.csv \
									  -share_vocab \
									  -save_data /home/fdamani/mol-edit/low-resource/sim/onmt-data-qed