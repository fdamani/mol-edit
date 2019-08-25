python ../../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-qed/checkpoints/train_valid_360_share/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_98000.pt \
					-random_sampling_topk 1 \
					-beam_size 20 \
					-length_penalty avg \
					-gpu 0 \
					-n_best 20 \
					-src /tigress/fdamani/mol-edit-data/data/qed/src_valid.csv \
					-output /tigress/fdamani/mol-edit-output/onmt-qed/preds/train_valid_360_share/output-firstiter-validset-20beam-model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_98000.csv