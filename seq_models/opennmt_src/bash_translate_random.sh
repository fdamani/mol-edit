python ../../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
					-random_sampling_topk -1 \
					-random_sampling_temp 1.0 \
					-beam_size 1 \
					-seed 829 \
					-gpu 0 \
					-src /tigress/fdamani/mol-edit-data/data/logp04/src_baseline_test.csv \
					-output /tigress/fdamani/mol-edit-output/onmt-logp04/preds/train_valid_share/output-firstiter-baseline-randomsampling-best-model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.csv