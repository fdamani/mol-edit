python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
					-random_sampling_topk 1 \
					-beam_size 1 \
					-length_penalty avg \
					-gpu 0 \
					-n_best 1 \
					-src /tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/div_seeds.csv \
					-output /tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/div_seeds/greedy/1.csv \