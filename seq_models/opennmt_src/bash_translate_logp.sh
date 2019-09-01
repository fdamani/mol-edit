python ../../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
					-random_sampling_topk 10 \
					-random_sampling_temp 1.0 \
					-beam_size 1 \
					-seed 820 \
					-gpu 0 \
					-src /tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/top800_top_90_pct_logp.csv \
					-output /tigress/fdamani/mol-edit-output/onmt-logp04/preds/train_valid_share/test_sets_top800_top_90/firstiter-randomsamplingtopk10-temp1-seed820-model.csv

python ../../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
					-random_sampling_topk 1 \
					-beam_size 20 \
					-length_penalty avg \
					-gpu 0 \
					-n_best 20 \
					-src /tigress/fdamani/mol-edit-output/onmt-logp04/preds/train_valid_share/test_sets_top800_top_90/firstiter-randomsamplingtopk10-temp1-seed820-model.csv \
					-output /tigress/fdamani/mol-edit-output/onmt-logp04/preds/train_valid_share/test_sets_top800_top_90/seconditer-20beam-20best-firstiterrandomsamplingtopk10temp1seed820.csv