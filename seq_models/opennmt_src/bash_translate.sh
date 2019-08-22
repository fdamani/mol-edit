python ../../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/model-rnnenc-rnndec-1layer-500wordembed-500embed-dotattention-adamoptim_step_100000.pt \
					-random_sampling_topk 1 \
					-beam_size 5 \
					-src /tigress/fdamani/mol-edit-data/data/logp04/src_baseline_test.csv \
					-output /tigress/fdamani/mol-edit-output/onmt-logp04/preds/output-baselinetest-5beam-model-rnnenc-rnndec-1layer-500wordembed-500embed-dotattention-adamoptim_step_100000.csv