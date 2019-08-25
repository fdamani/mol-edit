python ../../OpenNMT-py/train.py -data /tigress/fdamani/mol-edit-data/data/logp04/train_valid_share/data \
				-global_attention mlp \
				-word_vec_size 600 \
				-share_embeddings \
				-encoder_type brnn \
				-decoder_type rnn \
				-enc_layers 2 \
				-dec_layers 2 \
				-enc_rnn_size 600 \
				-dec_rnn_size 600 \
				-label_smoothing 0.1 \
				-rnn_type LSTM \
				-save_model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim \
				-train_steps 100000 \
				-valid_steps 500 \
				-batch_size 64 \
				-save_checkpoint_steps 1000 \
				-report_every 200 \
				-world_size 1 \
				-gpu_ranks 0 \
				-optim adam \
				-learning_rate 0.001