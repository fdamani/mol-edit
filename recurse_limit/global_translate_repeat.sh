#!/bin/bash
# Beam search, num beams = 20, return 20 best, rank by logp, and return top scoring candidate.
#seed_type="div_seeds"
#seed_type="src_valid"
seed_type="src_train_900maxdiv_seeds"
model_type="softmax_randtop2"
seed_file="/tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/${seed_type}.csv"
#output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/1.csv"
# number of random samples
num_rand_samples=5
for i in $(seq 1 $num_rand_samples)
do
	ind=$i
	echo ${ind}
	output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/seeds/${ind}.csv"
	python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
						-random_sampling_topk 2 \
						-random_sampling_temp 1.0 \
						-beam_size 1 \
						-seed ${ind} \
						-gpu 0 \
						-src ${seed_file} \
						-output ${output_file}

done
# Merge files input_dir and output_dir
input_dir="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/seeds"
output_dir="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands"
python merge_preds.py ${num_rand_samples} ${input_dir} ${output_dir} 1
#rank_type="toplogp"
rank_type="maxdeltasim"
#rank_type="mindeltasim"
#rank_type="minmolwt"
#rank_type="maxseedsim"

# number of recursive iterations
END=3
for i in $(seq 1 $END)
do
	ind=$i
	indplus1=$((i + 1))
	indminus1=$((i - 1))
	echo ${ind}
	echo ${indplus1}
	# merge files
	candssrcfile="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/${ind}.csv"
	rankoutputfile="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/${ind}.csv"
	prevseeds="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/${indminus1}.csv"
	#outputfile="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/${indplus1}.csv"
	echo ${candssrcfile}
	echo ${rankoutputfile}
	echo ${outputfile}

	if (($rank_type == "toplogp"))
	then
		echo rank_type is toplogp
		python rank_by_logp.py ${candssrcfile} ${rankoutputfile} ${num_rand_samples} ${prevseeds} ${seed_file}
	elif (($rank_type == "maxdeltasim"))
	then
		echo rank_type is maxdeltasim
		python rank_by_maxdeltasim.py ${candssrcfile} ${rankoutputfile} ${num_rand_samples} ${prevseeds} ${seed_file}
	elif (($rank_type == "mindeltasim"))
	then
		echo rank_type is mindeltasim
		python rank_by_mindeltasim.py ${candssrcfile} ${rankoutputfile} ${num_rand_samples} ${prevseeds} ${seed_file}
	elif (($rank_type == "minmolwt"))
	then
		echo rank_type is minmolwt
		python rank_by_minmolwt.py ${candssrcfile} ${rankoutputfile} ${num_rand_samples} ${prevseeds} ${seed_file}
	elif (($rank_type == "maxseedsim"))
	then
		echo rank_type is maxseedsim
		python rank_by_maxseedsim.py ${candssrcfile} ${rankoutputfile} ${num_rand_samples} ${prevseeds} ${seed_file}
	else
		echo ERROR. Correct rank_type not specified
	fi

	# number of random samples
	for j in $(seq 1 $num_rand_samples)
	do
		ind=$j
		output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/seeds/${ind}.csv"
		rand_seed=$j
		python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
							-random_sampling_topk 2 \
							-random_sampling_temp 1.0 \
							-beam_size 1 \
							-seed ${rand_seed} \
							-gpu 0 \
							-src ${rankoutputfile} \
							-output ${output_file}
	done
	python merge_preds.py ${num_rand_samples} ${input_dir} ${output_dir} ${indplus1}

done