#!/bin/bash
# Beam search, num beams = 20, return 20 best, rank by logp, and return top scoring candidate.
#seed_type="div_seeds"
#seed_type="src_valid"
#seed_type="src_train_900maxdiv_seeds"
seed_type="jin_test"
topk=5
model_type="softmax_randtop${topk}"
seed_file="/tigress/fdamani/mol-edit-data/data/qed/test_sets/selfies/${seed_type}.csv"

#rank_type="maxdeltasim"
#rank_type="logp"
rank_type="norank4"
#rank_type="mindeltasim"
#rank_type="minmolwt"
#rank_type="maxseedsim"

echo Running Script...
echo seed_type: ${seed_type}
echo Rank_type: ${rank_type}
echo Model_type: ${model_type}
mkdir /tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}
mkdir /tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}
mkdir /tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands
mkdir /tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/seeds
cp ${seed_file} /tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/0.csv
#output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/1.csv"
# number of random samples

num_rand_samples=5000
for i in $(seq 15000 20000)
do
	ind=$i
	echo ${ind}
	output_file="/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/seeds/${ind}.csv"
	python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-qed/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
						-random_sampling_topk ${topk} \
						-random_sampling_temp 1.0 \
						-beam_size 1 \
						-seed ${ind} \
						-gpu 0 \
						-src ${seed_file} \
						-output ${output_file}

done
# Merge files input_dir and output_dir
input_dir="/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/seeds"
output_dir="/tigress/fdamani/mol-edit-output/onmt-qed/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands"
num_rand_samples=5000
python merge_preds.py ${num_rand_samples} ${input_dir} ${output_dir} 1
exit 0
# number of recursive iterations
END=100
for i in $(seq 1 $END)
do
	ind=$i
	indplus1=$((i + 1))
	indminus1=$((i - 1))
	echo ${ind}
	echo ${indplus1}
	# merge files
	candssrcfile="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/${ind}.csv"
	rankoutputfile="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/${ind}.csv"
	prevseeds="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/${indminus1}.csv"
	#outputfile="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/${indplus1}.csv"
	echo ${candssrcfile}
	echo ${rankoutputfile}
	echo ${outputfile}

	python_rank_script="rank_by_${rank_type}.py"
	python ${python_rank_script} ${candssrcfile} ${rankoutputfile} ${num_rand_samples} ${prevseeds} ${seed_file}

	# number of random samples
	for j in $(seq 1 $num_rand_samples)
	do
		ind=$j
		output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/seeds/${ind}.csv"
		rand_seed=$j
		python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
							-random_sampling_topk ${topk} \
							-random_sampling_temp 1.0 \
							-beam_size 1 \
							-seed ${rand_seed} \
							-gpu 0 \
							-src ${rankoutputfile} \
							-output ${output_file}
	done
	python merge_preds.py ${num_rand_samples} ${input_dir} ${output_dir} ${indplus1}

done