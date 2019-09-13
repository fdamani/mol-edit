#!/bin/bash
# Beam search, num beams = 20, return 20 best, rank by logp, and return top scoring candidate.
#seed_type="div_seeds"
#seed_type="src_valid"
seed_type="src_train_900maxdiv_seeds"
topk=2
model_type = "beam"
#model_type="softmax_randtop${topk}"
seed_file="/tigress/fdamani/mol-edit-data/data/logp04/test_sets/selfies/${seed_type}.csv"

rank_type="maxdeltasim"
#rank_type="logp"
#rank_type="mindeltasim"
#rank_type="minmolwt"
#rank_type="maxseedsim"

echo Running Script...
echo seed_type: ${seed_type}
echo Rank_type: ${rank_type}
echo softmax_randtop${topk}

mkdir /tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}
mkdir /tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands
mkdir /tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/seeds
cp ${seed_file} /tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/0.csv
#output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/cands/1.csv"
# number of random samples
output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/1.csv"
python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
					-random_sampling_topk 1 \
					-beam_size 20 \
					-length_penalty avg \
					-gpu 0 \
					-n_best 20 \
					-src ${seed_file} \
					-output ${output_file}

# number of recursive iterations
END=4
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


	output_file="/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/${seed_type}/${model_type}/${rank_type}/cands/${indplus1}.csv"
	python ../OpenNMT-py/translate.py -model /tigress/fdamani/mol-edit-output/onmt-logp04/checkpoints/train_valid_share/model-mlpattention/model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000.pt \
						-random_sampling_topk 1 \
						-beam_size 20 \
						-length_penalty avg \
						-gpu 0 \
						-n_best 20 \
						-src ${rankoutputfile} \
						-output ${output_file}

done