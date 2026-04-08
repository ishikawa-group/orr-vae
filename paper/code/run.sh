# Run python script
# Defaults aligned with the paper (can be overridden via environment variables)
: "${LABEL_THRESHOLD:=0.3}"
: "${BATCH_SIZE:=16}"
: "${MAX_EPOCH:=200}"
: "${LATENT_SIZE:=32}"
: "${BETA:=1}"
: "${SEED:=0}"

python3 01_generate_random_structures.py --num 128 --seed ${SEED} --output_dir ${DATA_DIR}

python3 02_calculate_overpotentials.py run-all --iter 0 --base_dir ${OUTPUT_DIR}
python3 03_conditional_vae.py --iter 0 --label_threshold ${LABEL_THRESHOLD} --batch_size ${BATCH_SIZE} --max_epoch ${MAX_EPOCH} --beta ${BETA} --latent_size ${LATENT_SIZE} --seed ${SEED} --base_data_path ${DATA_DIR} --result_base_path ${RESULT_DIR}
python3 04_generate_new_structures.py --iter 0 --num 128 --overpotential_condition 1 --alloy_stability_condition 1 --latent_size ${LATENT_SIZE} --seed ${SEED} --output_dir ${DATA_DIR} --result_dir ${RESULT_DIR}

python3 02_calculate_overpotentials.py run-all --iter 1 --base_dir ${OUTPUT_DIR}
python3 03_conditional_vae.py --iter 1 --label_threshold ${LABEL_THRESHOLD} --batch_size ${BATCH_SIZE} --max_epoch ${MAX_EPOCH} --beta ${BETA} --latent_size ${LATENT_SIZE} --seed ${SEED} --base_data_path ${DATA_DIR} --result_base_path ${RESULT_DIR}
python3 04_generate_new_structures.py --iter 1 --num 128 --overpotential_condition 1 --alloy_stability_condition 1 --latent_size ${LATENT_SIZE} --seed ${SEED} --output_dir ${DATA_DIR} --result_dir ${RESULT_DIR}

python3 02_calculate_overpotentials.py run-all --iter 2 --base_dir ${OUTPUT_DIR}
python3 03_conditional_vae.py --iter 2 --label_threshold ${LABEL_THRESHOLD} --batch_size ${BATCH_SIZE} --max_epoch ${MAX_EPOCH} --beta ${BETA} --latent_size ${LATENT_SIZE} --seed ${SEED} --base_data_path ${DATA_DIR} --result_base_path ${RESULT_DIR}
python3 04_generate_new_structures.py --iter 2 --num 128 --overpotential_condition 1 --alloy_stability_condition 1 --latent_size ${LATENT_SIZE} --seed ${SEED} --output_dir ${DATA_DIR} --result_dir ${RESULT_DIR}

python3 02_calculate_overpotentials.py run-all --iter 3 --base_dir ${OUTPUT_DIR}
python3 03_conditional_vae.py --iter 3 --label_threshold ${LABEL_THRESHOLD} --batch_size ${BATCH_SIZE} --max_epoch ${MAX_EPOCH} --beta ${BETA} --latent_size ${LATENT_SIZE} --seed ${SEED} --base_data_path ${DATA_DIR} --result_base_path ${RESULT_DIR}
python3 04_generate_new_structures.py --iter 3 --num 128 --overpotential_condition 1 --alloy_stability_condition 1 --latent_size ${LATENT_SIZE} --seed ${SEED} --output_dir ${DATA_DIR} --result_dir ${RESULT_DIR}

python3 02_calculate_overpotentials.py run-all --iter 4 --base_dir ${OUTPUT_DIR}
python3 03_conditional_vae.py --iter 4 --label_threshold ${LABEL_THRESHOLD} --batch_size ${BATCH_SIZE} --max_epoch ${MAX_EPOCH} --beta ${BETA} --latent_size ${LATENT_SIZE} --seed ${SEED} --base_data_path ${DATA_DIR} --result_base_path ${RESULT_DIR}
python3 04_generate_new_structures.py --iter 4 --num 128 --overpotential_condition 1 --alloy_stability_condition 1 --latent_size ${LATENT_SIZE} --seed ${SEED} --output_dir ${DATA_DIR} --result_dir ${RESULT_DIR}

python3 02_calculate_overpotentials.py run-all --iter 5 --base_dir ${OUTPUT_DIR}
python3 03_conditional_vae.py --iter 5 --label_threshold ${LABEL_THRESHOLD} --batch_size ${BATCH_SIZE} --max_epoch ${MAX_EPOCH} --beta ${BETA} --latent_size ${LATENT_SIZE} --seed ${SEED} --base_data_path ${DATA_DIR} --result_base_path ${RESULT_DIR}
python3 04_generate_new_structures.py --iter 5 --num 128 --overpotential_condition 1 --alloy_stability_condition 1 --latent_size ${LATENT_SIZE} --seed ${SEED} --output_dir ${DATA_DIR} --result_dir ${RESULT_DIR}

python3 02_calculate_overpotentials.py run-all --iter 6 --base_dir ${OUTPUT_DIR}
