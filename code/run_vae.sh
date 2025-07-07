# Run python script
python3 01_generate_random_structures.py --num 128

python3 02_run_all_calculations.py --iter 0
python3 03_conditional_vae.py --iter 0 --max_epoch 200 --beta 1 --latent_size 32
python3 04_generate_new_structures.py --iter 0 --num 128 --overpotential_condition 1 --pt_fraction_condition 1 --latent_size 32
python3 05_visualize_latent_space.py --iter 0 --latent_size 32

python3 02_run_all_calculations.py --iter 1
python3 03_conditional_vae.py --iter 1 --max_epoch 200 --beta 1 --latent_size 32
python3 04_generate_new_structures.py --iter 1 --num 128 --overpotential_condition 1 --pt_fraction_condition 1 --latent_size 32
python3 05_visualize_latent_space.py --iter 1 --latent_size 32

python3 02_run_all_calculations.py --iter 2
python3 03_conditional_vae.py --iter 2 --max_epoch 200 --beta 1 --latent_size 32
python3 04_generate_new_structures.py --iter 2 --num 128 --overpotential_condition 1 --pt_fraction_condition 1 --latent_size 32
python3 05_visualize_latent_space.py --iter 2 --latent_size 32

python3 02_run_all_calculations.py --iter 3
python3 03_conditional_vae.py --iter 3 --max_epoch 200 --beta 1 --latent_size 32
python3 04_generate_new_structures.py --iter 3 --num 128 --overpotential_condition 1 --pt_fraction_condition 1 --latent_size 32
python3 05_visualize_latent_space.py --iter 3 --latent_size 32

python3 02_run_all_calculations.py --iter 4
python3 03_conditional_vae.py --iter 4 --max_epoch 200 --beta 1 --latent_size 32
python3 05_visualize_latent_space.py --iter 4 --latent_size 32

python3 06_analyze_orr_catalyst_data.py --iter 4