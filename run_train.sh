python train.py \
--seed 42 \
--output_dir ./test-v3 \
--do_train \
--do_eval \
--evaluation_strategy steps \
--eval_steps 500 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--num_train_epochs 6 \
--weight_decay 0.01 \
--lr_scheduler_type cosine \
--learning_rate 2e-5