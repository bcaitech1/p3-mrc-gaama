python inference.py \
--seed 42 \
--output_dir ../output \
--overwrite_output_dir \
--dataset_name /opt/ml/input/data/test_dataset \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--model_name_or_path /opt/ml/ckpts \
--do_predict