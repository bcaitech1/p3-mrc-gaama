python inference.py \
--seed 42 \
--output_dir ./test-v3 \
--dataset_name /opt/ml/input/data/data/test_dataset \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--model_name_or_path ./test-v3 \
--do_predict