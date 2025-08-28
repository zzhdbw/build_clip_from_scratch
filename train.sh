export HF_ENDPOINT=https://hf-mirror.com
# python make_model.py 

deepspeed --include "localhost:0,1,2,3,4,5,6,7" train.py \
    --deepspeed ./config/ds_z2_config.json \
    --output_dir ./ckpt \
    --model_name_or_path ./pretrained_model/clip-roberta \
    --data_dir $PWD/data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --trust_remote_code True \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --dataloader_num_workers=4 \
    --dataloader_pin_memory=True \
    --per_device_train_batch_size="512" \
    --per_device_eval_batch_size="512" \
    --learning_rate="5e-5" \
    --warmup_steps="0" \
    --weight_decay 0.1 \
    --logging_steps="5" \
    --overwrite_output_dir \
    --report_to="swanlab"