python pretrain_ssd.py --data VOC \
--arch base \
--batch-size 128 \
--save '/SHARE_ST/capp_storage/jeesak_dir/InstaNasWACV/MBv1-SSD/pretrained' \
--resume \
--resume_path '/SHARE_ST/capp_storage/jeesak_dir/InstaNasWACV/MBv1-SSD/pretrained/FP/model_best.pth.tar' \
--from_fp_pretrain \
--retraining 'True'  \
--lr 5e-4  \
--arch_type 'V1+SSD' \
--ActQ 'LSQ+' \
--num_classes 21 \
--conf_threshold 0.1 \
--augmentation 'True' \
--gpu '0'  \
--optimizer 'sgd'  \
--lr_type 'cosine'  \
--trainer 'train'  \
--full_pretrain 'False' \
--epochs 300 \
--image_size 300 \
--vgg_batchnorm 'False' \
--extras_wbit 4 \
--extras_abit 4 \
--head_wbit 8 \
--head_abit 8 \
--input_norm 'True' \
--action_list 3 4 5 \
--test_epoch 5 \
--abit 4