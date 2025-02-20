python pretrain_ssd.py --data VOC \
--arch base \
--batch-size 24 \
--save '/SHARE_ST/capp_storage/jeesak_dir/InstaNasWACV/MBv1-SSD/pretrained/FP' \
--resume \
--resume_path '/SHARE_ST/capp_storage/jeesak_dir/InstaQNAS_ECCV/MBv1-SSD/pretrained/fp_model_best.pth.tar' \
--retraining True  \
--lr 0.0005  \
--arch_type 'V1+SSD' \
--num_classes 21 \
--conf_threshold 0.1 \
--augmentation 'True' \
--gpu '0'  \
--optimizer 'sgd'  \
--lr_type 'cosine'  \
--trainer 'train'  \
--full_pretrain 'True' \
--epochs 300 \
--image_size 300 \
--input_norm 'True' \
--test_first