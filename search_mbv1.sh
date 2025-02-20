python search_multinomial.py  \
--data 'VOC' \
--arch_type 'V1+SSD' \
--ActQ 'LSQ+' \
--num_classes 21 \
--cv_dir '/data/jeesak/InstaQNAS/MBv1-SSD/search' \
--instassd_chkpt '/data/jeesak/InstaQNAS/MBv1-SSD/pretrained/model_best.pth.tar' \
--retraining 'True' \
--sample_eval_path './prec_thre06/sample_eval_test' \
--search_eval_path './prec_thre06/search_eval_test' \
--eval_path './prec_thre06/eval_test1' \
--agent_lr 5e-4 \
--reward_type 'prec+bops_thre2' \
--conf_threshold 0.1 \
--pos_w 30 \
--neg_w -10 \
--baseline_min 42 \
--baseline 98 \
--baseline_max 70 \
--prec_thre 0.6  \
--extras_wbit 4 \
--extras_abit 4 \
--head_wbit 8 \
--head_abit 8 \
--vgg_batchnorm 'False' \
--input_norm 'True' \
--agent_lr_type 'cosine'  \
--full_pretrain 'False' \
--lr_type 'cosine' \
--optimizer 'sgd' \
--image_size 300 \
--plot_result \
--abit 4 \
--action_list 3 4 5 \
--test_first \
--lr 5e-4 --batch_size 64 --epochs 300 --gpu '0'