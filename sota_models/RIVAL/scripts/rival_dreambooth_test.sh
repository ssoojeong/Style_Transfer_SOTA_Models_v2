CUDA_VISIBLE_DEVICES=2 python rival/test_variation_sdv1.py \
 --inf_config "configs/rival_t2i.json" \
 --inner_round 1 \
 --exp_folder "out/dreambooth_exps" \
 --pretrained_model_path "sj98/wiki_2" \
 --img_config "assets/images/configs_dreambooth_my.json" 