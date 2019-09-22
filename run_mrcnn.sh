python train_old.py \
    --train_module MRCNN \
    --log_dir log_mrcnn_chair_lamp_storagefurniture_pseudo_seg_2_ep20  \
    --restore_model_path log_spn_chair_lamp_storagefurniture_pseudo_seg_2_ep20/best_model_epoch_19.ckpt \
    --restore_scope shape_proposal_net \
    --category Chair Lamp StorageFurniture \
    --level_id 2  \
    --pseudo_seg  
