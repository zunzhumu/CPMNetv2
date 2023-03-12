python -u train_CPMv2_MRA.py  \
--log='logs/****.log' \
--save_model_dir="" \
--save_FrocResult_dir="" \
--lr=1e-4 --epochs=300  --num_workers=16 --batch-size=6 \
--topk=5 --lambda_cls=4.0 --lambda_shape=0.1 --lambda_offset=1.0 --lambda_iou=1.0 \
--norm_type='batchnorm' --head_norm='batchnorm' --act_type='ReLU' --num_sam=4 \
--val_csv='val.csv' --train_csv='train.csv' --outputDir='bbox01_aneurysm' --root='' &
