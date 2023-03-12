python -u train_CPMV2_RibFrac.py  \
--log='logs/***.log' \
--save_model_dir="" \
--save_FrocResult_dir="" \
--lr=1e-4 --epochs=400  --num_workers=40 --batch-size=3 \
--topk=5 --lambda_cls=4.0 --lambda_shape=0.1 --lambda_offset=1.0 --lambda_iou=1.0 \
--norm_type='batchnorm' --head_norm='batchnorm' --act_type='ReLU' --num_sam=14 \
--train_csv='train.csv' --test_csv='test.csv' --outputDir='bbox01' --root='' &