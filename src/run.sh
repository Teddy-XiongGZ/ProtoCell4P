for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        python run.py \
            --data lupus \
            --batch_size 4 \
            --lr 1e-4 \
            --max_epoch 75 \
            --task disease \
            --n_proto $n_p \
            --split_ratio 0.5 0.25 0.25 \
            --model ProtoCell \
            --exp_str 'ProtoCell_'$n_p'_proto_'$exp \
            --device cuda:0 \
            --seed $exp \
            --d_min 1 \
            --lambda_1 1 \
            --lambda_2 1 \
            --lambda_3 1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --lambda_6 1 \
            --subsample \
            --load_ct \
            --pretrained \
            --lr_pretrain 1e-2 \
            --max_epoch_pretrain 50 \
            --keep_sparse
    done
done

for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        python run.py \
            --data lupus \
            --batch_size 4 \
            --lr 1e-4 \
            --max_epoch 75 \
            --task population \
            --n_proto $n_p \
            --split_ratio 0.5 0.25 0.25 \
            --model ProtoCell \
            --exp_str 'ProtoCell_'$n_p'_proto_'$exp \
            --device cuda:0 \
            --seed $exp \
            --d_min 1 \
            --lambda_1 1 \
            --lambda_2 1 \
            --lambda_3 1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --lambda_6 1 \
            --subsample \
            --load_ct \
            --pretrained \
            --lr_pretrain 1e-2 \
            --max_epoch_pretrain 50 \
            --keep_sparse
    done
done


for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        python run.py \
            --data cardio \
            --batch_size 3 \
            --lr 5e-5 \
            --max_epoch 75 \
            --n_proto $n_p \
            --split_ratio 0.5 0.25 0.25 \
            --model ProtoCell \
            --exp_str 'ProtoCell_'$n_p'_proto_'$exp \
            --device cuda:0 \
            --seed $exp \
            --d_min 1 \
            --lambda_1 1 \
            --lambda_2 1 \
            --lambda_3 1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --lambda_6 1 \
            --subsample \
            --load_ct \
            --pretrained \
            --lr_pretrain 1e-2 \
            --max_epoch_pretrain 50 \
            --keep_sparse
    done
done

for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        python run.py \
            --data covid \
            --batch_size 3 \
            --lr 1e-4 \
            --max_epoch 75 \
            --n_proto $n_p \
            --split_ratio 0.5 0.25 0.25 \
            --model ProtoCell \
            --exp_str 'ProtoCell_'$n_p'_proto_'$exp \
            --device cuda:0 \
            --seed $exp \
            --d_min 1 \
            --lambda_1 1 \
            --lambda_2 1 \
            --lambda_3 1 \
            --lambda_4 1 \
            --lambda_5 1 \
            --lambda_6 1 \
            --subsample \
            --load_ct \
            --pretrained \
            --lr_pretrain 1e-2 \
            --max_epoch_pretrain 75 \
            --keep_sparse
    done
done