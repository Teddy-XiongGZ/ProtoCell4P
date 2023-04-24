for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        for data in "lupus"
        do
            for task in "disease"
            do
                # original
                python visualize.py \
                    --data $data \
                    --task $task \
                    --batch_size 4 \
                    --n_proto $n_p \
                    --model ProtoCell \
                    --subsample \
                    --load_ct \
                    --device cuda:0 \
                    --seed $exp \
                    --checkpoint_dir '../checkpoint/'$data'/'$task'/ProtoCell_'$n_p'_proto_'$exp \
                    --checkpoint_name 'best_model.pt' \
                    --type sample \
                    --k 20
            done
        done
    done
done

for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        for data in "lupus"
        do
            for task in "population"
            do
                # original
                python visualize.py \
                    --data $data \
                    --task $task \
                    --batch_size 4 \
                    --n_proto $n_p \
                    --model ProtoCell \
                    --subsample \
                    --load_ct \
                    --device cuda:0 \
                    --seed $exp \
                    --checkpoint_dir '../checkpoint/'$data'/'$task'/ProtoCell_'$n_p'_proto_'$exp \
                    --checkpoint_name 'best_model.pt' \
                    --type sample \
                    --k 20
            done
        done
    done
done


for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        for data in "cardio"
        do
            # original
            python visualize.py \
                --data $data \
                --batch_size 4 \
                --n_proto $n_p \
                --model ProtoCell \
                --subsample \
                --load_ct \
                --device cuda:0 \
                --seed $exp \
                --checkpoint_dir '../checkpoint/'$data'/ProtoCell_'$n_p'_proto_'$exp \
                --checkpoint_name 'best_model.pt' \
                --type sample \
                --k 20
        done
    done
done

for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        for data in "covid"
        do
            # original
            python visualize.py \
                --data $data \
                --batch_size 3 \
                --n_proto $n_p \
                --model ProtoCell \
                --subsample \
                --load_ct \
                --device cuda:0 \
                --seed $exp \
                --checkpoint_dir '../checkpoint/'$data'/ProtoCell_'$n_p'_proto_'$exp \
                --checkpoint_name 'best_model.pt' \
                --type sample \
                --k 20
        done
    done
done

for n_p in 4 8 16
do
    for exp in 1 2 3 4 5 6 7 8
    do
        for data in "lupus"
        do
            for task in "disease"
            do
                # without cell type
                python visualize.py \
                    --data $data \
                    --task $task \
                    --batch_size 4 \
                    --n_proto $n_p \
                    --model ProtoCell \
                    --subsample \
                    --device cuda:0 \
                    --seed $exp \
                    --checkpoint_dir '../checkpoint/'$data'/'$task'/ProtoCell_'$n_p'_proto_'$exp \
                    --checkpoint_name 'best_model.pt' \
                    --type sample \
                    --k 20

                # without pre-training
                python visualize.py \
                    --data $data \
                    --task $task \
                    --batch_size 4 \
                    --n_proto $n_p \
                    --model ProtoCell \
                    --subsample \
                    --load_ct \
                    --device cuda:0 \
                    --seed $exp \
                    --checkpoint_dir '../checkpoint/'$data'/'$task'/ProtoCell_wo_pre_'$n_p'_proto_'$exp \
                    --checkpoint_name 'best_model.pt' \
                    --type sample \
                    --k 20
            done
        done
    done
done
