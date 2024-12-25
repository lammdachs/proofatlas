#python -m select_problems
#python -m solve_problems_mixed --strategy 666
python -m create_dataset --strategy 666 --train_val_split 1.0
python -m train_mixed_model --epochs 64 --batch_size 1 --accumulate_grad_batches 128 --dim 1024 --gnn_layers 8 --transformer_layers 1 --n_heads 16 --dataset 666 --name mixed
python -m solve_problems_mixed --strategy mixed
