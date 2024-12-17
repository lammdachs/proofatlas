python -m select_problems
python -m solve_problems_mixed --strategy 666
python -m create_dataset --strategy 666
python -m train_mixed_model --epochs 256 --batch_size 1 --accumulate_grad_batches 128 --dim 768 --gnn_layers 12 --transformer_layers 12 --n_heads 12 --dataset 666 --name mixed
python -m solve_problems_mixed --strategy stage1
