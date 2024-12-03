#python -m select_problems
#python -m solve_problems --strategy 666
#python -m create_dataset --strategy 666
#python -m train_model --dataset 666 --name stage1 --layers 8 --epochs 256 --batch_size 256 --accumulate_grad_batches 1 --dim 256
python -m solve_problems --strategy stage1
