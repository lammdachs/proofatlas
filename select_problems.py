import argparse
from dotenv import load_dotenv
import os
import signal
import sys
from tqdm import tqdm

from foreduce.tptp.parser import TooLargeError, read_file as read_tptp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=2**16)
    parser.add_argument("--max_nodes", type=int, default=2**8)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--max_symbols", type=int, default=2**5)
    args = parser.parse_args()
    
    def signal_handler(signal, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    load_dotenv()
    TPTP = os.getenv('TPTP')

    success = 0
    file_too_large = 0
    not_pure_first_order = 0
    too_deep = 0
    problem_too_large = 0
    too_many_symbols = 0
    for dir, problem in (pbar := tqdm(
        [
            (d, p)
            for d in os.listdir(f'{TPTP}/Problems/')
            for p in os.listdir(f'{TPTP}/Problems/{d}')
            if p.endswith('.p')
        ],
        desc=f'Selecting Problems'
    )):
        pbar.set_postfix({
            'Problem': problem,
            'Success': success,
            'File too Large': file_too_large,
            'Not FO': not_pure_first_order,
            'Too Deep': too_deep,
            'Problem too Large': problem_too_large,
            'Too Many Symbols': too_many_symbols
        })
        try:
            p = read_tptp(
                f'{TPTP}/Problems/{dir}/{problem}',
                include_path=TPTP,
                max_size=args.max_size
            )
        except TooLargeError:
            file_too_large += 1
            continue
        except:
            not_pure_first_order += 1
            continue
        if p.depth() > args.max_depth:
            too_deep += 1
            continue
        if len(p.function_symbols()) + len(p.predicate_symbols()) > args.max_symbols:
            too_many_symbols += 1
            continue
        graph, _, _ = p.to_graph()
        if graph.number_of_nodes() > args.max_nodes:
            problem_too_large += 1
            continue
        os.makedirs(f'./problems/{dir}', exist_ok=True)
        with open(f'./problems/{dir}/{problem}', 'w') as f:
            f.write(p.to_tptp())
        success += 1
    pbar.set_postfix({
        'Success': success,
        'File too Large': file_too_large,
        'Not FO': not_pure_first_order,
        'Too Deep': too_deep,
        'Too Large': problem_too_large,
        'Too Many Symbols': too_many_symbols
    })
