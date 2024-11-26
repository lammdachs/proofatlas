import argparse
import os
from dotenv import load_dotenv

from foreduce.tptp.parser import read_file as read_tptp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=2*10)
    parser.add_argument("--max_nodes", type=int, default=2**8)
    args = parser.parse_args()
    
    env = load_dotenv()
    TPTP = env['TPTP']

    success = 0; total = 0
    for dir in os.listdir(TPTP + 'Problems'):
        os.makedirs('gnn/problems/' + dir, exist_ok=True)
        for problem in os.listdir(TPTP + 'Problems/' + dir):
            print(f"Processing {problem}...", end=' ')
            total += 1
            if not os.path.exists('gnn/problems/' + dir + '/' + problem):
                try:
                    p = read_tptp(
                        TPTP + 'Problems/' + dir +'/' + problem,
                        include_path=TPTP,
                        max_size=args.max_size
                    )
                except:
                    print('Failed to parse')
                    continue
                graph, _, _ = p.to_graph()
                if graph.number_of_nodes() > args.max_nodes_problem:
                    print('Graph too large')
                    continue
                with open('gnn/problems/' + dir + '/' + problem, 'w') as f:
                    f.write(p.to_tptp())
                print('Success')
                success += 1
    print(f"Successfully processed {success}/{total} problems.")
