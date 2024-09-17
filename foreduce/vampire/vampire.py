from pexpect import spawn, EOF
from sortedcontainers import SortedDict

from foreduce.fol.logic import Problem
from foreduce.vampire.parser import read_string


class VampireInteractive:
    def __init__(self, vampire_path, problem_path):
        self.path = vampire_path
        self.args = [
            vampire_path, 
            '--show_new', 'on',
            '--avatar', 'off',
            '--proof', 'off',
            '--manual_cs', 'on',
            '--time_limit', '0',
            '--saturation_algorithm', 'discount',
            problem_path
        ]
        self.problem, self.tree, self.mapping = Problem(), [], SortedDict({})
        self.proof = "% Running in auto input_syntax mode. Trying TPTP\n"
        self.active = []
        self.finished = False
        self.step_count = 0
        self._entered = False

    def read(self):
        if not self._entered:
            raise ValueError('Vampire not entered. Use **with** statement.')
        try:
            self.process.expect('Pick a clause:\r\n')
        except EOF:
            self.finished = True
        string = str(self.process.before, encoding='utf-8')
        string = string[string.find('\n')+1:].replace('\r\n', '\n')
        if 'Refutation found.' in string:
            self.finished = True
            self.proof += string
            return
        if string and not 'User error' in string:
            self.problem, self.tree, self.mapping = \
                read_string(string, self.problem, self.tree, self.mapping)
            self.proof += string
            self.active += [False for _ in range(len(self.problem.clauses) - len(self.active))]

    def step(self, i: int):
        if self.finished:
            raise EOFError('Vampire finished.')
        self.process.sendline(str(i+1))
        self.active[i] = True
        self.read()
        self.step_count += 1

    def __enter__(self):
        self._entered = True
        self.process = spawn(" ".join(self.args))
        self.read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.close()
        
    def __repr__(self):
        string = ""
        for i, (clause, active) in enumerate(zip(self.problem.clauses, self.active)):
            if active:
                string += f'\033[1m {i}: {clause}\033[0m\n'
            else:
                string += f'{i}: {clause}\n'
        return string
    

class VampireAutomatic:
    def __init__(
        self,
        vampire_path,
        problem_path,
        activation_limit=100,
        age_weight_ratio=(1, 1),
        selection='10'
    ):
        self.vampire_path = vampire_path
        self.problem_path = problem_path
        self.activation_limit = activation_limit
        self.age_weight_ratio = age_weight_ratio
        self.selection = selection
        self.problem, self.tree, self.mapping = Problem(), [], SortedDict({})
        self.proof = ""
        self.finished = False

    def run(self):
        args = [
            self.vampire_path,
            self.problem_path,
            '--show_new', 'on',
            '--avatar', 'off',
            '--proof', 'off',
            '--manual_cs', 'off',
            '--time_limit', '0',
            '--saturation_algorithm', 'discount',
            '--activation_limit', str(self.activation_limit),
            '--age_weight_ratio', f'{self.age_weight_ratio[0]}:{self.age_weight_ratio[1]}',
            '--selection', self.selection
        ]
        process = spawn(" ".join(args))
        process.expect(EOF)
        self.proof = str(process.before, encoding='utf-8').replace('\r\n', '\n')
        self.problem, self.tree, self.mapping = \
            read_string(self.proof, self.problem, self.tree, self.mapping)
        process.close()
        self.finished = True

    def __repr__(self):
        return self.proof
