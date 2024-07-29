import unittest
import torch
from foreduce.transformer.embedding import EmbeddingConfig, FormulaEmbedding


class UpdateConfig(unittest.TestCase):
    RESERVED_TOKENS = 2
    num_variables = 2
    num_functions = [2, 1]
    
    new_num_variables_0 = 4
    
    new_num_functions_1 = [2, 2]
    
    new_num_functions_2 = [2, 1, 1]
    
    new_num_variables_3 = 4
    new_num_functions_3 = [2, 2, 1]
    
    new_num_variables_4 = 1
    new_num_funtions_4 = [1, 1]

    def init_embedding(self):
        return FormulaEmbedding(EmbeddingConfig(
            self.RESERVED_TOKENS,
            self.num_variables,
            [i for i in self.num_functions],
            1    
        ))
    
    def test_update_variables(self):
        embedding = self.init_embedding()
        cfg = embedding.config
        embedding.update_config(self.new_num_variables_0, self.num_functions)
        self.assertEqual(cfg.RESERVED_TOKENS, self.RESERVED_TOKENS)
        self.assertEqual(cfg.num_variables, self.new_num_variables_0)
        self.assertEqual(cfg.num_functions, self.num_functions)
        for i in range(self.num_variables, cfg.num_variables):
            self.assertEqual(
                float(embedding.embeddings(torch.tensor([self.RESERVED_TOKENS + i], dtype=int))),
                float(embedding.embeddings(torch.tensor([self.RESERVED_TOKENS + self.num_variables], dtype=int)))
            )

    def test_update_functions_1(self):
        embedding = self.init_embedding()
        cfg = embedding.config
        embedding.update_config(self.num_variables, self.new_num_functions_1)
        self.assertEqual(cfg.RESERVED_TOKENS, self.RESERVED_TOKENS)
        self.assertEqual(cfg.num_variables, self.num_variables)
        self.assertEqual(cfg.num_functions, self.new_num_functions_1)
        for i in range(len(self.num_functions)):
            for j in range(self.num_functions[i], cfg.num_functions[i]):
                self.assertEqual(
                    float(embedding.embeddings(torch.tensor([cfg.RESERVED_TOKENS + cfg.num_variables + sum(cfg.num_functions[:i]) + j], dtype=int))),
                    float(embedding.embeddings(torch.tensor([cfg.RESERVED_TOKENS + cfg.num_variables + sum(cfg.num_functions[:i])], dtype=int)))
                )
    
    def test_update_functions_2(self):
        embedding = self.init_embedding()
        cfg = embedding.config
        embedding.update_config(self.num_variables, self.new_num_functions_2)
        self.assertEqual(cfg.RESERVED_TOKENS, self.RESERVED_TOKENS)
        self.assertEqual(cfg.num_variables, self.num_variables)
        self.assertEqual(cfg.num_functions, self.new_num_functions_2)
        for i in range(sum(cfg.num_functions[len(self.num_functions):])):
            self.assertEqual(
                float(embedding.embeddings.weight[-1-i]),
                float(embedding.embeddings.weight[-1-sum(cfg.num_functions[len(self.num_functions):])])
            )

    def test_update_all(self):
        embedding = self.init_embedding()
        cfg = embedding.config
        embedding.update_config(self.new_num_variables_3, self.new_num_functions_3)
        self.assertEqual(cfg.RESERVED_TOKENS, self.RESERVED_TOKENS)
        self.assertEqual(cfg.num_variables, self.new_num_variables_3)
        self.assertEqual(cfg.num_functions, self.new_num_functions_3)
        for i in range(self.num_variables, self.new_num_variables_3):
            self.assertEqual(
                float(embedding.embeddings(torch.tensor([cfg.RESERVED_TOKENS + i], dtype=int))),
                float(embedding.embeddings(torch.tensor([cfg.RESERVED_TOKENS + self.num_variables], dtype=int)))
            )
        for i in range(len(self.num_functions)):
            for j in range(self.num_functions[i], cfg.num_functions[i]):
                self.assertEqual(
                    float(embedding.embeddings(torch.tensor([cfg.RESERVED_TOKENS + cfg.num_variables + sum(cfg.num_functions[:i]) + j], dtype=int))),
                    float(embedding.embeddings(torch.tensor([cfg.RESERVED_TOKENS + cfg.num_variables + sum(cfg.num_functions[:i])], dtype=int)))
                )
        for i in range(sum(cfg.num_functions[len(self.num_functions):])):
            self.assertEqual(
                float(embedding.embeddings.weight[-1-i]),
                float(embedding.embeddings.weight[-1-sum(cfg.num_functions[len(self.num_functions):])])
            )

    def test_null_update(self):
        embedding = self.init_embedding()
        cfg = embedding.config
        embedding.update_config(self.new_num_variables_4, self.new_num_funtions_4)
        self.assertEqual(cfg.RESERVED_TOKENS, self.RESERVED_TOKENS)
        self.assertEqual(cfg.num_variables, self.num_variables)
        self.assertEqual(cfg.num_functions, self.num_functions)
        

if __name__ == "__main__":
    unittest.main()