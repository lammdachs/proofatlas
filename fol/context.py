class Context:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.sorts = {}
        self.symbols = {}
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass