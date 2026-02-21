import inspect

def my_func(a, b, **state):
    pass

sig = inspect.signature(my_func)
print([(p.name, p.kind) for p in sig.parameters.values()])
