import typing


class MyType:
    pass


# PEP 604 Union
u = MyType | None

# typing.Optional
o = typing.Optional[MyType]

print("PEP 604:")
print("origin:", getattr(u, "__origin__", None))
print("args:", getattr(u, "__args__", []))
print("typing get_origin:", typing.get_origin(u))
print("typing get_args:", typing.get_args(u))

print("\nOptional:")
print("origin:", getattr(o, "__origin__", None))
print("args:", getattr(o, "__args__", []))
print("typing get_origin:", typing.get_origin(o))
print("typing get_args:", typing.get_args(o))
