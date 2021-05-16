import toml

dict_toml = toml.load(open('pyproject.toml'))
print(dict_toml["tool"])
