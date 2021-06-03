import toml

dict_toml = toml.load(open('cmvc/config.toml'))
print(dict_toml["tool"])
