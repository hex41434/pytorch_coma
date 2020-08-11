from config_parser import read_config


config = read_config(fname='')

print(str(config))

# with open('testconfig.txt','w') as f: 
#     conf = str(config)
#     f.write(conf)