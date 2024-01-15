import yaml
'''
1. 本版本的yaml为Version: 5.4.1，所以抛弃了yaml.load(stream)方法，
    所以会报以下错误：TypeError: load() missing 1 required positional argument: 'Loader'
    所以在yaml.load(stream, Loader)形式的文件中输入为yaml.load(file_stream, Loader=yaml.FullLoader)
2. 在yaml中不能使用Tab，需要使用四个space代替
3. yaml注释形式为# 
4. yaml对于""和''不敏感，但是对大小写敏感
'''

def load_yaml(data_dir):
    with open(data_dir) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config

if __name__ == "__main__":
    data_dir = './default.yaml'
    config = load_yaml(data_dir)
    print(config)