import torch

def load_model():
    # 加载第一个模型参数
    original_state_dict = torch.load('./encoder_state.pt')
    # 创建一个新的字典来存储修改键名后的参数
    new_state_dict = {}

    # 将第一个模型参数添加到新字典中
    for key, value in original_state_dict.items():
        new_state_dict[key] = value

    # 加载第二个模型参数
    original_state_dict1 = torch.load('./classifier.pt')
    for key, value in original_state_dict1.items():
        # 修改第二个模型参数的键名，避免与第一个模型参数冲突
        new_key = 'classifier_' + key  # 在键名前添加一个前缀，以示区分
        new_state_dict[new_key] = value

    return new_state_dict

def print_model_state_dict(model_state_dict):
    for key, value in model_state_dict.items():
        print("key: {}, shape: {}".format(key, value.shape))

def save_model_as_bin(model_state_dict, save_path):
    torch.save(model_state_dict, save_path)
    print("Model state dict saved as binary file to:", save_path)

# 加载模型
new_model_state_dict = load_model()

# 打印新模型的参数
print_model_state_dict(new_model_state_dict)

# 保存新模型的参数为二进制文件
save_model_as_bin(new_model_state_dict, "./combined_model.bin")
