import torch
import torch.nn as nn
import torchvision.models as models

### resnet18
class HashModel(nn.Module):
    def __init__(self,hash_length):
        super(HashModel, self).__init__()
        resmodel=models.resnet18(pretrained=True)
        self.feature_extractor=nn.Sequential(*list(resmodel.children())[:-1])#一个继承nn.module的model它包含一个叫做children()的函数，这个函数可以用来提取出model每一层的网络结构，在此基础上进行修改即可,【：-1】就是说去掉最后一层#预训练resnet模型的后两层是(avg pooling层和FC层)
        self.hashcoder=nn.Sequential(nn.Linear(resmodel.fc.in_features, int(hash_length)),nn.Tanh())

    def forward(self,x):
        feature=self.feature_extractor(x)
        feature = feature.view(feature.size(0), -1) #view()函数作用是将一个多行的Tensor,拼接成一行
        embeddings=self.hashcoder(feature)
        return embeddings

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
