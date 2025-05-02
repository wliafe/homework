class Vocab:
    '''词元表'''

    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if not reserved_tokens:  # 保留词元
            reserved_tokens = []
        self.unk = 0  # 未知词元索引为0
        tokens = [item[0] for item in tokens.items() if item[1] > min_freq]  # 删除低频词元
        self.idx_to_token = ['<unk>']+reserved_tokens+tokens  # 建立词元列表
        # 建立词元字典
        reserved_tokens_dict = {value: index+1 for index, value in enumerate(reserved_tokens)}
        tokens_dict = {value: index+1+len(reserved_tokens_dict) for index, value in enumerate(tokens)}
        self.token_to_idx = {'<unk>': 0}
        self.token_to_idx.update(reserved_tokens_dict)
        self.token_to_idx.update(tokens_dict)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, indices):
        if isinstance(indices, (list, tuple)):
            return [self.__getitem__(index) for index in indices]
        return self.idx_to_token[indices]

    def to_indices(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.to_indices(token) for token in tokens]
        return self.token_to_idx.get(tokens, self.unk)
