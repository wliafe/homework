class Vocab:
    '''词元表'''

    def __init__(self, tokens: dict, min_freq: int = 0, reserved_tokens: list = None):
        '''初始化'''
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

    def __len__(self) -> int:
        '''返回词表大小'''
        return len(self.idx_to_token)

    def __getitem__(self, indices: int | list[int] | tuple[int]) -> str | list[str]:
        '''根据索引返回词元'''
        if isinstance(indices, (list, tuple)):
            return [self.__getitem__(index) for index in indices]
        return self.idx_to_token[indices]

    def to_indices(self, tokens: str | list[str] | tuple[str]) -> int | list[int]:
        '''根据词元返回索引'''
        if isinstance(tokens, (list, tuple)):
            return [self.to_indices(token) for token in tokens]
        return self.token_to_idx.get(tokens, self.unk)
