import torch


def collate_fn(data):  # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    print('data: ', type(data), len(data))
    print('data[0]: ', type(data[0]), len(data[0]))
    print('data[0][0]: ', type(data[0][0]), data[0][0])
    print('data[0][1]: ', type(data[0][1]), data[0][1])
    print('data[0][0] length: ', len(data[0][0]))
    print('data[0][0][0]: ', type(data[0][0][0]))
    data.sort(key=lambda x: len(x[0][0]), reverse=False)  # 按照数据长度升序排序，注意x为data的每一个元素，它是一个tuple，其中x[0]为真实数据，此处为一个二维列表
    data_list = []
    label_list = []
    min_len = len(data[0][0][0]) # 最短的数据长度
    print('min_len: ', min_len)
    for batch in range(0, len(data)): #
        data_list.append(data[batch][0][0][:min_len])
        label_list.append(data[batch][1])
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor)
    return data_copy