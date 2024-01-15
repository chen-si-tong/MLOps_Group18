import numpy as np


# class about get batched train data and validation data 

class DataSampler(object):
    """
    DataSampler is used for obtaining batched train data and validation data.

    Args:
    - inputs: Input data for training or validation.
    - batch_size: The batch size for sampling data.

    Attributes:
    - inputs: Transposed and stacked input data.
    - batch_size: The specified batch size.
    - num_cols: Number of columns in the input data.
    - len: Length of the input data.

    Methods:
    - __len__: Returns the length of the input data.
    - __iter__: Returns the data sampler itself.
    - __next__: Returns the next batch of data.
    - next: Returns the next batch of data.

    Usage:
    - Create an instance of DataSampler to sample batches of training or validation data.
    - Iterate over the DataSampler to obtain batches of data.
    """
    def __init__(self, inputs, batch_size=64):
        self.inputs = inputs 
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))
    def __len__(self):
        return self.len 
    def __iter__(self):
        return self   
    def __next__(self): 
        return self.next() 
    def next(self): 
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :] 
        return [out[:, i] for i in range(self.num_cols)]
    
#  class about get batched test data 
class DataSamplerForTest(DataSampler):
    """
    DataSamplerForTest is used for obtaining batched test data.

    Args:
    - inputs: Input data for testing.
    - batch_size: The batch size for testing data. If batch_size is positive, data will be split into batches.

    Attributes:
    - idx_group: List of indices indicating the data groups.
    - group_id: Current group index.

    Methods:
    - next: Retrieves the next batch of test data.

    Usage:
    - Create an instance of DataSamplerForTest to sample test data in batches for testing purposes.
    """
    def __init__(self, inputs, batch_size=64):
        super(DataSamplerForTest, self).__init__(inputs, batch_size=batch_size)        
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)] 
            
        self.group_id = 0
    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration 
                
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]


