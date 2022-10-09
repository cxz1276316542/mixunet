import numpy as np
import paddle

x = paddle.to_tensor([[0.2], [0.2],[0.8]], dtype='float32')
print (x.shape)
y = paddle.to_tensor([1,0,0,0], dtype='float32')
print (y.shape)


