import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P




class DeulingNet(nn.Cell):

    def __init__(self, input_size, hidden_size, A_output_size, compute_type=mstype.float32):
        super(FullyConnectedNet, self).__init__()
        self.linear1 = nn.Dense(
            input_size,
            hidden_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.linear2 = nn.Dense(
            hidden_size,
            hidden_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.V_linear = nn.Dense(
            hidden_size,
            1,
            weight_init="XavierUniform").to_float(compute_type)
        self.A_linear = nn.Dense(
            hidden_size,
            output_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.relu = nn.ReLU()
        self.cast = P.Cast()
        self.max = P.max()
        self.sigmoid = nn.Sigmoid()
    def construct(self, x):
        feature = self.relu(self.linear2(self.relu(self.linear1(x))))
        V = self.V_linear(feature)
        A = self.A_linear(feature)
        output = self.max(A) + V
        output = self.cast(output, mstype.float32)
        return output