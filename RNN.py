import torch
import torch.nn as nn

class RNN_UNIT(nn.Module):
  def __init__(self, at_dim, xt_dim, output = None, output_size = None):
    super(RNN_UNIT, self).__init__()
    self.output = output if output is not None else True
    # input weights
    self.w_xa = nn.Parameter(torch.empty(at_dim, xt_dim))
    nn.init.xavier_uniform_(self.w_xa)
    # hidden weights
    self.w_aa = nn.Parameter(torch.empty(at_dim, at_dim))
    nn.init.orthogonal_(self.w_aa)
    # output weights
    if output_size is not None:
      self.w_ay = nn.Parameter(torch.empty(output_size, at_dim))
      nn.init.xavier_uniform_(self.w_ay)
      self.b_y = nn.Parameter(torch.zeros(output_size))  
    # baises
    self.b_a = nn.Parameter(torch.zeros(at_dim))

  def forward(self, prev_at, input_xt, activation = torch.tanh):
    at = torch.matmul(self.w_xa, input_xt) + torch.matmul(self.w_aa, prev_at) + self.b_a
    at = activation(at)

    if self.output:
      yt = torch.matmul(self.w_ay, at) + self.b_y
      yt = torch.softmax(yt , dim = -1)
      return at, yt
    return at, yt
  
