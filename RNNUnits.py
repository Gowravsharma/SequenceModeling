import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_UNIT(nn.Module):
  def __init__(self, at_dim, xt_dim, output = True, output_size = None):
    super(RNN_UNIT, self).__init__()
    self.output = output
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
    return at

class GRU_UNIT(nn.Module):
  def __init__(self, hidden_dim, input_dim, output=True, output_size=None):
    super(GRU_UNIT, self).__init__()
    self.output = output
    # Update gate weights
    self.w_xz = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_hz = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_z = nn.Parameter(torch.zeros(hidden_dim))

    # Reset gate weights
    self.w_xr = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_hr = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_r = nn.Parameter(torch.zeros(hidden_dim))

    # Candidate hidden state weights
    self.w_xh = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_h = nn.Parameter(torch.zeros(hidden_dim))

    # Output weights
    if output_size is not None:
      self.w_hy = nn.Parameter(torch.empty(output_size, hidden_dim))
      self.b_y = nn.Parameter(torch.zeros(output_size))
    
    # Initialization
    for w in [self.w_xz, self.w_xr, self.w_xh]:
      nn.init.xavier_uniform_(w)
    for w in [self.w_hz, self.w_hr, self.w_hh]:
      nn.init.orthogonal_(w)
    if output_size is not None:
      nn.init.xavier_uniform_(self.w_hy)

  def forward(self, prev_h, x_t):
    z_t = torch.sigmoid(torch.matmul(self.w_xz, x_t) + torch.matmul(self.w_hz, prev_h) + self.b_z)
    r_t = torch.sigmoid(torch.matmul(self.w_xr, x_t) + torch.matmul(self.w_hr, prev_h) + self.b_r)
    h_hat = torch.tanh(torch.matmul(self.w_xh, x_t) + torch.matmul(self.w_hh, r_t * prev_h) + self.b_h)
    h_t = (1 - z_t) * prev_h + z_t * h_hat

    if self.output:
      y_t = torch.matmul(self.w_hy, h_t) + self.b_y
      y_t = torch.softmax(y_t, dim=-1)
      return h_t, y_t
    return h_t

class LSTM_UNIT(nn.Module):
  def __init__(self, hidden_dim, input_dim, output=True, output_size=None):
    super(LSTM_UNIT, self).__init__()
    self.output = output
    # Forget gate
    self.w_xf = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_hf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_f = nn.Parameter(torch.zeros(hidden_dim))

    # Input gate
    self.w_xi = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_hi = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_i = nn.Parameter(torch.zeros(hidden_dim))

    # Candidate cell state
    self.w_xg = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_hg = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_g = nn.Parameter(torch.zeros(hidden_dim))

    # Output gate
    self.w_xo = nn.Parameter(torch.empty(hidden_dim, input_dim))
    self.w_ho = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
    self.b_o = nn.Parameter(torch.zeros(hidden_dim))

    # Output weights
    if output_size is not None:
      self.w_hy = nn.Parameter(torch.empty(output_size, hidden_dim))
      self.b_y = nn.Parameter(torch.zeros(output_size))
    
    # Initialization
    for w in [self.w_xf, self.w_xi, self.w_xg, self.w_xo]:
      nn.init.xavier_uniform_(w)
    for w in [self.w_hf, self.w_hi, self.w_hg, self.w_ho]:
        nn.init.orthogonal_(w)
    if output_size is not None:
      nn.init.xavier_uniform_(self.w_hy)

  def forward(self, prev_h, prev_c, x_t):
    f_t = torch.sigmoid(torch.matmul(self.w_xf, x_t) + torch.matmul(self.w_hf, prev_h) + self.b_f)
    i_t = torch.sigmoid(torch.matmul(self.w_xi, x_t) + torch.matmul(self.w_hi, prev_h) + self.b_i)
    g_t = torch.tanh(torch.matmul(self.w_xg, x_t) + torch.matmul(self.w_hg, prev_h) + self.b_g)
    c_t = f_t * prev_c + i_t * g_t
    o_t = torch.sigmoid(torch.matmul(self.w_xo, x_t) + torch.matmul(self.w_ho, prev_h) + self.b_o)
    h_t = o_t * torch.tanh(c_t)

    if self.output:
      y_t = torch.matmul(self.w_hy, h_t) + self.b_y
      y_t = torch.softmax(y_t, dim=-1)
      return h_t, c_t, y_t
    return h_t, c_t

