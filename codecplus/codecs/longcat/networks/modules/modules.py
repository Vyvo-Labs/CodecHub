import torch
import torch.nn as nn


class GLSTM(nn.Module):

    def __init__(self, in_features=None, out_features=None, hidden_size=896, groups=2):
        super().__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups
        self.hidden_size = hidden_size_t

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
        out = torch.chunk(out, self.groups, dim=-1)

        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1)).contiguous()

        out = out.transpose(1, 2).contiguous()

        return out
    

    def forward_stream(
        self, 
        x,
        list1_h_input, 
        list1_c_input, 
        list2_h_input, 
        list2_c_input, 
        lstm_buffer_input
    ):
        """
        Forward pass for streaming inference with LSTM states.
        
        Args:
            x: Input tensor
            list1_h_input: Previous hidden state for first LSTM layer
            list1_c_input: Previous cell state for first LSTM layer
            list2_h_input: Previous hidden state for second LSTM layer  
            list2_c_input: Previous cell state for second LSTM layer
            lstm_buffer_input: LSTM buffer input for context
            
        Returns:
            tuple: (output, list1_h, list1_c, list2_h, list2_c, lstm_buffer)
        """
        lstm_buffer_length = 8
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
        out = torch.chunk(out, self.groups, dim=-1)
        
        out,(list1_h, list1_c) = self.lstm_list1[0](out[0],(list1_h_input, list1_c_input))
       
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
       
        out,(list2_h, list2_c) = self.lstm_list2[0](out[0],(list2_h_input, list2_c_input))

        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1)).contiguous()

        out = out.transpose(1, 2).contiguous()
        
        lstm_buffer_update = torch.cat([lstm_buffer_input[:, :, :], out], dim=-1)
        
        out = lstm_buffer_update[:,:,-(lstm_buffer_length - 1)-out.size(2):]
        # only return last lstm_buffer_length frames as context
        out_lstm = lstm_buffer_update[:, :, -lstm_buffer_length:]
        
        return out, list1_h, list1_c, list2_h, list2_c, out_lstm