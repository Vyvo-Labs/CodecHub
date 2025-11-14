import torch
import torch.nn as nn

from codecplus.codecs.longcat.networks.modules.model import DecoderBlock, WNConv1d, Snake1d
from codecplus.codecs.longcat.networks.modules.modules import GLSTM
from codecplus.codecs.longcat.networks.modules.causal_model import CausalDecoderBlock


class MemDecoder(nn.Module):

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        groups: int = 1,
        lookahead_frame: int = 3,
        lstm_nums: int = 1,
        is_final_causal: bool = True,
    ):
        super().__init__()

        # [B, dim, T]
        # add mem layer here. keep shape.
        self.mem_layers = nn.ModuleList([GLSTM(groups=groups, hidden_size=input_channel) for _ in range(lstm_nums)])
        
        # Add first conv layer
        kernel_size = 2 * lookahead_frame + 1
        layers = [WNConv1d(input_channel, channels, kernel_size=kernel_size, padding=lookahead_frame)]
        
        # Add final conv layer
        if is_final_causal:
            for i, stride in enumerate(rates):
                input_dim = channels // 2**i
                output_dim = channels // 2**(i + 1)
                layers += [CausalDecoderBlock(input_dim, output_dim, stride)]

            layers += [
                Snake1d(output_dim),
                nn.ZeroPad1d((6, 0)),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        else:
            # Add upsampling + MRF blocks
            for i, stride in enumerate(rates):
                input_dim = channels // 2**i
                output_dim = channels // 2**(i + 1)
                layers += [DecoderBlock(input_dim, output_dim, stride)]

            layers += [
                Snake1d(output_dim),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
                nn.Tanh(),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        for mem_layer in self.mem_layers:
            x = mem_layer(x)
        x = self.model(x)

        return x

    
    def forward_stream(
        self, 
        x,
        last_c0_mem1, 
        last_h0_mem1, 
        last_c0_mem2, 
        last_h0_mem2, 
        lstm_buffer_input
    ):
        """
        Forward pass for streaming inference with LSTM memory states.
        
        Args:
            x: Input tensor
            last_c0_mem1: Previous cell state for first LSTM memory
            last_h0_mem1: Previous hidden state for first LSTM memory  
            last_c0_mem2: Previous cell state for second LSTM memory
            last_h0_mem2: Previous hidden state for second LSTM memory
            lstm_buffer_input: LSTM buffer input for streaming
            
        Returns:
            tuple: (output, list1_h, list1_c, list2_h, list2_c, lstm_buffer)
        """

        # Process through first memory layer with streaming
        x, list1_h, list1_c, list2_h, list2_c, lstm_buffer = self.mem_layers[0].forward_stream(
            x, last_c0_mem1, last_h0_mem1, last_c0_mem2, last_h0_mem2, lstm_buffer_input
        )
        
        # Process through the main model
        x = self.model(x)
        
        return x, list1_h, list1_c, list2_h, list2_c, lstm_buffer


if __name__ == "__main__":
    x = torch.randn([8, 1024, 1])
    model = MemDecoder(1024, 1536, [8, 5, 4, 2])
    y = model(x)
    print(y.shape)