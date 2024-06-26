import torch.nn as nn
from torch.nn import Module
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_layers import LinearPack
from model_padding import mask_it
from model_attention import ScaledDotProductAttention


class LastElementExtractor(nn.Module): 
    def __init__(self): 
        super(LastElementExtractor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu = torch.device('cpu')
    
    def forward(self, packed, lengths): 
        lengths = torch.tensor(lengths, device=self.device)
        sum_batch_sizes = torch.cat((
            torch.zeros(2, dtype=torch.int64, device=self.device),
            torch.cumsum(packed.batch_sizes, 0).to(self.device)
        ))
        sorted_lengths = lengths[packed.sorted_indices]
        last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0), device=self.device)
        last_seq_items = packed.data[last_seq_idxs]
        last_seq_items = last_seq_items[packed.unsorted_indices]
        return last_seq_items







class Encoder(Module): 
    def __init__(self, a, size_list, in_size, in2_size, in3_size, hid_size):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.in2_size = in2_size
        self.in3_size = in3_size    # this is newly added for lin2
        self.hid_size = hid_size
        self.size_list = size_list
        # temp not use embed_in, directly go to RNN
        self.lin_1 = LinearPack(in_dim=in_size, out_dim=in2_size)
        self.lin_2 = LinearPack(in_dim=in2_size, out_dim=in3_size)
        self.rnn = HM_LSTM(a, in3_size, size_list)  # changed in_size to in2_size, since the actual data size is in2_size, instead of in_size
        # self.lin_2 = LinearPack(in_dim=size_list[1], out_dim=hid_size)

    def inits(self, batch_size, device):
        h_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
        c_t1 = torch.zeros((self.size_list[0], batch_size), dtype=torch.float, device=device, requires_grad=False)
        z_t1 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)
        h_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
        c_t2 = torch.zeros((self.size_list[1], batch_size), dtype=torch.float, device=device, requires_grad=False)
        z_t2 = torch.zeros((1, batch_size), dtype=torch.float, device=device, requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden

    def forward(self, inputs, in_mask, hidden):
        """
        Args:
            inputs: input data (B, L, I)
            in_mask: masking (B, L)
            hidden: HM_LSTM
        """
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        h_1, h_2, z_1, z_2, hidden = self.rnn(enc_x, hidden) # (B, L, I3) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)  # it seems that it might affect the outcome, so apply mask here as well
        # hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        # hid_r = mask_it(hid_r, in_mask)
        hid_r = h_2
        return (hid_r, z_1, z_2)
    
    def encode(self, inputs, in_mask, hidden): 
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, I2)
        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        h_1, h_2, z_1, z_2, hidden = self.rnn(enc_x, hidden) # (B, L, I3) -> (B, L, S0) -> (B, L, S1)
        h_2 = mask_it(h_2, in_mask)
        # hid_r = self.lin_2(h_2) # (B, L, S1) -> (B, L, H)
        # hid_r = mask_it(hid_r, in_mask)
        hid_r = h_2
        return (hid_r, z_1, z_2)

class Decoder(Module): 
    def __init__(self, size_list, in2_size, hid_size, out_size, num_layers=1):
        super(Decoder, self).__init__()
        self.in2_size = in2_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.size_list = size_list
        self.num_layers = num_layers

        self.lin_1 = LinearPack(in_dim=out_size, out_dim=size_list[1])  # NOTE: we use out size as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], in2_size, num_layers=self.num_layers, batch_first=True)  # only using h_2, therefore only size_list[1]
        self.attention = ScaledDotProductAttention(q_in=in2_size, kv_in=hid_size, qk_out=in2_size, v_out=in2_size)
        self.lin_2 = LinearPack(in_dim=in2_size, out_dim=out_size)

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.in2_size), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.in2_size), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.out_size), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights

class SimpleEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(SimpleEncoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[1], hidden_size=size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)

        return enc_x

class SimpleDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13   , 64, 16, 3]: similar to encoder, just layer 0 different
        super(SimpleDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[2]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[2]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights

class PhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(PhxLearner, self).__init__()

        self.encoder = SimpleEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = SimpleDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        
        return dec_out, attn_w
    
    def encode(self, inputs, in_mask): 
        batch_size = inputs.size(0)
        hidden = self.encoder.inits(batch_size=batch_size, device=self.device)
        return self.encoder.encode(inputs, in_mask, hidden)






class RLEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(RLEncoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[0], hidden_size=size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        # self.act = nn.Tanh()
        # self.bn = nn.BatchNorm1d(size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        # enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = inputs

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        # enc_x = self.act(enc_x)
        
        # enc_x = enc_x.permute(0, 2, 1)
        # enc_x = self.bn(enc_x)
        # enc_x = enc_x.permute(0, 2, 1)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        # enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = inputs

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        # enc_x = self.act(enc_x)

        # enc_x = enc_x.permute(0, 2, 1)
        # enc_x = self.bn(enc_x)
        # enc_x = enc_x.permute(0, 2, 1)

        return enc_x
    
class RLBiEncoder(Module): 
    """
    This time we only leave LSTM in the encoder. It is bidirectional, and we want, through this, make the model as simple as possible 
    and avoid any influence from external matters. 
    """
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(RLBiEncoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[0], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=0.5, bidirectional=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        # enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = inputs
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        # enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = inputs
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
class LBiREncoder(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(LBiREncoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
class LBiRInitDecoder(Module): 
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LBiRInitDecoder, self).__init__()
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=True)
        self.attention = ScaledDotProductAttention(q_in=2 * size_list[3], kv_in=2 * size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((2 * self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((2 * self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
    
class MockingjoyEncoder(Module): 
    """
    This time we only leave LSTM in the encoder. It is bidirectional, and we want, through this, make the model as simple as possible 
    and avoid any influence from external matters. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(MockingjoyEncoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, H)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, H) -> (B, L, H)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin_1(inputs) # (B, L, I) -> (B, L, H)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, H) -> (B, L, H)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
class MockingjoyDecoder(Module): 
    def __init__(self, size_list, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(MockingjoyDecoder, self).__init__()
        self.lin_1 = nn.Linear(size_list[3], size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(size_list[3])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin_1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.lin_2(x)
        return x
    

class RALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(RALDecoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[0], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        # dec_in_token = init_in
        dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            # dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_in_token, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
        # return outputs, None

class InitRALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(InitRALDecoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[0], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            # dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_in_token, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
    
class InitLRALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        super(InitLRALDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(size_list[1], size_list[3], num_layers=num_layers, batch_first=True)
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = nn.Linear(size_list[3], size_list[0])
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        length = hid_r.size(1) # get length
        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
    
class RleALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(RleALDecoder, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[0], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token =  self.lin_3(init_in)
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])

        outputs = []
        attention_weights = []
        for t in range(length):
            # dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_in_token, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights

class LHYEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(LHYEncoder, self).__init__()
        self.lin = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[1], hidden_size=size_list[3], num_layers=num_layers, batch_first=True)

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin(inputs) # (B, L, I2) -> (B, L, I3)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin(inputs) # (B, L, I2) -> (B, L, I3)

        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)

        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)

        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)

        return enc_x

class LHYDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LHYDecoder, self).__init__()
        self.rnn_in_dim = size_list[3]
        self.rnn_out_dim = size_list[1]

        self.rnn = nn.LSTM(self.rnn_in_dim, self.rnn_out_dim, num_layers=num_layers, batch_first=True)
        # self.lin = LinearPack(in_dim=size_list[1], out_dim=size_list[0])
        self.lin = nn.Linear(size_list[1], size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask, init_in, hidden, inputs_lens):
        dec_x = pack_padded_sequence(hid_r, inputs_lens, batch_first=True, enforce_sorted=False)

        dec_x, (hn, cn) = self.rnn(dec_x)  # (B, L, I1) -> (B, L, I2)

        dec_x, _ = pad_packed_sequence(dec_x, batch_first=True)

        dec_x = self.lin(dec_x)
        
        return dec_x
    
class LRLEncoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [39, 64, 16, 3]
        super(LRLEncoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])
        self.rnn = nn.LSTM(input_size=size_list[1], hidden_size=size_list[2], num_layers=num_layers, batch_first=True)
        self.lin_2 = nn.Linear(size_list[2], size_list[3])

    def forward(self, inputs, inputs_lens, in_mask=None, hidden=None):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        return enc_x
    
    def encode(self, inputs, inputs_lens, in_mask=None, hidden=None): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x) # (B, L, I2) -> (B, L, I3)
        return enc_x

class LRALDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LRALDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], size_list[3], num_layers=num_layers, batch_first=True)
        # self.lin_2 = LinearPack(in_dim=size_list[2], out_dim=size_list[3])
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        # self.lin_3 = LinearPack(in_dim=size_list[3], out_dim=size_list[0])
        self.lin_3 = nn.Linear(size_list[3], size_list[0])
        # self.act = nn.Tanh()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length
        # last_hid_r = self.last_element_extractor()

        # dec_in_token = init_in
        # dec_in_token = self.lin_3()
        dec_in_token = torch.zeros((hid_r.size(0), 1, self.size_list[0]), dtype=torch.float, device=hid_r.device, requires_grad=False)

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # dec_x = self.lin_2(dec_x)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
        # return outputs, None

class LRLDecoder(Module): 
    def __init__(self, size_list, num_layers=1):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LRLDecoder, self).__init__()
        self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[1])  # NOTE: we use out size (last in size_list) as input size to lin, because we will take the direct output from last layer in dec. 
        self.rnn = nn.LSTM(size_list[1], size_list[3], num_layers=num_layers, batch_first=True)
        self.lin_3 = nn.Linear(size_list[3], size_list[0])
        # self.act = nn.Tanh()
        self.last_element_extractor = LastElementExtractor()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, inputs_lens, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        # dec_in_token = init_in
        # dec_in_token = self.lin_3(hid_r[:, -2:-1, :])
        packed_hid_r = pack_padded_sequence(hid_r, inputs_lens, batch_first=True, enforce_sorted=False)
        last_hid_r = self.last_element_extractor(packed_hid_r, inputs_lens).unsqueeze(1)
        dec_in_token = self.lin_3(last_hid_r)

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x, hidden = self.rnn(dec_x, hidden)
            # dec_x = self.lin_2(dec_x)
            # dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_3(dec_x)
            # dec_x = self.act(dec_x)
            outputs.append(dec_x)
            # attention_weights.append(attention_weight)

            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        # attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        # attention_weights = attention_weights.squeeze(2)
        outputs = mask_it(outputs, in_mask)     # test! I think it should be correct. 
        
        return outputs, attention_weights
        # return outputs, None


class TwoLinPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(TwoLinPhxLearner, self).__init__()

        self.encoder = LRLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = LRALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    

class TwoLinNoAttentionPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(TwoLinNoAttentionPhxLearner, self).__init__()

        self.encoder = LRLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = LRLDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, input_lens, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)


class SimplerPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(SimplerPhxLearner, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = RALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)

class SimplerPhxLearnerInit(Module):
    # RL + RAL(Init)
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(SimplerPhxLearnerInit, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = InitRALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
class LBiR_LBiRInit(Module):
    # RL + RAL(Init)
    def __init__(self, enc_size_list, dec_size_list, num_layers=1, dropout=0.5):
        # input = (batch_size, time_steps, in_size); 
        super(LBiR_LBiRInit, self).__init__()

        self.encoder = LBiREncoder(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = LBiRInitDecoder(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
class PhonePredNet(Module): 
    # RL + L
    def __init__(self, enc_size_list, out_dim, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(PhonePredNet, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = nn.Linear(enc_size_list[3], out_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        enc_out = self.encoder(inputs, input_lens, in_mask)
        pred_out = self.decoder(enc_out)
        return pred_out
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
    def predict_on_output(self, output): 
        output = nn.Softmax(dim=-1)(output)
        preds = torch.argmax(output, dim=-1)
        return preds
    
class PhonePredCTCNet(Module): 
    # RL + L
    def __init__(self, enc_size_list, out_dim, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(PhonePredCTCNet, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = nn.Linear(enc_size_list[3], out_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        enc_out = self.encoder(inputs, input_lens, in_mask)
        pred_out = self.decoder(enc_out)
        pred_out = self.softmax(pred_out)
        return pred_out
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
    def predict_on_output(self, output): 
        # output = nn.Softmax(dim=-1)(output)
        preds = torch.argmax(output, dim=-1)
        return preds
    
class PhonePredCTCBiNet(Module): 
    # RL + L
    def __init__(self, enc_size_list, out_dim, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(PhonePredCTCBiNet, self).__init__()

        self.encoder = RLBiEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = nn.Linear(2 * enc_size_list[3], out_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        enc_out = self.encoder(inputs, input_lens, in_mask)
        pred_out = self.decoder(enc_out)
        pred_out = self.softmax(pred_out)
        return pred_out
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
    def predict_on_output(self, output): 
        # output = nn.Softmax(dim=-1)(output)
        preds = torch.argmax(output, dim=-1)
        return preds
    
class ReconMockingjoyNet(Module): 
    # RL + L
    def __init__(self, enc_size_list, dec_size_list, num_layers=1, dropout=0.5):
        # input = (batch_size, time_steps, in_size); 
        super(ReconMockingjoyNet, self).__init__()

        self.encoder = MockingjoyEncoder(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = MockingjoyDecoder(size_list=dec_size_list, dropout=dropout)


    def forward(self, inputs, input_lens, in_mask):
        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out = self.decoder(enc_out)
        return dec_out
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)

    

class LRLInitLRALNet(Module):
    # LRL + LRAL(Init)
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(LRLInitLRALNet, self).__init__()

        self.encoder = LRLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = InitLRALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)
        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)
    
class SimplerPhxLearnerle(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(SimplerPhxLearnerle, self).__init__()

        self.encoder = RLEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.last_element_extractor = LastElementExtractor()
        self.decoder = RleALDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        packed_hid_r = pack_padded_sequence(enc_out, input_lens, batch_first=True, enforce_sorted=False)
        last_hid_r = self.last_element_extractor(packed_hid_r, input_lens).unsqueeze(1)
        init_in = last_hid_r
        dec_out, attn_w = self.decoder(enc_out, in_mask, init_in, dec_hid)
        return dec_out, attn_w
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)



class LHYPhxLearner(Module):
    def __init__(self, enc_size_list, dec_size_list, num_layers=1):
        # input = (batch_size, time_steps, in_size); 
        super(LHYPhxLearner, self).__init__()

        self.encoder = LHYEncoder(size_list=enc_size_list, num_layers=num_layers)
        self.decoder = LHYDecoder(size_list=dec_size_list, num_layers=num_layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)

        enc_out = self.encoder(inputs, input_lens, in_mask)
        dec_out = self.decoder(enc_out, in_mask, None, None, input_lens)
        
        return dec_out, None
    
    def encode(self, inputs, input_lens, in_mask): 
        return self.encoder(inputs, input_lens, in_mask)




################ New Code Place [since 20240216] #######################################################################
class VAEEncoderV1(Module): 
    """
    VAE V1: Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VAEEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)

    def forward(self, inputs, inputs_lens):
        """
        Args:
            inputs: input data (B, L, I)
            inputs_lens: input lengths
            in_mask: masking (B, L), abolished, since now we have packing and padding
            hidden: HM_LSTM, abolished
        """
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x
    
class VAEDecoderV1(Module): 
    """
    VAE V1: Bidirectional LSTM + Linear; no attention. 
            This is not autoregressive, because we are making it predict each timestep only. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VAEDecoderV1, self).__init__()
        # self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=True)
        # self.attention = ScaledDotProductAttention(q_in=2 * size_list[3], kv_in=2 * size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[0])

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, lengths): 
        dec_x = hid_r
        dec_x = pack_padded_sequence(dec_x, lengths, batch_first=True, enforce_sorted=False)
        dec_x, (hn, cn) = self.rnn(dec_x)  # (B, L, I1) -> (B, L, I2)
        dec_x, _ = pad_packed_sequence(dec_x, batch_first=True)
        dec_x = self.lin_2(dec_x)
        return dec_x
    
class VAENetV1(Module):
    # RL + RAL(Init) [VAE]
    def __init__(self, enc_size_list, dec_size_list, num_layers=1, dropout=0.5):
        # input = (batch_size, time_steps, in_size); 
        super(VAENetV1, self).__init__()

        self.encoder = VAEEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VAEDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # we take in bi-directional hidrs, and output only 8-dimensional
        self.mu_lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])
        self.logvar_lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, inputs, input_lens):
        enc_out = self.encoder(inputs, input_lens)

        mu = self.mu_lin(enc_out)
        logvar = self.logvar_lin(enc_out)
        z = self.reparameterise(mu, logvar)

        dec_out = self.decoder(z, input_lens)
        return dec_out, (mu, logvar)
    
    def encode(self, inputs, input_lens): 
        enc_out = self.encoder(inputs, input_lens)
        mu = self.mu_lin(enc_out)
        logvar = self.logvar_lin(enc_out)
        return mu, logvar
    

class VAENetV1a(Module):
    # RL + RAL(Init) [non-VAE]
    def __init__(self, enc_size_list, dec_size_list, num_layers=1, dropout=0.5):
        # input = (batch_size, time_steps, in_size); 
        super(VAENetV1a, self).__init__()

        self.encoder = VAEEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VAEDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # we take in bi-directional hidrs, and output only 8-dimensional
        self.lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])
        # self.logvar_lin = nn.Linear(enc_size_list[3] * 2, enc_size_list[3])

    # def reparameterise(self, mu, logvar):
    #     epsilon = torch.randn_like(mu)
    #     return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, inputs, input_lens):
        enc_out = self.encoder(inputs, input_lens)

        # mu = self.mu_lin(enc_out)
        # logvar = self.logvar_lin(enc_out)
        # z = self.reparameterise(mu, logvar)

        z = self.lin(enc_out)

        dec_out = self.decoder(z, input_lens)
        return dec_out, None
    
    def encode(self, inputs, input_lens): 
        enc_out = self.encoder(inputs, input_lens)
        # mu = self.mu_lin(enc_out)
        # logvar = self.logvar_lin(enc_out)
        z = self.lin(enc_out)
        return z, None
    


################################# VQVAE #################################
class VQEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VQEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])

        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(-1)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
class VQDecoderV1(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VQDecoderV1, self).__init__()
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])

        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights

class VQVAEV1(Module):
    # RL + RAL(Init)
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(VQVAEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.vq_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        self.vq_embedding.weight.data.uniform_(-1.0 / embedding_dim,
                                               1.0 / embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)

        # now finding the closest vector
        # ze: [B, L, C]
        # embedding: [K, C]
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)
        # stop gradient
        dec_in = ze + (zq - ze).detach()

        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)

        return ze, zq
    

class VQPhonePredV1(Module): 
    # RL + L
    def __init__(self, in_dim, out_dim):
        # input = (batch_size, time_steps, in_size); 
        super(VQPhonePredV1, self).__init__()
        self.decoder = nn.Linear(in_dim, out_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, z):
        pred_out = self.decoder(z)
        pred_out = self.softmax(pred_out)
        return pred_out
    
    def predict_on_output(self, output): 
        preds = torch.argmax(output, dim=-1)
        return preds


############################ CTC Predictor ############################
class VQEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(VQEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])

        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(-1)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
    def encode(self, inputs, inputs_lens): 
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        enc_x = self.lin_2(enc_x)   # this merges the bidirectional into one.
        return enc_x
    
class VQDecoderV1(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VQDecoderV1, self).__init__()
        self.lin_1 = nn.Linear(size_list[0], size_list[3])
        self.rnn = nn.LSTM(input_size=size_list[3], hidden_size=size_list[3], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.lin_2 = nn.Linear(size_list[3], size_list[0])

        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list[3]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list[0]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x)
            attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights

class VQVAEV1(Module):
    # RL + RAL(Init)
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(VQVAEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.vq_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        self.vq_embedding.weight.data.uniform_(-1.0 / embedding_dim,
                                               1.0 / embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)

        # now finding the closest vector
        # ze: [B, L, C]
        # embedding: [K, C]
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)
        # stop gradient
        dec_in = ze + (zq - ze).detach()

        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        embedding = self.vq_embedding.weight.data
        B, L, C = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, 1, K, C)
        ze_broadcast = ze.reshape(B, L, 1, C)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, -1) # (B, L, K)
        nearest_neighbor = torch.argmin(distance, -1)   # (B, L)

        zq = self.vq_embedding(nearest_neighbor)    # (B, L, C)

        return ze, zq