import torch.nn as nn
from torch.nn import Module
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_layers import LinearPack
from model_padding import mask_it
from model_attention import ScaledDotProductAttention
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
    
    
class VQDecoderV2(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    同时也输出attention_out
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(VQDecoderV2, self).__init__()
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
            # 这里不能detach，因为training中还要做backprop
            outputs.append(dec_x)
            attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        # Attention decoder
        length = hid_r.size(1) # get length

        dec_in_token = init_in

        outputs = []
        attn_outs = []
        attention_weights = []
        for t in range(length):
            dec_x = self.lin_1(dec_in_token)
            dec_x = self.act(dec_x)
            dec_x, hidden = self.rnn(dec_x, hidden)
            dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            # 我的理解是：虽然如果不clone，list里面的tensor也会跟着变
            # 但是因为这里记录的dec_x在后来都会进入网络中，网络是自带clone的，所以这里不clone也没问题
            attn_outs.append(dec_x.clone())
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x.clone())
            attention_weights.append(attention_weight.clone())
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attn_outs = torch.stack(attn_outs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attn_outs = attn_outs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attn_outs, attention_weights

class EncoderV1(Module): 
    """
    Linear * 1 + Unidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(EncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])
        self.rnn = nn.LSTM(input_size=size_list["lin1"], hidden_size=size_list["hid"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=False)
        # self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])
        self.act = nn.ReLU()

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
        enc_x, (hn, cn) = self.rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
        enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
        return enc_x, (hn, cn)
    
    def encode(self, inputs, inputs_lens): 
        return self.forward(inputs, inputs_lens)
    
class EncoderV2(Module): 
    """
    Linear * 1 + Unidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(EncoderV2, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])

        self.rnns = nn.ModuleList()
        for i in range(num_layers): 
            # TODO: currently we don't support changing dimensions. 
            self.rnns.append(nn.LSTM(input_size=size_list["rnn_in"], hidden_size=size_list["rnn_out"], 
                                     batch_first=True, bidirectional=False))
        # self.lin_2 = nn.Linear(size_list[3] * 2, size_list[3])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for rnn in self.rnns: 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x = self.dropout(enc_x)
        
        return enc_x, (hn, cn)  # so obviously we are returning the last output. 
    
    def encode(self, inputs, inputs_lens): 
        outs = []
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for rnn in self.rnns: 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = rnn(enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            outs.append(enc_x)
        return outs, (hn, cn)  # so obviously we are returning the last output. 
    

class EncoderV3(Module): 
    """
    Linear + Bidirectional LSTM + Linear (merge bidirectional output)
    注意！因爲這次是bidirectional，且用ModuleList來層曡LSTM，所以每一層都需要merge. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(EncoderV3, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["rnn_in"])

        self.rnns = nn.ModuleList()
        self.mergers = nn.ModuleList()
        for i in range(num_layers): 
            # TODO: currently we don't support changing dimensions. 
            # NOTE: now rnn_in === rnn_out
            self.rnns.append(nn.LSTM(input_size=size_list["rnn_in"], hidden_size=size_list["rnn_out"], 
                                     batch_first=True, bidirectional=True))
            self.mergers.append(nn.Linear(size_list["rnn_out"] * 2, size_list["rnn_in"]))
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, inputs_lens):
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for i in range(len(self.rnns)): 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = self.rnns[i](enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x = self.dropout(enc_x)
            enc_x = self.mergers[i](enc_x)
        return enc_x, (hn, cn)  # so obviously we are returning the last output. 
    
    def encode(self, inputs, inputs_lens): 
        outs = []
        enc_x = self.lin_1(inputs) # (B, L, I0) -> (B, L, I1)
        enc_x = self.act(enc_x)
        for i in range(len(self.rnns)): 
            enc_x = pack_padded_sequence(enc_x, inputs_lens, batch_first=True, enforce_sorted=False)
            enc_x, (hn, cn) = self.rnns[i](enc_x)  # (B, L, I1) -> (B, L, I2)
            enc_x, _ = pad_packed_sequence(enc_x, batch_first=True)
            enc_x = self.dropout(enc_x)
            enc_x = self.mergers[i](enc_x)
            outs.append(enc_x)
        return outs, (hn, cn)  # so obviously we are returning the last output. 
    

class GoodDecoderV2(nn.Module):
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(GoodDecoderV2, self).__init__()
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])
        self.rnn = nn.LSTM(input_size=size_list["lin1"], hidden_size=size_list["hid"], 
                            batch_first=True, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin_2 = nn.Linear(size_list["hid"], size_list["in"])
        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list["in"]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, _ = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []
        attention_outs = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, attn_out = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            attention_outs.append(attn_out)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_outs = torch.stack(attention_outs, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        attention_outs = attention_outs.squeeze(2)
        return outputs, attention_weights, attention_outs

    def forward_step(self, input_item, hidden, encoder_outputs, encoder_mask):
        output = self.lin_1(input_item)
        output = self.act(output)
        output, hidden = self.rnn(output, hidden)
        attn_out, attention_weight = self.attention(output, encoder_outputs, 
                                                  encoder_outputs, encoder_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
        output = output + attn_out  # This is mimicking Transformer
        output = self.lin_2(output)
        return output, hidden, attention_weight, attn_out

class DecoderV2(nn.Module):
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(DecoderV2, self).__init__()
        self.lin_1 = nn.Linear(size_list["in"], size_list["lin1"])
        self.rnn = nn.LSTM(input_size=size_list["lin1"], hidden_size=size_list["hid"], 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=False)
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin_2 = nn.Linear(size_list["hid"], size_list["in"])
        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list["hid"]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list["in"]), dtype=torch.float, device=device, requires_grad=False)
        return hidden, dec_in_token

    def forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, _ = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        return outputs, attention_weights
    
    def attn_forward(self, hid_r, in_mask, init_in, hidden):
        batch_size, max_length, _ = hid_r.size()
        dec_in_token = init_in
        outputs = []
        attention_weights = []
        attention_outs = []

        for i in range(max_length): 
            dec_out_token, hidden, attention_weight, attn_out = self.forward_step(dec_in_token, hidden, hid_r, in_mask)
            outputs.append(dec_out_token)
            attention_weights.append(attention_weight)
            attention_outs.append(attn_out)
            dec_in_token = dec_out_token.detach()

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_outs = torch.stack(attention_outs, dim=1)
        outputs = outputs.squeeze(2)
        attention_weights = attention_weights.squeeze(2)
        attention_outs = attention_outs.squeeze(2)
        return outputs, attention_weights, attention_outs

    def forward_step(self, input_item, hidden, encoder_outputs, encoder_mask):
        output = self.lin_1(input_item)
        output = self.act(output)
        output, hidden = self.rnn(output, hidden)
        output, attention_weight = self.attention(output, encoder_outputs, 
                                                  encoder_outputs, encoder_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
        attn_out = output
        output = self.lin_2(output)
        return output, hidden, attention_weight, attn_out
    

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


############################ Normal AE with same structure as VQVAE ############################
class AEV1(Module):
    # RL + RAL(Init)
    # AEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(AEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq


############################ Normal AE + word-id with same structure as VQVAE ############################
class WIDAEV1(Module):
    # RL + RAL(Init)
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(WIDAEV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.word_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        self.combine_layer = nn.Linear(2 * enc_size_list[3], enc_size_list[3])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask, word_info):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        dec_in = zq
        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask, word_info): 
        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        return ze, zq   # !!! Check the use of ze and zq in later stages. Don't mix!!!
    
class WIDAEV2(Module):
    # Fixed word embedding
    # RL + RAL(Init)
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(WIDAEV2, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.word_embedding = nn.Embedding(embedding_dim, enc_size_list[3])
        torch.nn.init.orthogonal_(self.word_embedding.weight)
        self.word_embedding.weight.requires_grad = False  # Set trainable to False
        self.combine_layer = nn.Linear(2 * enc_size_list[3], enc_size_list[3])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask, word_info):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        dec_in = zq
        dec_out, attn_w = self.decoder(dec_in, in_mask, init_in, dec_hid)
        return dec_out, attn_w, (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask, word_info): 
        ze = self.encoder(inputs, input_lens)
        word_emb = self.word_embedding(word_info)
        word_emb = word_emb.unsqueeze(1).repeat_interleave(ze.shape[1], dim=1)
        cat_ze = torch.cat([ze, word_emb], -1)
        zq = self.combine_layer(cat_ze) 
        return ze, zq   # !!! Check the use of ze and zq in later stages. Don't mix!!!
    

############################ Multitask Learning (20240524) ############################
class CTCDecoderV2(Module): 
    """
    注意：decoder is only for classification for CTC pred.
    No LSTM is used here, but attention is used here as well.  
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(CTCDecoderV2, self).__init__()
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin = nn.Linear(size_list["hid"], size_list["class"])
        self.softmax = nn.LogSoftmax(dim=-1)
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask):
        outputs, attention_weights = self.attention(hid_r, hid_r, hid_r, in_mask.unsqueeze(1))
        outputs = self.lin(outputs)
        outputs = self.softmax(outputs)
        return outputs, attention_weights
    
class LinAttnDecoder(Module): 
    """
    NOTE: General purpose Linear + Attention Decoder. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(LinAttnDecoder, self).__init__()
        self.attention = ScaledDotProductAttention(q_in=size_list["in"], kv_in=size_list["in"], qk_out=size_list["in"], v_out=size_list["in"])
        self.lin = nn.Linear(size_list["in"], size_list["out"])
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask):
        outputs, attention_weights = self.attention(hid_r, hid_r, hid_r, in_mask.unsqueeze(1))
        outputs = self.lin(outputs)
        return outputs, attention_weights
    
class AEPPV1(Module):
    # Autoencoder + phoneme prediction
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV1, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
class AEPPV2(Module):
    # Autoencoder + phoneme prediction
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    # NOTE: test version. Only train with phone prediction
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV2, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        # self.ae_decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        # dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        # ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        # return follows: dec_out, attn_w, z
        # TODO: tomorrow just write the trining loop. 
        return (pp_dec_out, pp_dec_out), (pp_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
class AEPPV3(Module):
    # Reconstruction + phoneme prediction
    # Reconstruction uses the same linear + attention structure as PP. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV3, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = LinAttnDecoder(size_list={"in": dec_size_list[3], "out": dec_size_list[0]}, num_layers=num_layers, dropout=dropout)
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq

class AEPPV4(Module):
    # Autoencoder + phoneme prediction
    # WIDAEV1 also returns ze and zq, just to make it consistent with VQVAE
    # This is just a normal autoencoder, but with the same structure as VQVAE
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV4, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
class AEPPV5(Module):
    # Only Recon
    # use encoder hidden as decoder hidden. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV5, self).__init__()

        self.encoder = EncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = DecoderV2(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
    def attn_encode(self, inputs, input_lens, in_mask): 
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)
        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w, attn_z = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        return (ze, zq, attn_z), ae_dec_out, ae_attn_w
    
class AEPPV6(Module):
    # Only Recon
    # use encoder hidden as decoder hidden. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV6, self).__init__()

        self.encoder = EncoderV2(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = GoodDecoderV2(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze, enc_hid = self.encoder(inputs, input_lens)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        zqs = zes
        return zes, zqs
    
    def attn_encode(self, inputs, input_lens, in_mask): 
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)
        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        zqs = zes
        dec_in = zes[-1]
        ae_dec_out, ae_attn_w, attn_z = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, enc_hid)  # use enc_hid instead of zero hid
        return (zes, zqs, attn_z), ae_dec_out, ae_attn_w
    

class AEPPV7(Module):
    # Only PP
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV7, self).__init__()
        self.encoder = EncoderV3(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)

        ze, enc_hid = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        return (pp_dec_out, pp_dec_out), (pp_attn_w, pp_attn_w), (ze, zq)
    
    def fruitfulforward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)

        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zqs = zes
        dec_in = zes[-1]    # use the last LSTM layer's hidden representation
        pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        return (pp_dec_out, pp_dec_out), (pp_attn_w, pp_attn_w), (zes, zqs)
    
    def encode(self, inputs, input_lens, in_mask): 
        zes, enc_hid = self.encoder.encode(inputs, input_lens)
        zqs = zes
        return zes, zqs
    

class AEPPV8(Module):
    # 在4的基础上增加了attn_forward
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AEPPV8, self).__init__()

        self.encoder = VQEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = VQDecoderV2(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        # phoneme prediction decoder, this one is not auto-regressive, therefore we can use bidirectional
        # LSTM to enhance performance. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq
    
    def attn_forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        dec_hid, init_in = self.ae_decoder.inits(batch_size=batch_size, device=self.device)

        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, attn_out, ae_attn_w = self.ae_decoder.attn_forward(dec_in, in_mask, init_in, dec_hid)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, attn_out), (ae_attn_w, pp_attn_w), (ze, zq)

################################ Chung's AE Replication ################################
class AEEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        super(AEEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["out_lin1"])
        self.rnn = nn.LSTM(input_size=size_list["out_lin1"], hidden_size=size_list["out_rnn"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list["out_rnn"] * 2, size_list["out_lin2"])

        self.act = nn.ReLU()

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
    
class AEDecoderV1(Module): 
    """
    注意：decoder是自回归的，因而无需bidirectional
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(AEDecoderV1, self).__init__()
        self.lin_1 = nn.Linear(size_list["in"], size_list["out_lin1"])
        self.rnn = nn.LSTM(input_size=size_list["out_lin1"], hidden_size=size_list["out_rnn"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=False)
        self.lin_2 = nn.Linear(size_list["out_rnn"], size_list["out_lin2"])
        # self.attention = ScaledDotProductAttention(q_in=size_list[3], kv_in=size_list[3], qk_out=size_list[3], v_out=size_list[3])
        self.act = nn.ReLU()

        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def inits(self, batch_size, device): 
        h0 = torch.zeros((self.num_layers, batch_size, self.size_list["out_rnn"]), dtype=torch.float, device=device, requires_grad=False)
        c0 = torch.zeros((self.num_layers, batch_size, self.size_list["out_rnn"]), dtype=torch.float, device=device, requires_grad=False)
        hidden = (h0, c0)
        dec_in_token = torch.zeros((batch_size, 1, self.size_list["in"]), dtype=torch.float, device=device, requires_grad=False)
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
            # dec_x, attention_weight = self.attention(dec_x, hid_r, hid_r, in_mask.unsqueeze(1))    # unsqueeze mask here for broadcast
            dec_x = self.lin_2(dec_x)
            outputs.append(dec_x)
            # attention_weights.append(attention_weight)
            # Use the current output as the next input token
            dec_in_token = dec_x

        outputs = torch.stack(outputs, dim=1)   # stack along length dim
        # attention_weights = torch.stack(attention_weights, dim=1)
        outputs = outputs.squeeze(2)
        # attention_weights = attention_weights.squeeze(2)
        attention_weights = None
        return outputs, attention_weights


class AE_Chung(Module):
    # Reconstruction + phoneme prediction
    # Reconstruction uses the same linear + attention structure as PP. 
    def __init__(self, enc_size_list, dec_size_list, ctc_decoder_size_list, num_layers=1, dropout=0.5):
        super(AE_Chung, self).__init__()

        self.encoder = AEEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.ae_decoder = LinAttnDecoder(size_list={"in": dec_size_list[3], "out": dec_size_list[0]}, num_layers=num_layers, dropout=dropout)
        # self.pp_decoder = CTCDecoderV2(size_list=ctc_decoder_size_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
        batch_size = inputs.size(0)
        ze = self.encoder(inputs, input_lens)
        # concatenate hidden representation and word embedding. Then go through a linear layer (= combine)
        zq = ze
        dec_in = ze
        ae_dec_out, ae_attn_w = self.ae_decoder(dec_in, in_mask)
        # pp_dec_out, pp_attn_w = self.pp_decoder(dec_in, in_mask)
        pp_dec_out, pp_attn_w = ae_dec_out, ae_attn_w
        # return follows: dec_out, attn_w, z
        return (ae_dec_out, pp_dec_out), (ae_attn_w, pp_attn_w), (ze, zq)
    
    def encode(self, inputs, input_lens, in_mask): 
        ze = self.encoder(inputs, input_lens)
        zq = ze
        return ze, zq

############################ CTC Predictor [20240223] ############################
class CTCEncoderV1(Module): 
    """
    Linear + Bidirectional LSTM
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [39, 64, 16, 3]
        super(CTCEncoderV1, self).__init__()
        # self.lin_1 = LinearPack(in_dim=size_list[0], out_dim=size_list[3])
        self.lin_1 = nn.Linear(size_list["in"], size_list["hid"])
        self.rnn = nn.LSTM(input_size=size_list["hid"], hidden_size=size_list["hid"], 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.lin_2 = nn.Linear(size_list["hid"] * 2, size_list["hid"])

        self.act = nn.ReLU()

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
    
class CTCDecoderV1(Module): 
    """
    注意：decoder is only for classification for CTC pred. 
    This time we use dictionary to pass by the configurations. 
    """
    def __init__(self, size_list, num_layers=1, dropout=0.5):
        # size_list = [13, 64, 16, 3]: similar to encoder, just layer 0 different
        super(CTCDecoderV1, self).__init__()
        self.attention = ScaledDotProductAttention(q_in=size_list["hid"], kv_in=size_list["hid"], qk_out=size_list["hid"], v_out=size_list["hid"])
        self.lin_2 = nn.Linear(size_list["hid"], size_list["class"])
        self.softmax = nn.LogSoftmax(dim=-1)
        # vars
        self.num_layers = num_layers
        self.size_list = size_list

    def forward(self, hid_r, in_mask):
        outputs, attention_weights = self.attention(hid_r, hid_r, hid_r, in_mask.unsqueeze(1))
        outputs = self.lin_2(outputs)
        outputs = self.softmax(outputs)
        return outputs, attention_weights

class CTCPredNetV1(Module):
    def __init__(self, enc_size_list, dec_size_list, embedding_dim, num_layers=1, dropout=0.5):
        # embedding_dim: the number of discrete vectors in hidden representation space
        super(CTCPredNetV1, self).__init__()

        self.encoder = CTCEncoderV1(size_list=enc_size_list, num_layers=num_layers, dropout=dropout)
        self.decoder = CTCDecoderV1(size_list=dec_size_list, num_layers=num_layers, dropout=dropout)
        self.vq_embedding = nn.Embedding(embedding_dim, enc_size_list["hid"])
        self.vq_embedding.weight.data.uniform_(-1.0 / embedding_dim,
                                               1.0 / embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, input_lens, in_mask):
        # inputs : batch_size * time_steps * in_size
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

        dec_out, attn_w = self.decoder(dec_in, in_mask)
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
    
    def predict_on_output(self, output): 
        # output = nn.Softmax(dim=-1)(output)
        preds = torch.argmax(output, dim=-1)
        return preds