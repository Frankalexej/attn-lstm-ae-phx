import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, q_in, kv_in, qk_out, v_out):
        super(ScaledDotProductAttention, self).__init__()
        self.w_q = nn.Linear(q_in, qk_out)
        self.w_k = nn.Linear(kv_in, qk_out)
        self.w_v = nn.Linear(kv_in, v_out)
        self.d_k = qk_out

    def forward(self, q, k, v, mask=None):
        """
        q: Query tensor of shape (batch_size, num_queries, d_k)
        k: Key tensor of shape (batch_size, num_keys, d_k)
        v: Value tensor of shape (batch_size, num_values, d_v), num_keys = num_values
        mask: Mask tensor of shape (batch_size, num_queries, num_keys)

        Returns: Output tensor of shape (batch_size, num_queries, d_v)
        """
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Step 1: Compute the dot product between queries and keys
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, num_queries, num_keys)

        # Step 2: Scale the attention scores
        attn_scores = attn_scores / (self.d_k ** 0.5)

        # Step 3: Apply the mask (if any)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Compute the softmax of the attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_queries, num_keys)

        # Step 5: Multiply the attention weights with the values
        output = torch.bmm(attn_weights, v)  # (batch_size, num_queries, d_v)

        return output, attn_weights