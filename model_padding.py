# --------------------------
# Padding contains function related to padding, masking as well as 
# loss calculations with pad maskings. Also, testing codes were included as backup. 
# --------------------------
import torch
import random

def generate_mask_from_padding(padded_sequence, padding_value=0):
    """
    Generate a mask tensor from a padded sequence tensor.
    """
    mask = (padded_sequence != padding_value).float()  # mask is 1 for valid elements, 0 for padded elements
    return mask

def generate_mask_from_lengths(lengths, max_length=None):
    """
    Generate a mask for a batch of sequences with varying lengths.

    Args:
    lengths (list of int): A list of sequence lengths.
    max_length (int, optional): The maximum length of the mask. If not provided, the maximum length will be the maximum value in the 'lengths' list.

    Returns:
    torch.Tensor: A boolean tensor with shape (batch_size, max_length) where element (i, j) is True if j < lengths[i], otherwise False.
    """
    if max_length is None:
        max_length = max(lengths)
    batch_size = len(lengths)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return mask

def generate_mask_from_lengths_mat(lengths, device): 
    num_elements = len(lengths)
    max_length = max(lengths)
    # Create a tensor of shape (num_elements, max_length) with increasing indices along the columns
    indices = torch.arange(max_length, device=device).view(1, -1)

    # Create a tensor of shape (num_elements, max_length) by broadcasting the lengths
    lengths_tensor = torch.tensor(lengths, device=device).view(-1, 1)
    
    # Compare the indices tensor with the lengths tensor to generate the desired tensor
    tensor = (indices < lengths_tensor).to(torch.int32)
    return tensor

def mask_it(original, mask): 
    # Apply the mask to the tensor
    # Expand the mask along the feature dimension
    # b, t, f = original.shape
    mask_expanded = mask.unsqueeze(-1).float()
    return original * mask_expanded

# def masked_loss(loss_fn, y_hat, y, mask):
#     """
#     Computes the masked loss given a loss function, predicted values, ground truth values, and a mask tensor.
#     """
#     b, t, f = y_hat.shape
#     mask = mask.unsqueeze(-1).expand((b, t, f)).bool()  # both y_hat and y are of same size
#     # Apply mask to the predicted and ground truth values
#     y_hat_masked = y_hat.masked_select(mask)
#     y_masked = y.masked_select(mask)

#     # Calculate the loss using the masked values
#     loss = torch.sum(loss_fn(y_hat_masked, y_masked)) / torch.sum(mask)
#     # loss = loss_fn(y_hat_masked, y_masked)

#     return loss

# --------------------------
# Testing codes
# --------------------------
def collate_fn_test(batch):
    batch_first = True
    # (xx, yy) = zip(*batch)
    xx = batch
    x_lens = [len(x) for x in xx]
    # y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
    # yy_pad = pad_sequence(yy, batch_first=batch_first, padding_value=0)
    return xx_pad, x_lens

def generate_random_matrix(hidden_dim):
    random_matrix = torch.randn(hidden_dim, hidden_dim)
    return random_matrix

if __name__ == '__main__':
    # code to run when the script is executed
    batch_size = 5
    hidden_size = 3

    # Create a list to store the tensors
    tensor_list = []

    # Generate random tensors
    for i in range(batch_size):
        # Randomly choose the length of the tensor
        length = random.randint(1, 5)

        # Create a random tensor of shape (length, hidden_size)
        tensor = torch.randn(length, hidden_size)

        # Append the tensor to the list
        tensor_list.append(tensor)
    padded_tensor, tl = collate_fn_test(tensor_list)
    
    mask = generate_mask_from_padding(padded_tensor)
    
    W = generate_random_matrix(hidden_size)

    transformed_padded_tensor = torch.matmul(padded_tensor, W)
    
    masked_transformed_padded_tensor = mask_it(transformed_padded_tensor, mask)
    
    lf = nn.MSELoss()
    
    plain_loss = lf(masked_transformed_padded_tensor, padded_tensor)
    
    mloss = masked_loss(lf, masked_transformed_padded_tensor, padded_tensor, mask)
    
    print(plain_loss, mloss)

# Conclusion: this method works, as it stops the loss to be decreased by counting the paddings in. 

# It might not be okay to use PackedSequence, therefore, it would be important to manually get rid of paddings when computing. It seems that manually applying masking can perform similarly to PackedSequence. 