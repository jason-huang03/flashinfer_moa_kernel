import torch
import flashinfer
import random
import argparse
import icecream
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_attention_mask

parser = argparse.ArgumentParser(description='Test the performance of the flashinfer')
parser.add_argument("--num_devices", type=int, default=1, help="device id")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--kv_len", type=int, default=6001, help="kv length")
parser.add_argument("--num_kv_heads", type=int, default=32, help="number of kv heads")
parser.add_argument("--num_qo_heads", type=int, default=32, help="number of qo heads")
args = parser.parse_args()

## assume block size to be 64 ##

batch_size = args.batch_size
kv_len = args.kv_len
qo_len = kv_len
assert(kv_len > 64)
num_kv_heads = args.num_kv_heads
head_dim = 128
num_qo_heads = args.num_qo_heads

padding_mask = torch.ones(batch_size, kv_len, dtype=torch.int64)
padding_lengths = torch.zeros((batch_size, ), dtype=torch.long, device='cuda')

# randomly pad a few tokens on the left
for i in range(batch_size):
    # temp_len = random.randint(0, 32)
    temp_len = random.randint(0, 64)
    padding_mask[i, : temp_len] = 0
    padding_lengths[i] = temp_len

padding_mask = _prepare_4d_causal_attention_mask_for_sdpa(
    padding_mask, 
    (batch_size, kv_len), 
    torch.randn(batch_size, kv_len, num_kv_heads * head_dim, device='cuda', dtype=torch.float16), 
    0)
padding_mask = (padding_mask == 0)
assert(num_kv_heads == num_qo_heads)

def create_block_mask(num_global_blocks, num_band_blocks, block_size, token_len):
    # first create a boolen causal mask. with true under the diagnol
    blocked_token_len = (token_len + block_size - 1) // block_size

    causal_mask = torch.ones(num_qo_heads, token_len, token_len, dtype=torch.bool)
    causal_mask = torch.tril(causal_mask, diagonal=0)

    blocked_causal_mask = torch.ones(num_qo_heads, blocked_token_len, blocked_token_len, dtype=torch.bool)
    blocked_causal_mask = torch.tril(blocked_causal_mask, diagonal=0)

    for k in range(num_qo_heads):
        global_blocks = num_global_blocks[k]
        band_blocks = num_band_blocks[k]
        
        for i in range(blocked_token_len):
            for j in range(blocked_token_len):
                if j > i:
                    continue

                if (not j < global_blocks) and (not j >= i - band_blocks + 1):
                    blocked_causal_mask[k, i, j] = False

    # expand the blocked mask with block size = 64
    blocked_causal_mask_expanded = blocked_causal_mask.unsqueeze(2).unsqueeze(4)
    blocked_causal_mask_expanded = blocked_causal_mask_expanded.repeat(1, 1, block_size, 1, block_size)
    blocked_causal_mask_expanded = blocked_causal_mask_expanded.view(num_qo_heads, 64 * blocked_token_len, 64 * blocked_token_len)
    blocked_causal_mask_expanded = blocked_causal_mask_expanded[:, :token_len, :token_len].contiguous() * causal_mask

    return blocked_causal_mask_expanded

for i in range(args.num_devices):
    print(f"device: {i}")
    device_id = i
    num_band_blocks = [random.randint(1, kv_len // 64) for _ in range(num_qo_heads)]
    num_global_blocks = [random.randint(1, 5) for _ in range(num_qo_heads)]
    
    print("num global blocks for each head:")
    print(num_global_blocks)
    print("num band blocks for each head:")
    print(num_band_blocks)

    causal_mask = create_block_mask(num_global_blocks, num_band_blocks, 64, kv_len).to(device_id)
    # calculate the and of causal mask and padding mask
    if isinstance(padding_mask, torch.Tensor):
        causal_mask = causal_mask.unsqueeze(0) & (padding_mask.to(torch.bool).to(device_id))

    q = torch.randn(batch_size, qo_len, num_qo_heads, head_dim).half().to(device_id) # prefill attention
    k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim).half().to(device_id) 
    v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim).half().to(device_id) 

    print(f"q shape: {q.shape}")
    print(f"kv shape: {k.shape}")

    ### moa kernel ###
    num_band_blocks = torch.tensor(num_band_blocks, dtype=torch.long).to(device_id)
    num_global_blocks = torch.tensor(num_global_blocks, dtype=torch.long).to(device_id)
    o = flashinfer.moa_prefill(q, k, v, left_padding_lengths=padding_lengths, causal=True, num_global_blocks=num_global_blocks, num_band_blocks=num_band_blocks, kv_layout="NHD")
    ##################

    k_trans = k.transpose(1, 2)
    v_trans = v.transpose(1, 2)

    ### moa kernel ###
    o_trans = flashinfer.moa_prefill(q, k_trans, v_trans, left_padding_lengths=padding_lengths, causal=True, num_global_blocks=num_global_blocks, num_band_blocks=num_band_blocks, kv_layout="HND")
    ##################

    print(f"difference between NHD and HND version: ")
    icecream.ic(torch.max(torch.abs(o - o_trans)))

    # expand the batch dimnesion for sdpa
    q_sdpa = q.clone().transpose(1, 2).contiguous()
    k_sdpa = k.clone().transpose(1, 2).contiguous()
    v_sdpa = v.clone().transpose(1, 2).contiguous()

    k_sdpa = repeat_kv(k_sdpa, num_qo_heads // num_kv_heads)
    v_sdpa = repeat_kv(v_sdpa, num_qo_heads // num_kv_heads)
    k_sdpa = k_sdpa.contiguous()
    v_sdpa = v_sdpa.contiguous()

    ### sdpa ###
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=causal_mask, dropout_p=0.0
    ).transpose(1, 2).contiguous()
    ############

    # check the results
    print(f"difference between moa and sdpa: ")
    for j in range(batch_size):
        padding_len = padding_lengths[j]
        icecream.ic(torch.max(torch.abs(o[j, padding_len:] - o_sdpa[j, padding_len:])))