import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmlatch.attention import Attention
from mmlatch.util import pad_mask


class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""

    def __init__(self, batch_first=True):
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length
        )
        return x


class PackSequence(nn.Module):
    def __init__(self, batch_first=True):
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )
        lengths = lengths[x.sorted_indices]
        return x, lengths


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0,
        rnn_type="lstm",
        packed_sequence=True,
        device="cpu",
    ):

        super(RNN, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        self.out_size = hidden_size

        if bidirectional and merge_bi == "cat":
            self.out_size = 2 * hidden_size

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def _merge_bi(self, forward, backward):
        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188

        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)

        return self._merge_bi(last_forward_out, last_backward_out)

    def merge_hidden_bi(self, out):
        if not self.bidirectional:
            return out

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])

        return self._merge_bi(forward, backward)

    def forward(self, x, lengths, initial_hidden=None):
        self.rnn.flatten_parameters()

        if self.packed_sequence:
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
            lengths = lengths.to(self.device)

        if initial_hidden is not None:
            out, hidden = self.rnn(x, initial_hidden)
        else:
            out, hidden = self.rnn(x)

        if self.packed_sequence:
            out = self.unpack(out, lengths)

        out = self.drop(out)
        last_timestep = self._final_output(out, lengths)
        out = self.merge_hidden_bi(out)

        return out, last_timestep, hidden

class MemoryModule(nn.Module):
    def __init__(self, memory_slots, memory_dim, controller_dim):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim

        self.memory = nn.Parameter(torch.randn(memory_slots, memory_dim))
        self.key_layer = nn.Linear(controller_dim, memory_dim)
        self.write_layer = nn.Linear(controller_dim, memory_dim)
        self.erase_layer = nn.Linear(controller_dim, memory_dim)

    def forward(self, controller_out):
        """
        controller_out: (B, controller_dim)
        Returns:
            memory_read: (B, memory_dim)
        """
        # Generate key vector
        key = self.key_layer(controller_out)  # (B, memory_dim)

        # Cosine similarity for memory addressing
        mem_norm = self.memory / (self.memory.norm(dim=-1, keepdim=True) + 1e-8)
        key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)

        sim = torch.matmul(key_norm, mem_norm.t())  # (B, memory_slots)
        weights = torch.softmax(sim, dim=-1)  # (B, memory_slots)

        # Read from memory
        memory_read = torch.matmul(weights, self.memory)  # (B, memory_dim)

        # Optional: write to memory
        erase = torch.sigmoid(self.erase_layer(controller_out))  # (B, memory_dim)
        add = self.write_layer(controller_out)  # (B, memory_dim)
        w_exp = weights.unsqueeze(-1)  # (B, memory_slots, 1)

        erase = erase.unsqueeze(1)  # (B, 1, memory_dim)
        add = add.unsqueeze(1)      # (B, 1, memory_dim)

        with torch.no_grad():
            updated_memory = self.memory * (1 - (w_exp * erase)).mean(0) + (w_exp * add).mean(0)
            self.memory.copy_(updated_memory)

        return memory_read

    def reset_memory(self):
        nn.init.randn_(self.memory)

class DifferentiableMemory(nn.Module):
    def __init__(self, memory_slots, memory_dim, controller_dim):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim

        self.memory = nn.Parameter(torch.randn(memory_slots, memory_dim))  # learnable
        self.key_layer = nn.Linear(controller_dim, memory_dim)
        self.erase_layer = nn.Linear(controller_dim, memory_dim)
        self.write_layer = nn.Linear(controller_dim, memory_dim)

    def _address_memory(self, key):
        # key: (B, memory_dim)
        mem_norm = self.memory / (self.memory.norm(dim=-1, keepdim=True) + 1e-8)
        key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(key_norm, mem_norm.t())  # (B, memory_slots)
        weights = torch.softmax(sim, dim=-1)        # (B, memory_slots)
        return weights

    def read(self, controller_out):
        key = self.key_layer(controller_out)        # (B, memory_dim)
        weights = self._address_memory(key)         # (B, memory_slots)
        memory_read = torch.matmul(weights, self.memory)  # (B, memory_dim)
        return memory_read, weights

    def write(self, controller_out, weights):
        erase = torch.sigmoid(self.erase_layer(controller_out)).unsqueeze(1)  # (B, 1, memory_dim)
        add = self.write_layer(controller_out).unsqueeze(1)                   # (B, 1, memory_dim)
        weights = weights.unsqueeze(-1)                                       # (B, memory_slots, 1)

        erase_matrix = (1 - weights * erase).mean(dim=0)                      # (memory_slots, memory_dim)
        add_matrix = (weights * add).mean(dim=0)                              # (memory_slots, memory_dim)

        # update memory with learned ops
        self.memory.data = self.memory.data * erase_matrix + add_matrix

    def forward(self, controller_out):
        read_val, weights = self.read(controller_out)
        self.write(controller_out, weights)
        return read_val

class BetterMemory(nn.Module):
    def __init__(self, memory_slots, memory_dim, controller_dim, dropout=0.2):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim

        self.memory = nn.Parameter(torch.randn(memory_slots, memory_dim))
        self.key_layer = nn.Linear(controller_dim, memory_dim)
        self.erase_layer = nn.Linear(controller_dim, memory_dim)
        self.write_layer = nn.Linear(controller_dim, memory_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(memory_dim)

        self.gate = nn.Linear(controller_dim, memory_dim)

    def _address_memory(self, key):
        mem_norm = F.normalize(self.memory, dim=-1)
        key_norm = F.normalize(key, dim=-1)
        sim = torch.matmul(key_norm, mem_norm.t())     # (B, memory_slots)
        weights = torch.softmax(sim, dim=-1)
        return weights

    def read(self, controller_out):
        key = self.key_layer(controller_out)
        weights = self._address_memory(key)
        read_val = torch.matmul(weights, self.memory)
        return read_val, weights

    def write(self, controller_out, weights):
        erase = torch.sigmoid(self.erase_layer(controller_out)).unsqueeze(1)
        add = self.write_layer(controller_out).unsqueeze(1)
        weights = weights.unsqueeze(-1)  # (B, memory_slots, 1)

        erase_matrix = (1 - weights * erase).mean(dim=0)
        add_matrix = (weights * add).mean(dim=0)

        # Apply dropout before writing
        updated_memory = self.memory.data * erase_matrix + self.dropout(add_matrix)
        self.memory.data = self.ln(updated_memory)  # normalize memory

    def forward(self, controller_seq):
        if controller_seq.dim() == 2:
            # Shape: (B, D) — single timestep
            read_val, weights = self.read(controller_seq)
            self.write(controller_seq, weights)
            g = torch.sigmoid(self.gate(controller_seq))
            return g * controller_seq + (1 - g) * read_val

        elif controller_seq.dim() == 3:
            # Shape: (B, T, D) — full sequence
            B, T, D = controller_seq.shape
            out = []
            for t in range(T):
                c_t = controller_seq[:, t, :]
                read_val, weights = self.read(c_t)
                self.write(c_t, weights)
                g = torch.sigmoid(self.gate(c_t))
                out_t = g * c_t + (1 - g) * read_val
                out.append(out_t.unsqueeze(1))
            return torch.cat(out, dim=1)

        else:
            raise ValueError(f"Unsupported shape for controller_seq: {controller_seq.shape}")

class AttentiveRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0.1,
        rnn_type="lstm",
        packed_sequence=True,
        attention=False,
        return_hidden=False,
        device="cpu",
        memory_augmented=False
    ):
        super(AttentiveRNN, self).__init__()
        self.device = device
        self.rnn = RNN(
            input_size,
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            device=device,
        )
        self.out_size = self.rnn.out_size
        self.attention = None
        self.return_hidden = return_hidden

        if attention:
            self.attention = Attention(attention_size=self.out_size, dropout=dropout)

        self.memory_augmented = memory_augmented
        if memory_augmented:
            self.memory_module = BetterMemory(
                memory_slots=20,
                # memory_dim=hidden_size,
                memory_dim=self.out_size,  # match RNN output dim
                controller_dim=self.out_size
            )
            self.memory_gate = nn.Linear(self.out_size, self.out_size)

    def forward(self, x, lengths, initial_hidden=None):
        out, last_hidden, _ = self.rnn(x, lengths, initial_hidden=initial_hidden)

        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device)
            )
            if not self.return_hidden:
                out = out.sum(1)
        else:
            out = last_hidden

        if self.memory_augmented:
            mem_out = self.memory_module(out)
            gate = torch.sigmoid(self.memory_gate(out))
            out = gate * out + (1 - gate) * mem_out

        return out

