import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):
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
