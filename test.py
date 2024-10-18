from collections import defaultdict
import numpy as np
import torch

observation = [10, 2, 1,4,5,6,7]
next_state = torch.tensor(
                observation, dtype=torch.float32
            )

print(next_state, next_state.dim(), next_state.shape)

next_state1 = next_state.unsqueeze(0)
print(next_state1, next_state1.dim(), next_state1.shape)

next_state2 = next_state.unsqueeze(1)
print(next_state2, next_state2.dim(), next_state2.shape)

us3 = next_state2.unsqueeze(0)
print(us3, us3.dim(), us3.shape)