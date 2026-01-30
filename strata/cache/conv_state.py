import torch
from torch import Tensor


class ConvStateStore:
    """
    Fixed-size arena for ShortConv layer state storage. Manages per-sequence conv state for 
    all ShortConv layers, using a ring buffer pattern for incremental decode updates.
    """

    def __init__(
        self,
        num_layers: int,
        max_sequences: int,
        kernel_size: int,
        conv_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize conv state arena for all ShortConv layers.

        :param num_layers: Number of ShortConv layers
        :param max_sequences: Maximum concurrent sequences
        :param kernel_size: Convolution kernel size
        :param conv_dim: Convolution hidden dimension
        :param dtype: Data type for state tensors
        """
        self.num_layers = num_layers
        self.max_sequences = max_sequences
        self.state_len = kernel_size - 1
        self.conv_dim = conv_dim
        self.dtype = dtype
        self.states = []
        for _ in range(num_layers):
            state = torch.zeros(
                max_sequences, self.state_len, conv_dim,
                dtype=dtype,
            )
            self.states.append(state)
        self.allocated_slots = set()


    def allocate_slot(self, seq_id: int) -> int:
        """Allocate a state slot for a sequence.

        :param seq_id: Sequence identifier
        :returns: Allocated slot index
        """
        slot = seq_id % self.max_sequences
        self.allocated_slots.add(slot)
        return slot


    def deallocate_slot(self, slot: int) -> None:
        """Deallocate a state slot and clear its data.

        :param slot: Slot index to deallocate
        :returns: None
        """
        if slot in self.allocated_slots:
            self.allocated_slots.discard(slot)
            for layer_idx in range(self.num_layers):
                self.states[layer_idx][slot].zero_()


    def get_state(self, layer_idx: int, slots: Tensor) -> Tensor:
        """Get conv state for specified slots at a layer.

        :param layer_idx: Index of the ShortConv layer
        :param slots: Tensor of slot indices
        :returns: State tensor of shape [batch, state_len, conv_dim]
        """
        return self.states[layer_idx][slots]


    def set_state(self, layer_idx: int, slots: Tensor, new_state: Tensor) -> None:
        """Set conv state for specified slots at a layer.

        :param layer_idx: Index of the ShortConv layer
        :param slots: Tensor of slot indices
        :param new_state: New state tensor of shape [batch, state_len, conv_dim]
        :returns: None
        """
        self.states[layer_idx][slots] = new_state


    def get_full_state_for_layer(self, layer_idx: int) -> Tensor:
        """Get the full state tensor for a layer.

        :param layer_idx: Index of the ShortConv layer
        :returns: State tensor of shape [max_sequences, state_len, conv_dim]
        """
        return self.states[layer_idx]


    def memory_usage_bytes(self) -> int:
        """Calculate total memory usage of the conv state store."""
        element_size = self.states[0].element_size()
        elements_per_layer = self.max_sequences * self.state_len * self.conv_dim
        return self.num_layers * elements_per_layer * element_size
