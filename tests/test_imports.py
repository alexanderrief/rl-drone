from src.rl.buffer import Buffer
from src.rl.policy import PolicyNetwork
from src.rl.value import ValueNetwork


def test_buffer_instantiates():
    Buffer(128, 32)


def test_policy_instantiates():
    PolicyNetwork(input_dim=6, output_dim=3)


def test_value_instantiates():
    ValueNetwork(input_dim=6)
