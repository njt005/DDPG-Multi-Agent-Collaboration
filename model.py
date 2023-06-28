import typing as t
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Policy approximation"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: t.Tuple[int, ...],
        seed: int = 5,
    ) -> None:
        """__init__

        Initialize a linear MLP with RELU activation functions.

        Parameters
        ----------
        state_size : int
            size of input state
        action_size : int
            size of output (number of available actions)
        hidden_sizes : t.Tuple[int, ...]
            hidden layer sizes
        seed : int, optional
            random seed for network, by default 5
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = (state_size,) + hidden_sizes + (action_size,)
        self.model = self.build_model(layer_sizes)

    def build_model(self, layer_sizes: t.Tuple[int, ...]) -> None:
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(nn.BatchNorm1d(layer_sizes[i - 1]))
            layers.append(
                nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
            )

            # Last layer no activation function
            if i != len(layer_sizes) - 1:
                layers.append(nn.ReLU())

        # Continuous output
        model = nn.Sequential(*layers, nn.Tanh())
        model.apply(init_weights)
        return model

    def forward(self, state: torch.Tensor):
        return self.model(state)


class Critic(nn.Module):
    """State-Action Value Approximation (Q-Network)"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes_s: t.Tuple[int, ...],
        hidden_sizes_sa: t.Tuple[int, ...],
        seed: int = 5,
    ) -> None:
        """__init__

        Initialize a linear MLP with RELU activation functions.

        Parameters
        ----------
        state_size : int
            size of input state
        action_size : int
            size of output (number of available actions)
        hidden_sizes_s : t.Tuple[int, ...]
            hidden layer sizes for state only
        hidden_sizes_sa : t.Tuple[int, ...]
            hidden layer sizes for state concatenated with action
        seed : int, optional
            random seed for network, by default 5
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes_s = (state_size,) + hidden_sizes_s
        layer_sizes_sa = (hidden_sizes_s[-1] + action_size,) + hidden_sizes_sa + (1,)
        self.model_s, self.model_sa = self.build_model(layer_sizes_s, layer_sizes_sa)

    def build_model(
        self, layer_sizes_s: t.Tuple[int, ...], layer_sizes_sa: t.Tuple[int, ...]
    ) -> None:
        layers_s = []
        for i in range(1, len(layer_sizes_s)):
            layers_s.append(nn.BatchNorm1d(layer_sizes_s[i - 1]))
            layers_s.append(
                nn.Linear(
                    in_features=layer_sizes_s[i - 1], out_features=layer_sizes_s[i]
                )
            )

            layers_s.append(nn.ReLU())
        model_s = nn.Sequential(*layers_s)

        layers_sa = []
        for i in range(1, len(layer_sizes_sa)):
            layers_sa.append(
                nn.Linear(
                    in_features=layer_sizes_sa[i - 1], out_features=layer_sizes_sa[i]
                )
            )

            if i != len(layer_sizes_sa) - 1:
                layers_sa.append(nn.ReLU())
        model_sa = nn.Sequential(*layers_sa, nn.Sigmoid())

        model_s.apply(init_weights)
        model_sa.apply(init_weights)

        return model_s, model_sa

    def forward(self, state, action):
        xs = self.model_s(state)
        return self.model_sa(torch.cat((xs, action), dim=1))


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
