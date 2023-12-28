import torch.nn as nn
import torch
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment dynamic embedding. The dynamic embedding is used to modify query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": StaticEmbedding,
        "csp": CSPDynamicEmbedding,
        "atsp": StaticEmbedding,
        "cvrp": StaticEmbedding,
        "svrp": StaticEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": StaticEmbedding,
        "spctsp": StaticEmbedding,
        "op": StaticEmbedding,
        "dpp": StaticEmbedding,
        "mdpp": StaticEmbedding,
        "pdp": StaticEmbedding,
        "mtsp": StaticEmbedding,
        "smtwtp": StaticEmbedding,
    }

    if env_name not in embedding_registry:
        log.warning(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}. Defaulting to StaticEmbedding."
        )
    return embedding_registry.get(env_name, StaticEmbedding)(**config)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


class SDVRPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Split Delivery Vehicle Routing Problem (SDVRP).
    Embed the following node features to the embedding space:
        - demand_with_depot: demand of the customers and the depot
    The demand with depot is used to modify the query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    """

    def __init__(self, embedding_dim, linear_bias=False):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embedding_dim, bias=linear_bias)

    def forward(self, td):
        demands_with_depot = td["demand_with_depot"][..., None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class CSPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Covering Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - guidence_vec: guidence value of the customers
    The guidence value is used to modify the query, key and value vectors of the attention mechanism
    based on the current covered state of the nodes (which is changing during the decoding step).
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(CSPDynamicEmbedding, self).__init__()
        self.proj_guidence = nn.Linear(1, 3 * embedding_dim, linear_bias, dtype=torch.float32)

    def forward(self, td):

        (glimpse_key_dynamic, 
         glimpse_val_dynamic, 
         logit_key_dynamic 
         )= self.proj_guidence(td["guidence_vec"].unsqueeze(-2)
                               .transpose(1, 2).float()).chunk(3, dim=-1)  # [batch, num_loc, 3*embed_dim] -> [batch, num_loc, embed_dim] for every

        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
