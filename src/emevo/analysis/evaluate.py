"""Evaluate saved policy"""

import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from emevo import Env
from emevo.environments.circle_foraging import CFState
from emevo.exp_utils import SavedPhysicsState
from emevo.rl import ppo_normal as ppo


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate(network: ppo.NormalPPONet, obs: jax.Array) -> ppo.Output:
    return network(obs)


def load_network(env: Env, policy_path: list[Path]) -> ppo.NormalPPONet:
    input_size = int(np.prod(env.obs_space.flatten().shape))
    act_size = int(np.prod(env.act_space.shape))
    ref_net = ppo.NormalPPONet(input_size, 64, act_size, jax.random.PRNGKey(0))
    net_params = []
    for policy_path_i in policy_path:
        pponet = eqx.tree_deserialise_leaves(policy_path_i, ref_net)
        # Append only params of the network, excluding functions (etc. tanh).
        net_params.append(eqx.filter(pponet, eqx.is_array))
    net_params = jax.tree.map(lambda *args: jnp.stack(args), *net_params)
    network = eqx.combine(net_params, ref_net)
    return network


def eval_policy(
    env: Env,
    physstate_path: list[Path],
    policy_path: list[Path],
    agent_index: int | None = None,
) -> list[tuple[ppo.Output, CFState, int]]:
    env_state, _ = env.reset(jax.random.PRNGKey(0))
    network = load_network(env, policy_path)
    # Get obs
    n_agents = env.n_max_agents
    zero_action = jnp.zeros((n_agents, *env.act_space.shape))
    outputs = []
    for physpath in physstate_path:
        # agent_index
        if agent_index is None:
            file_name = physpath.stem
            if "slot" in file_name:
                agent_index = int(file_name[file_name.index("slot") + 4 :])
            else:
                print("Set --agent-index")
                return []

        phys_state = SavedPhysicsState.load(physpath)
        loaded_phys = phys_state.set_by_index(..., env_state.physics)
        env_state = dataclasses.replace(env_state, physics=loaded_phys)
        _, timestep = env.step(env_state, zero_action)
        obs_array = timestep.obs.as_array()
        obs_i = obs_array[agent_index]
        output = evaluate(network, obs_i)
        outputs.append((output, env_state, agent_index))
    return outputs
