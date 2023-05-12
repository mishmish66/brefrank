import brax

from brax import envs

from brax.training.agents import ppo
from brax import geometry as braxgeo
import brax.base
import jax
from jax import numpy as jp


@jax.jit
def flying_reward(state: envs.State, action: jax.Array) -> jax.Array:
    contact = braxgeo.contact(  # type: brax.base.Contact
        self.sys, state.pipeline_state.x  # type: ignore
    )

    num_contacts = contact.link_idx[0].shape[0]  # type: ignore

    return state.reward + num_contacts * -0.25


@jax.jit
def lazy_reward(state: envs.State, action: jax.Array) -> jax.Array:
    movement = jp.linalg.norm(action, axis=-1)

    return state.reward + movement * -0.5


@jax.jit
def steady_reward(state: envs.State, action: jax.Array) -> jax.Array:
    body_y_vel = state.pipeline_state.xd.vel[0]  # type: ignore

    return state.reward + body_y_vel**2 * -0.5


@jax.jit
def tall_reward(state: envs.State, action: jax.Array) -> jax.Array:
    z_height = state.pipeline_state.x.pos[0][2]  # type: ignore

    return state.reward + z_height * 0.25


reward_flavors = [flying_reward, lazy_reward, steady_reward, tall_reward]
