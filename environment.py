import numpy as np

from mpe2 import simple_spread_v3


class MPEEnvironment:
    def __init__(
        self,
        n_agents=3,
        max_cycles=25
    ):
        self.n_agents = n_agents
        self.max_cycles = max_cycles

        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            max_cycles=max_cycles,
            continuous_actions=True
        )

        self.agents = self.env.possible_agents

        self.observation_dim = (
            self.env.observation_space(
                self.agents[0]
            ).shape[0]
        )

        self.action_dim = (
            self.env.action_space(
                self.agents[0]
            ).shape[0]
        )

        self.collision_penalty = 1.0
        self.time_penalty = 0.01

        self._episode_over = True

    def reset(self, seed=None):
        observations, infos = self.env.reset(
            seed=seed
        )

        self._episode_over = False

        observations = self._observations_to_array(
            observations
        )

        return observations

    def step(self, actions):
        if self._episode_over:
            raise RuntimeError(
                "step() called on a finished episode. "
                "Call reset() first."
            )

        action_dict = {
            agent: np.asarray(
                actions[i],
                dtype=np.float32
            )
            for i, agent in enumerate(self.agents)
        }

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        ) = self.env.step(action_dict)

        next_observations = (
            self._observations_to_array(
                observations
            )
        )

        team_reward = self._calculate_reward()

        terminated = any(
            bool(terminations[agent])
            for agent in self.agents
        )

        truncated = any(
            bool(truncations[agent])
            for agent in self.agents
        )

        if terminated or truncated:
            self._episode_over = True

        return (
            next_observations,
            team_reward,
            terminated,
            truncated,
            infos
        )

    def _calculate_reward(self):
        world = self.env.unwrapped.world

        agents = world.agents
        landmarks = world.landmarks

        coverage_reward = 0.0

        for landmark in landmarks:
            distances = [
                np.linalg.norm(
                    agent.state.p_pos
                    - landmark.state.p_pos
                )
                for agent in agents
            ]

            minimum_distance = min(distances)

            coverage_reward -= minimum_distance

        collision_count = 0

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                distance = np.linalg.norm(
                    agents[i].state.p_pos
                    - agents[j].state.p_pos
                )

                collision_distance = (
                    agents[i].size
                    + agents[j].size
                )

                if distance < collision_distance:
                    collision_count += 1

        collision_reward = (
            -self.collision_penalty
            * collision_count
        )

        time_reward = -self.time_penalty

        total_reward = (
            coverage_reward
            + collision_reward
            + time_reward
        )

        return float(total_reward)

    def _observations_to_array(
        self,
        observations
    ):
        return np.stack(
            [
                observations[agent]
                for agent in self.agents
            ]
        ).astype(np.float32)

    def close(self):
        self.env.close()
