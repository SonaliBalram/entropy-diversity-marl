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

        # =================================================
        # CREATE MPE ENVIRONMENT
        # =================================================

        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            max_cycles=max_cycles,
            continuous_actions=True
        )

        # =================================================
        # AGENT NAMES
        # =================================================

        self.agents = self.env.possible_agents

        # =================================================
        # OBSERVATION / ACTION DIMENSIONS
        # =================================================
        #
        # For simple_spread_v3 with N=3:
        #     observation_dim = 18
        #         self_vel(2), self_pos(2),
        #         landmark_rel_pos(3*2=6),
        #         other_agent_rel_pos(2*2=4),
        #         communication(2*2=4)
        #
        #     action_dim = 5
        #         [no_op, left, right, down, up]
        #         physical force is
        #             u_x = a[2] - a[1]
        #             u_y = a[4] - a[3]
        #         (a[0] is ignored by MPE)
        #
        # NOTE: action_dim is 5, not 2. Any comment or report
        # text saying "2 continuous components" refers to the
        # resulting 2-D force, not the action vector the actor
        # actually outputs.

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

        # =================================================
        # REWARD PARAMETERS
        # =================================================

        self.collision_penalty = 1.0
        self.time_penalty = 0.01

        # Guard so that stepping a finished episode raises a
        # clear error instead of a bare KeyError from PettingZoo.
        self._episode_over = True


    # =====================================================
    # RESET
    # =====================================================

    def reset(self, seed=None):

        observations, infos = self.env.reset(
            seed=seed
        )

        self._episode_over = False

        observations = self._observations_to_array(
            observations
        )

        return observations


    # =====================================================
    # STEP
    # =====================================================

    def step(self, actions):

        """
        Returns (Gymnasium-style 5-tuple):

            next_observations : [n_agents, observation_dim]
            team_reward       : float
            terminated        : bool  -- TRUE task completion
            truncated         : bool  -- time limit reached
            infos             : dict

        terminated and truncated are returned SEPARATELY on
        purpose. See the note in maddpg_system.update():
        only `terminated` may be used to cut the bootstrap in
        the Bellman target. simple_spread has no terminal
        state at all, so `terminated` is always False here.
        """

        if self._episode_over:
            raise RuntimeError(
                "step() called on a finished episode. "
                "Call reset() first."
            )

        # -------------------------------------------------
        # Convert array into agent dictionary
        # -------------------------------------------------

        action_dict = {
            agent: np.asarray(
                actions[i],
                dtype=np.float32
            )
            for i, agent in enumerate(self.agents)
        }

        # -------------------------------------------------
        # Step environment
        # -------------------------------------------------

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        ) = self.env.step(action_dict)

        # -------------------------------------------------
        # Convert observations
        # -------------------------------------------------

        next_observations = (
            self._observations_to_array(
                observations
            )
        )

        # -------------------------------------------------
        # CUSTOM TEAM REWARD
        # -------------------------------------------------
        #
        # Computed from world state AFTER the transition,
        # which is correct: r_t = R(s_{t+1}).
        # The environment's own `rewards` dict is deliberately
        # ignored -- document this in the write-up, because it
        # means returns are NOT comparable to published
        # simple_spread numbers.

        team_reward = self._calculate_reward()

        # -------------------------------------------------
        # TERMINATION vs TRUNCATION
        # -------------------------------------------------

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


    # =====================================================
    # CUSTOM REWARD
    # =====================================================

    def _calculate_reward(self):

        world = self.env.unwrapped.world

        agents = world.agents
        landmarks = world.landmarks

        # -------------------------------------------------
        # COVERAGE REWARD
        # -------------------------------------------------

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


        # -------------------------------------------------
        # COLLISION PENALTY
        # -------------------------------------------------
        #
        # Each unordered pair is counted ONCE. Native MPE
        # counts each collision twice (once per agent), so
        # this baseline's collision term is half the native
        # weight for the same collision_penalty value.

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


        # -------------------------------------------------
        # TIME PENALTY
        # -------------------------------------------------

        time_reward = -self.time_penalty


        total_reward = (
            coverage_reward
            + collision_reward
            + time_reward
        )

        return float(total_reward)


    # =====================================================
    # OBSERVATION CONVERSION
    # =====================================================

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


    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        self.env.close()
