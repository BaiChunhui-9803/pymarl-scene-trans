from modules.agents import REGISTRY as agent_REGISTRY
from components.custom_action_selectors import REGISTRY as action_REGISTRY
import torch as th
from src.components.epsilon_schedules import DecayThenFlatSchedule


# This multi-agent controller shares parameters between agents
class CustomController:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.args = args
        self.agent = None
        # input_shape = self._get_input_shape(scheme)
        self._build_agents()
        self.agent_output_type = args.agent_output_type
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.action_selector = action_REGISTRY[args.action_selector](args)

        # self.hidden_states = None

    def select_actions(self, env, ep_batch, t_ep, t_env, bs=0, test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][bs][t_ep]
        state_im = ep_batch["im_state"][bs][t_ep]
        cluster_action = self.agent.cluster_qtable.choose_action(state_im, self.get_epsilon(t_env, test_mode))
        env.cluster.update(env.agents, env.enemies)
        cluster_result = getattr(env.cluster, cluster_action)()
        self.agent.update_combat_qtable_dict(cluster_result)
        self.agent.sub_clusters_qtable_tag = (cluster_result[0], cluster_result[1])
        state_clu = env.get_clu_state(cluster_result)
        combat_action = (self.agent.combat_qtable_dict[self.agent.sub_clusters_qtable_tag]
                              .choose_action(state_clu, self.get_epsilon(t_env, test_mode)))
        # model = self.get_model()
        # agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # chosen_actions = self.action_selector.select_action(model, cur_state, avail_actions, t_env, test_mode=test_mode)
        # return getattr(env, combat_action)(cluster_result)
        return getattr(env, "action_MIX_lure_weakest")(cluster_result)

    def get_model(self):
        model = {
            "cluster_qtable": self.agent.get_cluster_qtable(),
            "combat_qtable_dict": self.agent.get_combat_qtable_dict()
        }
        return model

    def get_state(self):
        return self.agent.get_state()

    def get_epsilon(self, t_env, test_mode=False):
        self.epsilon = self.schedule.eval(t_env)
        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0
        return self.epsilon















    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.agent](self.scheme["avail_actions"], self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
