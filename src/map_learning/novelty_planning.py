"""Experimental deterministic MCTS rewards; archives contain observations, not oracle states."""
import numpy as np
import torch
from .uncertainty_planning import UncertaintyPlanner, SearchNode
from .probabilistic import mixture_mean


class NoveltyPlanner(UncertaintyPlanner):
    MODES = ('root_distance', 'path_distance', 'archive_path', 'disagreement_path', 'recent_archive')

    def __init__(self, trainer, config, mode, seed=0):
        super().__init__(trainer, config, seed)
        if not self.mean_mode or mode not in self.MODES:
            raise ValueError('Novelty experiments require mean dynamics and a supported reward')
        self.mode = mode
        self.observations = {}
        self.archive = None
        self.step_scale = 1.
        self.recent_observations = []
        self.recent = None

    @torch.no_grad()
    def refresh(self, replay):
        # Lossless deduplication of identical observations accelerates a minimum;
        # this does not merge nearby observations or change the training replay.
        for obs in replay.observations + replay.next_observations:
            self.observations.setdefault(obs.tobytes(), obs)
        if getattr(self,'recent_observations',[]):
            self.recent=self.trainer.model.encode(self.trainer.tensor(np.stack(self.recent_observations)))
        raw = np.stack(list(self.observations.values()))
        self.archive = self.trainer.model.encode(self.trainer.tensor(raw))
        indices = np.linspace(0, len(replay)-1, min(len(replay), 2048), dtype=int)
        x, _, y = replay.transitions_at(indices)
        zx, zy = self.trainer.model.encode(self.trainer.tensor(np.concatenate([x,y]))).chunk(2)
        lengths = (zy-zx).norm(dim=-1)
        positive = lengths[lengths>0]
        self.step_scale = max(float(positive.median()) if len(positive) else 1., 1e-8)

    @torch.no_grad()
    def observe(self, observation):
        if self.mode=='recent_archive':
            self.recent_observations=(self.recent_observations+[observation.copy()])[-16:]
            self.recent=self.trainer.model.encode(self.trainer.tensor(np.stack(self.recent_observations)))
        key = observation.tobytes()
        if key not in self.observations:
            self.observations[key] = observation.copy()
            z = self.trainer.model.encode(self.trainer.tensor(observation[None]))
            self.archive = torch.cat([self.archive, z])

    def prepare(self, node):
        if node.parameters is not None:
            return
        n = self.trainer.dynamics.actions
        node.parameters, disagreement = self.distribution_reward(
            node.state.expand(n,-1), torch.arange(n,device=self.trainer.device))
        node.successors = mixture_mean(node.parameters).mean(0)
        distance = torch.cdist(node.successors, node.history, compute_mode='donot_use_mm_for_euclid_dist').min(-1).values
        normalization = (self.reward_variance * node.state.shape[-1])**.5
        if self.mode == 'root_distance':
            reward = (node.successors-self.root_state).norm(dim=-1)/normalization
        elif self.mode == 'path_distance':
            reward = distance/normalization
        elif self.mode in ('archive_path','recent_archive'):
            archive_distance = torch.cdist(node.successors, self.archive, compute_mode='donot_use_mm_for_euclid_dist').min(-1).values
            reward = (torch.minimum(distance, archive_distance)+(distance if self.mode=='recent_archive' else 0))/normalization
        else:
            reward = disagreement*distance/(distance+self.step_scale)
        node.rewards = reward.cpu().numpy()
        node.disagreement = disagreement.cpu().numpy()

    def simulate(self, node, depth):
        if depth == self.config.horizon:
            return 0.
        self.prepare(node)
        unused = np.flatnonzero(node.counts == 0)
        if len(unused):
            action = int(self.rng.choice(unused))
        else:
            average = node.returns/node.counts
            bonus = self.config.uct_exploration*self.uct_scale*np.sqrt(np.log(node.counts.sum()+1)/node.counts)
            scores = average+bonus
            action = int(self.rng.choice(np.flatnonzero(scores == scores.max())))
        value = float(node.rewards[action])
        if depth+1 < self.config.horizon:
            outcomes = node.children[action]
            if not outcomes:
                nxt = node.successors[action:action+1]
                child = SearchNode(nxt, self.trainer.dynamics.actions)
                child.history = torch.cat([node.history,nxt])
                outcomes.append(child); self.nodes_created += 1
            else:
                child = outcomes[int(self.rng.integers(len(outcomes)))]
            value += self.config.uncertainty_discount*self.simulate(child,depth+1)
        node.counts[action] += 1; node.returns[action] += value
        return value

    def mcts(self,z):
        self.root_state = z
        root = SearchNode(z,self.trainer.dynamics.actions); root.history = torch.cat([self.recent,z]) if self.mode=='recent_archive' and self.recent is not None else z
        self.nodes_created = 1; self.prepare(root)
        scale = max(float(root.rewards.max()),1e-12)
        if self.mode != 'disagreement_path':
            # Archive rewards at a known root can be nearly zero even with a
            # valuable frontier downstream; keep UCT on an observed-step scale.
            scale = max(scale,self.step_scale/(self.reward_variance*z.shape[-1])**.5)
        self.uct_scale = scale*sum(self.config.uncertainty_discount**i for i in range(self.config.horizon))
        for _ in range(self.config.simulations):self.simulate(root,0)
        choices = np.flatnonzero(root.counts==root.counts.max()); action=int(self.rng.choice(choices))
        self.last_root = root
        self.last_metrics = {'root_disagreement':float(root.disagreement[action]),'root_reward':float(root.rewards[action]),
            'search_return':float(root.returns[action]/root.counts[action]),'search_nodes':self.nodes_created,
            'root_visits':root.counts.tolist(),'root_rewards':root.rewards.tolist(),
            'root_q':(root.returns/np.maximum(root.counts,1)).tolist(),'reward_mode':self.mode}
        return action

    def shooting(self,z):
        """Persistent random action sequences, scored with the same reward."""
        c=self.config;count=c.candidates;state=z.expand(count,-1);history=[state]
        if self.mode=='recent_archive' and self.recent is not None:
            history=[r[None].expand(count,-1) for r in self.recent]+history
        previous=self.rng.integers(self.trainer.dynamics.actions,size=count)
        scores=torch.zeros(count,device=z.device);first=None
        normalizer=(self.reward_variance*z.shape[-1])**.5
        for depth in range(c.horizon):
            action=np.where(self.rng.random(count)<c.action_repeat_probability,previous,
                            self.rng.integers(self.trainer.dynamics.actions,size=count))
            previous=action
            if first is None:first=action.copy()
            params,disagreement=self.distribution_reward(state,torch.as_tensor(action,device=z.device))
            successor=mixture_mean(params).mean(0)
            distance=(successor[:,None]-torch.stack(history,dim=1)).norm(dim=-1).min(-1).values
            if self.mode in ('archive_path','recent_archive'):
                archive_distance=torch.cdist(successor,self.archive,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values
                reward=(torch.minimum(distance,archive_distance)+(distance if self.mode=='recent_archive' else 0))/normalizer
            elif self.mode=='path_distance':reward=distance/normalizer
            elif self.mode=='root_distance':reward=(successor-z).norm(dim=-1)/normalizer
            else:reward=disagreement*distance/(distance+self.step_scale)
            scores+=c.uncertainty_discount**depth*reward
            state=successor;history.append(state)
        best=int(scores.argmax());self.last_metrics={'search_return':float(scores[best]),'reward_mode':self.mode,'candidate_persistence':c.action_repeat_probability}
        return int(first[best])

    def reset(self):
        super().reset()
        self.recent_observations=[]
        self.recent=None
