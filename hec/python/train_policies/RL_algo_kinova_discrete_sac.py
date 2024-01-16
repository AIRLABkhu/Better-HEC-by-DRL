import rospy
from hec.srv import actionSrv, actionSrvResponse
from hec.srv import nextSrv

import random

import numpy as np
import copy
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


import wandb


wandb.init(project="semester-project")
wandb.config["more"] = "custom"



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class PolicyNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()


class ValueNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(ValueNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
                
        # Actor Network 

        self.actor_local = PolicyNetwork(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     

        # Critic Network (w/ Target Network)

        self.critic1 = ValueNetwork(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = ValueNetwork(state_size, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = ValueNetwork(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = ValueNetwork(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    
    def get_action(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(np.array(state)).float().to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


def act(cur_state):
    #network out put:
    raw_action = agent.get_action(cur_state)

    #scale of actions
    if raw_action == 0:
        det_action = np.asarray([0.002,0,0,0,0,0])
    elif raw_action == 1:
        det_action = np.asarray([-0.002,0,0,0,0,0])
    elif raw_action == 2:
        det_action = np.asarray([0,0.002,0,0,0,0])
    elif raw_action == 3:
        det_action = np.asarray([0,-0.002,0,0,0,0])
    elif raw_action == 4:
        det_action = np.asarray([0,0,0.002,0,0,0])
    elif raw_action == 5:
        det_action = np.asarray([0,0,-0.002,0,0,0])
    elif raw_action == 6:
        det_action = np.asarray([0,0,0,2,0,0])
    elif raw_action == 7:
        det_action = np.asarray([0,0,0,-2,0,0])
    elif raw_action == 8:
        det_action = np.asarray([0,0,0,0,2,0])
    elif raw_action == 9:
        det_action = np.asarray([0,0,0,0,-2,0])
    elif raw_action == 10:
        det_action = np.asarray([0,0,0,0,0,2])
    elif raw_action == 11:
        det_action = np.asarray([0,0,0,0,0,-2])
    else :
        det_action = np.asarray([0,0,0,0,0,0])
    return raw_action, det_action



action = [0,0,0,0,0,0]
raw_action = 100
next_state = [0,0,0,0,0,0,0]
reward = -100.0
done = False
action_srv = [0,0,0,0,0,0]
org_action_srv = [0,0,0,0,0,0]
raw_action_srv = 100
loss = 0.1

def step(req):
    global action, raw_action
    response = actionSrvResponse()
    response.action = np.array(action)#action
    response.raw_action = raw_action
    return response

def step2():
    set_ns_r_name = '/next_state_reward'
    rospy.wait_for_service(set_ns_r_name)
    set_ns_r = rospy.ServiceProxy(set_ns_r_name, nextSrv)
    try:
        resp1 = set_ns_r(1)
    except:
        return np.zeros((7,)), 0, 0, np.zeros((6,)), 100, 0
    return resp1.next_state, resp1.reward, resp1.done, resp1.action_srv, resp1.raw_action_srv, resp1.loss


if __name__ == "__main__":
    actor_checkpoint = "/home/airlab/robot_ws/src/franka_cal_sim_single/python/checkpoints/actor_checkpoint"
 
    action_dim = 13
    state_dim  = 7
    hidden_dim = 256
    rospy.init_node('RL_client', anonymous=True)

    agent = SAC(state_dim, action_dim, device = device)
    buffer = ReplayBuffer(buffer_size = 100000, batch_size = 128, device=device)


    max_frames  = 20000
    max_steps   = 10
    frame_idx   = 0
    rewards     = []

    reward_list = []
    episode = 0

    state = np.array([0,0,0,0,0,0,0])
    episode_reward = 0
    last = 0
    done = False
    i = 0
    j = 0
    org_reward = -100
    org_loss = 0.1
    rate = rospy.Rate(1)
    s = rospy.Service('/action_pose', actionSrv, step)
    

    while not rospy.is_shutdown():
        if frame_idx >= max_frames:
            rospy.signal_shutdown('Shutting Down')
        
        if i==0 and j==0:
            state, org_reward, done, action_srv1, raw_action_srv, loss = step2()
            if state[0] == 0 and state[1] == 0 and state[2] == 0 and state[3] == 0 and state[4] == 0 and state[5] == 0:
                continue
            if state[0] == next_state[0] and state[1] == next_state[1] and state[2] == next_state[2] and state[3] == next_state[3] and state[4] == next_state[4] and state[5] == next_state[5]:
                continue
            j = 1
        
        raw_action, action = act(state)
        
        step(None)
        next_state, reward, done, action_srv, raw_action_srv, loss = step2()

        next_state = np.asarray(next_state)
 


        if raw_action_srv == 100:
            continue

        if next_state[0] == 0 and next_state[1] == 0 and next_state[2] == 0 and next_state[3] == 0 and next_state[4] == 0 and next_state[5] == 0:
            continue
        
        if org_action_srv[0] != 12:
            if state[0] == next_state[0] and state[1] == next_state[1] and state[2] == next_state[2] and state[3] == next_state[3] and state[4] == next_state[4] and state[5] == next_state[5]:
                continue
        
        if org_action_srv[0] != 12:
            if org_action_srv[0] == action_srv[0] and org_action_srv[1] == action_srv[1] and org_action_srv[2] == action_srv[2] and org_action_srv[3] == action_srv[3] and org_action_srv[4] == action_srv[4] and org_action_srv[5] == action_srv[5]:
                continue
    

        if raw_action_srv == 12:
            done = True
        else:
            done = False

        if i == max_steps-1:
            done = True


        buffer.add(state, raw_action_srv, reward, next_state, done)
        if len(buffer) > 128:
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(i, buffer.sample(), gamma=0.99)
        
        rospy.loginfo(rospy.get_caller_id() + 'got state %s',state)
        
        
        org_action_srv = action_srv
        org_reward = reward
        org_loss = loss
        state = next_state
        episode_reward += reward
        frame_idx += 1
        i += 1

        rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
        rospy.loginfo(rospy.get_caller_id() + 'got action %s',action_srv)
        rospy.loginfo(rospy.get_caller_id() + 'got raw action %s',raw_action_srv)
        rospy.loginfo(rospy.get_caller_id() + 'got next state %s',next_state)


        if done or i == (max_steps):
            print("steps: "+str(frame_idx))
            wandb.log({"total_rewards": episode_reward})
            wandb.log({"episode length": i})
            print("total_rewards: "+str(episode_reward))

        if episode%10==0:
            torch.save(agent.actor_local.state_dict(), actor_checkpoint)
            if len(buffer) > 128:
                wandb.log({"ave_rewards": sum(rewards[-10:])/10})
                wandb.log({"policy_loss": policy_loss})
                wandb.log({"alpha_loss": alpha_loss})
                wandb.log({"bellmann_error1": bellmann_error1})
                wandb.log({"bellmann_error2": bellmann_error2})
                wandb.log({"current_alpha": current_alpha})

        
        if done or i>=max_steps:
            state = np.array([0,0,0,0,0,0,0])
            rewards.append(episode_reward)
            episode_reward = 0
            last = 0
            done = False
            episode += 1
            i = 0
            j = 0
            org_reward = -100
            org_loss = 0.1

        rate.sleep()