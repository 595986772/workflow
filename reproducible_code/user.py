from utils import generate_random_dag, generate_random_pos,update_single_user_position
import numpy as np
from task import Task
#from solver import Optimizer
from agent import Agent
import copy
import networkx as nx
import tracemalloc
import matplotlib.pyplot as plt
import time
import torch
from input import GCN_paramaters as gcn_paramaters
import json
from critical_path_reward import (
    CAUSAL_MAKESPAN_REWARD_MODES,
    CausalMakespanIncrementReward,
    CriticalPathPotential,
    compose_reward,
)
from hindsight_critical_path import (
    bottleneck_contribution_scores,
    posterior_critical_path,
)
from sample_efficient_guidance import (
    adaptive_guidance_probability,
    compute_nominal_upward_ranks,
    normalized_remote_predecessor_data,
    normalized_task_features,
    q_margin_confidence,
)


DAG_COMPLETION_PROTOCOL_VERSION = "all_real_tasks_max_exit_v2"

def generate_task_features(tasks, numberofservices):
    node_features = []
    for task in tasks.values():
        service_one_hot = [0] * numberofservices
        if 1 <= task.service <= numberofservices:
            service_one_hot[task.service - 1] = 1
        node_features.append([
            task.cpu_cycle / 1e6,
            task.input_data_length,
            len(task.predecessors),
            len(task.successors),
        ] + service_one_hot)
    return torch.tensor(node_features, dtype=torch.float)

def generate_edge_index(graph):
    edge_index = [[float(u) - 1, float(v) - 1] for u, v in graph.edges()]
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

class DecisionParameters:
    def __init__(self,numberofservices):
        self.distances={}
        self.servers_per_services={}
        self.queue_length={}
        self.cpu_freq = {}
        for n in range(numberofservices+1):
            self.servers_per_services[n]=[]

class User:
    def __init__(self,env,id,pos0,application_graph, max_cpu_cycles,max_data_length,numberofservices,acc,v0,power,bandwidth,numberofservers,deadline,learning_arguments,algorithm,filename_png,arrival_time=0.0):
        
        self.simulator = None
        self.env=env
        self.id = id
        self.numberofservers = numberofservers
        self.numberofservices = numberofservices
        N0 = 1.3803*290*(10**-23)*bandwidth
        self.snr = power/N0
        self.bw = bandwidth
        self.acc=acc #Acceleration
        self.v0=v0 # Velocity 0
        self.to_servers_rate={}
        #self.agent = Agent(algorithm = algorithm,learning_arguments = learning_arguments,numberofservers  = numberofservers,numberofservices = numberofservices,max_cpu_cycles = max_cpu_cycles,max_data_length=max_data_length,filename_png=None)
        self.gamma=0.9
        self.arrival_time = float(arrival_time)
        if not np.isfinite(self.arrival_time) or self.arrival_time < 0.0:
            raise ValueError(
                "arrival_time must be finite and non-negative"
            )
        # Generate random task DAG for each user
        self.max_cpu_cycles = max_cpu_cycles
        self.max_data_length = max_data_length
        self.numberoftasks = self.task_generate(application_graph,max_cpu_cycles,max_data_length,numberofservices,acc,v0,power,bandwidth,numberofservers)
        (
            self.nominal_upward_ranks,
            self.maximum_nominal_rank,
        ) = compute_nominal_upward_ranks(
            self.tasks_init,
            self.max_cpu_cycles,
            self.max_data_length,
        )
        self.deadline = deadline
        self.nearest_server = 0
        # define user position
        #self.pos = generate_random_pos(xlim,ylim)
        self.pos0 = pos0
        self.pos = pos0
        #self.decisionparameters=DecisionParameters(self.numberofservices) # initialize DecisionParameters
        self.reset()
        self.complete = False
        self.time_step=0

        self.node_features = generate_task_features(self.tasks_init, self.numberofservices)
        self.edge_index =  generate_edge_index(self.DAG)
    def setpos0(self):
        self.pos = self.pos0
    #def set_agent(self,agent):
        #self.agent = agent
    def find_nearest_server(self,servers):
        distances={}
        for s in servers.values():
            # calculate the distance between the user and servers
            distances[s.id]=np.sqrt(np.power(self.pos[0]-s.pos[0],2)+np.power(self.pos[1]-s.pos[1],2))
        nearest_server = min(distances, key=distances.get)
        #print(f'nearest server user {self.id} at position {self.pos[0],self.pos[1]}: {self.nearest_server}')
        self.rate_to_gateway = self.bw*np.log2(1+self.snr/np.power(distances[nearest_server],2)) #TODO calculate the rate
        for s in servers.values():
            self.to_servers_rate[s.id] = self.bw*np.log2(1+self.snr/np.power(distances[s.id],2)) #TODO calculate the rate
        
        return nearest_server
    def adjust_deadline(self,newvalue):
        #self.deadline = 1.1*min(newvalue,self.deadline)
        self.deadline = self.deadline+0.1*(newvalue - self.deadline)
        #self.deadline = self.deadline #TODO I removed deadline update in this line
    def task_generate(self,DAG,max_cpu_cycles,max_data_length,numberofservices,acc,v0,power,bandwidth,numberofservers):
        # Read dictionary of graphs from JSON file 
        self.DAG = nx.DiGraph(DAG)
        self.tasks_init={}
        for edge in DAG.edges:
            self.DAG[edge[0]][edge[1]]['datalength'] = int(self.DAG[edge[0]][edge[1]]['datalength'] *max_data_length)
        self.DAG2=self.DAG.copy()
        topological_order = nx.topological_sort(self.DAG)
        for i in list(topological_order):
            node = self.DAG.nodes[i]
            if node['service']>0:
                self.tasks_init[i] = Task(user_id = self.id,tasknumber=i,cpu_cycles = int(max_cpu_cycles*node['cpucycle']),service=int(numberofservices*(node['service']-1))+1,DAG=self.DAG)
                self.DAG2.nodes[i]['cpucycle'] = int(max_cpu_cycles*node['cpucycle'])
                self.DAG2.nodes[i]['service'] = int(numberofservices*(node['service']-1))+1
                self.DAG2.nodes[i]['input']=self.tasks_init[i].input_data_length
        self.DAG.remove_node('0')

        self.tasks = copy.deepcopy(self.tasks_init)
        self.exit_task_ids = tuple(
            task_id
            for task_id, task in self.tasks_init.items()
            if not task.successors
        )
        if not self.exit_task_ids:
            raise ValueError("Application DAG must contain a real exit task")
        self.lasttask = i
        # Plot the DAG with node and edge attributes
        if False:
            plt.figure(figsize=(12, 8))
            pos = nx.nx_agraph.graphviz_layout(self.DAG2, prog="dot")  # Requires pygraphviz
            nx.draw(self.DAG2, pos, with_labels=False, node_color='skyblue', node_size=2000, edge_color='gray', linewidths=1, font_size=15, font_color='black', font_weight='bold', arrows=True)

            # Draw node attributes
            node_labels = {node: f"{node}\n{self.DAG2.nodes[node]}" for node in self.DAG2.nodes}
            nx.draw_networkx_labels(self.DAG2, pos, labels=node_labels, font_size=10)

            # Draw edge attributes
            edge_labels = {(u, v): f"{self.DAG2[u][v]['datalength']}" for u, v in self.DAG2.edges}
            nx.draw_networkx_edge_labels(self.DAG2, pos, edge_labels=edge_labels, font_size=10)

            plt.title("DAG with Node and Edge Attributes")
            plt.savefig(f"dags/DAG_with_Node_and_Edge_Attributes_user_{self.id}.png")
            plt.close()
        return len(self.DAG.nodes.items())

    def reset(self):
        self.tasks = copy.deepcopy(self.tasks_init)
        for task in self.tasks.values():
            if not task.predecessors:
                task.result.ready_time = self.arrival_time
        self.ongoing_tasks = {}
        self.done_tasks = {}
        self.task_completion_counts = {
            task_id: 0 for task_id in self.tasks_init
        }
        self.finish_time_of_application = float('inf')
        self.episode_decision_count = 0
        self.episode_policy_actions = 0
        self.episode_cache_guidance_actions = 0
        self.episode_feedback_guidance_actions = 0
        self.episode_adaptive_cache_actions = 0
        self.episode_guidance_ready_count = 0
        self.episode_history_update_count = 0
        self.episode_q_confidence_sum = 0.0
        self.episode_q_confidence_count = 0
        self.episode_expert_confidence_sum = 0.0
        self.episode_expert_confidence_count = 0
        self.episode_guidance_probability_sum = 0.0
        self.episode_reward_sum = 0.0
        self.episode_reward_identity_error = 0.0
        self.episode_potential_initial = 0.0
        self.episode_potential_final = 0.0
        self.episode_hcpr_exact_path_tasks = 0
        self.episode_hcpr_total_tasks = 0
        self.episode_hcpr_criticality_sum = 0.0
        self.episode_hcpr_selected_tasks = 0
        self.critical_path_potential = None
        if (
            self.simulator is not None
            and self.simulator.reward_mode
            in CAUSAL_MAKESPAN_REWARD_MODES
        ):
            self.critical_path_potential = (
                CausalMakespanIncrementReward(
                    deadline=self.simulator.reward_scale,
                    application_origin=self.arrival_time,
                )
            )
            self.episode_potential_initial = (
                self.critical_path_potential.initial
            )
            self.episode_potential_final = (
                self.critical_path_potential.current
            )
        elif (
            self.simulator is not None
            and self.simulator.reward_mode
            in {
                'critical_path_potential',
                'terminal_plus_potential',
            }
        ):
            self.critical_path_potential = CriticalPathPotential(
                tasks=self.tasks_init,
                deadline=self.simulator.reward_scale,
                max_cpu_cycles=self.max_cpu_cycles,
                max_data_length=self.max_data_length,
            )
            self.episode_potential_initial = (
                self.critical_path_potential.initial
            )
            self.episode_potential_final = (
                self.critical_path_potential.current
            )
        self._last_decision_context = None

    def step(self,ts): #TODO
        #self.pos = (self.pos[0]+0.5*self.acc[0]*np.power(ts,2)+self.v0[0]*ts,self.pos[1]+0.5*self.acc[1]*np.power(ts,2)+self.v0[1]*ts)
        self.pos = update_single_user_position(self.simulator.xlim,self.simulator.ylim,self.pos, self.simulator.user_directions[self.id], self.simulator.street_positions_x, self.simulator.street_positions_y, self.simulator.vmax, self.simulator.vmin, self.simulator.street_width, ts)

        new_nearest_server = self.find_nearest_server(self.simulator.servers)
        # if new_nearest_server != self.nearest_server:
        #     trainingserver_coef = 0.1
        #     federatedlearning_server_coeff = self.simulator.input_dict["federated_learning_param_server"]  # e.g., 0.5

        #     # Get the local model's state_dict
        #     local_state_dict = self.agent.agent.TrainNet.model.state_dict()

        #     # Update weights of the previous nearest server
        #     # Assume servers have an update_weights method that accepts a state_dict and a coefficient
        #     self.simulator.servers[self.nearest_server].update_weights(local_state_dict, trainingserver_coef)

        #     # Update the nearest server index
        #     self.nearest_server = new_nearest_server

        #     # Get the new nearest server's model state_dict
        #     new_server_state_dict = self.simulator.servers[self.nearest_server].agent.agent.TrainNet.model.state_dict()

        #     # Adjust the user's model weights using weights_adjustment function
        #     self.weights_adjustement(new_server_state_dict, federatedlearning_server_coeff)
        self.nearest_server = new_nearest_server
        task = self.decision_on_task()
        # self.getinformation(self.simulator.servers)
        #self.task_embeddings = self.simulator.servers[self.nearest_server].task_gcn(self.node_features, self.edge_index)
        self.task_embeddings = None
        observation= self.simulator.servers[self.nearest_server].agent.state(task,self.tasks_init,self.done_tasks,self.deadline,self.simulator.server_service_info,nearest_server_id  = self.nearest_server,user_direction=self.simulator.user_directions[self.id],DAG=self.DAG,Embeddings=self.task_embeddings,server_quality=self.simulator.broker.causal_server_quality())

        return task,observation
    

    # def getinformation(self,servers):
    #     for s in servers.values():
    #         for service in s.services:
    #             self.decisionparameters.servers_per_services[service].append(s.id)
    #         # store the queue length as a decision parameter

    #         self.decisionparameters.queue_length[s.id]=sum([task.cpu_cycle for task in s.task_queue])
    #         self.decisionparameters.cpu_freq[s.id] = s.frequency

    #def observe(self,task,servers,nearest_server_id):
       #return self.agent.state(task,self.tasks_init,self.done_tasks,self.deadline,self.simulator.server_service_info,nearest_server_id,DAG=self.DAG,Embeddings=self.task_embeddings)   

    def decision_on_task(self):
        for task0 in self.ongoing_tasks.values():
            if task0.assigned_server == -1:
                return task0
        task = None
        if len(self.tasks.values()):
            t = min(self.tasks.values(), key=lambda obj: obj.result.ready_time)
            task = t
            self.tasks.pop(t.task_number)
            self.ongoing_tasks[t.task_number]=t
        return task
    
    def decision_on_task1(self):
        # Scheduling
        for task0 in self.ongoing_tasks.values():
            if task0.assigned_server == -1:
                return task0
        task=None
        for t in self.tasks.values():
            # scheduling based on the first task that does not have any predecessor
            if (len(t.not_done_predecessors)==0):
                task = t
                self.tasks.pop(t.task_number)
                self.ongoing_tasks[t.task_number]=t
                break
        return task
        # decision on choosing a server

    def cache_hit_vector(self, task):
        if task.service <= 0:
            return np.zeros(self.numberofservers, dtype=bool)
        return (
            self.simulator.server_service_info[:, task.service - 1] > 0.5
        )

    def remote_data_vector(self, task):
        return np.asarray(
            [
                normalized_remote_predecessor_data(
                    task=task,
                    done_tasks=self.done_tasks,
                    candidate_server=server_id,
                    max_data_length=self.max_data_length,
                )
                for server_id in range(self.numberofservers)
            ],
            dtype=float,
        )

    def update_history_feedback(self, task):
        context = self._last_decision_context
        if (
            not self.simulator.training
            or not self.simulator.history_feedback_guidance
            or context is None
        ):
            return

        guide = self.simulator.servers[
            context["source_server"]
        ].history_feedback_guide
        if guide is None:
            return

        predecessor_ready = max(
            (
                self.done_tasks[predecessor_id].result.finish_time
                for predecessor_id in task.predecessors
            ),
            default=0.0,
        )
        observed_path_delay = max(
            0.0,
            task.result.finish_time - predecessor_ready,
        )
        guide.update(
            server_id=context["action"],
            cache_hit=context["cache_hit"],
            normalized_cpu=context["normalized_cpu"],
            normalized_input=context["normalized_input"],
            normalized_remote_data=context["normalized_remote_data"],
            observed_path_delay=observed_path_delay,
        )
        self.episode_history_update_count += 1

    def decision_on_server(self,observation,task):
        self._last_decision_context = None
        if task is None:
            return -1
        else:
            if self.simulator.alg == "random":
                return np.random.choice(self.numberofservers)
            if self.simulator.alg == "nearest_server":
                return self.nearest_server
            if self.simulator.alg == "nearest_with_service":
                candidate_servers = [
                    server
                    for server in self.simulator.servers.values()
                    if task.service in server.services
                ]
                if not candidate_servers:
                    return self.nearest_server
                return min(
                    candidate_servers,
                    key=lambda server: np.linalg.norm(
                        np.asarray(self.pos) - np.asarray(server.pos)
                    ),
                ).id
            #return np.argmax([self.optimizer.optimal_x[int(task.task_number),s] for s in range(self.numberofservers)])
           # return self.nearest_server
            #return 0 #TODO Cloud server
            
            
            #for s in self.simulator.servers.values():
            #    for service in s.services:
            #        if service not in servers_per_services:
            #            servers_per_services[service] = []
            #        servers_per_services[service].append(s.id)
            #if task.service not in servers_per_services:
                #print(f"Service {task.service} not available in any server.")
            #    return np.random.choice(self.numberofservers)
            #return np.random.choice(servers_per_services[task.service])
            #return min(selected_servers_distances, key=selected_servers_distances.get) #TODO minimum distance regarding to service
            #return min(self.decisionparameters.distances, key=self.decisionparameters.distances.get) #TODO minimum distance
            #return np.random.choice(self.numberofservers) #TODO RANDOM ACTION

        
            else:
                source_server_id = self.nearest_server
                source_server = self.simulator.servers[source_server_id]
                cache_hits = self.cache_hit_vector(task)
                remote_data = None
                selected_action = None

                if (
                    self.simulator.training
                    and (
                        self.simulator.history_feedback_guidance
                        or self.simulator.adaptive_guidance_gate
                    )
                ):
                    q_confidence = 0.0
                    if self.simulator.adaptive_guidance_gate:
                        with torch.no_grad():
                            q_values = (
                                source_server.agent.agent.TrainNet.predict(
                                    observation
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        q_confidence = q_margin_confidence(q_values)
                        self.episode_q_confidence_sum += q_confidence
                        self.episode_q_confidence_count += 1

                    guidance_action = None
                    guidance_candidates = None
                    expert_confidence = 0.0
                    guidance_ready = False
                    if self.simulator.history_feedback_guidance:
                        remote_data = self.remote_data_vector(task)
                        normalized_cpu, normalized_input = (
                            normalized_task_features(
                                task,
                                self.max_cpu_cycles,
                                self.max_data_length,
                            )
                        )
                        guide = source_server.history_feedback_guide
                        recommendation = guide.recommend(
                            cache_hits=cache_hits,
                            normalized_cpu=normalized_cpu,
                            normalized_input=normalized_input,
                            remote_data_by_server=remote_data,
                        )
                        if recommendation["ready"]:
                            guidance_ready = True
                            guidance_action = recommendation["action"]
                            expert_confidence = recommendation["confidence"]
                    else:
                        guidance_candidates = np.flatnonzero(cache_hits)
                        if guidance_candidates.size:
                            guidance_ready = True
                            expert_confidence = 1.0

                    if guidance_ready:
                        self.episode_guidance_ready_count += 1
                        self.episode_expert_confidence_sum += (
                            expert_confidence
                        )
                        self.episode_expert_confidence_count += 1
                        criticality = (
                            self.nominal_upward_ranks[task.task_number]
                            / self.maximum_nominal_rank
                        )
                        if self.simulator.adaptive_guidance_gate:
                            dqn_agent = source_server.agent.agent
                            epsilon_range = max(
                                dqn_agent.initial_epsilon
                                - dqn_agent.min_epsilon,
                                1e-12,
                            )
                            handoff_factor = np.clip(
                                (
                                    dqn_agent.epsilon
                                    - dqn_agent.min_epsilon
                                )
                                / epsilon_range,
                                0.0,
                                1.0,
                            )
                            guidance_probability = (
                                adaptive_guidance_probability(
                                    q_confidence=q_confidence,
                                    task_criticality=criticality,
                                    expert_confidence=expert_confidence,
                                    maximum_probability=(
                                        self.simulator
                                        .history_feedback_max_probability
                                    ),
                                    handoff_factor=handoff_factor,
                                )
                            )
                        else:
                            guidance_probability = (
                                self.simulator
                                .history_feedback_fixed_probability
                                * expert_confidence
                            )
                        self.episode_guidance_probability_sum += (
                            guidance_probability
                        )
                        if np.random.rand() < guidance_probability:
                            if guidance_action is not None:
                                selected_action = int(guidance_action)
                                self.episode_feedback_guidance_actions += 1
                            else:
                                selected_action = int(
                                    np.random.choice(guidance_candidates)
                                )
                                self.episode_adaptive_cache_actions += 1

                if (
                    selected_action is None
                    and self.simulator.training
                    and np.random.rand() < self.simulator.beta
                ):
                    candidate_servers = np.flatnonzero(cache_hits)
                    if candidate_servers.size:
                        selected_action = int(
                            np.random.choice(candidate_servers)
                        )
                    else:
                        selected_action = int(
                            np.random.choice(self.numberofservers)
                        )
                    self.episode_cache_guidance_actions += 1

                if selected_action is None:
                    selected_action = int(
                        source_server.offloading_desicion(observation, task)
                    )
                    self.episode_policy_actions += 1

                if (
                    self.simulator.training
                    and self.simulator.history_feedback_guidance
                ):
                    if remote_data is None:
                        remote_data = self.remote_data_vector(task)
                    normalized_cpu, normalized_input = (
                        normalized_task_features(
                            task,
                            self.max_cpu_cycles,
                            self.max_data_length,
                        )
                    )
                    self._last_decision_context = {
                        "source_server": int(source_server_id),
                        "action": int(selected_action),
                        "cache_hit": bool(cache_hits[selected_action]),
                        "normalized_cpu": float(normalized_cpu),
                        "normalized_input": float(normalized_input),
                        "normalized_remote_data": float(
                            remote_data[selected_action]
                        ),
                    }
                return selected_action


            x_values={
                (0,0,'1',0) : 1.0, 
                (0,0,'1',1) : 0.0,
                (0,0,'1',2) : 0.0,
                (0,0,'2',0) : 0.0,
                (0,0,'2',1) : 0.0,
                (0,0,'2',2) : 1.0,
                (0,0,'3',0) : 0.0,
                (0,0,'3',1) : 0.0,
                (0,0,'3',2) : 1.0,
                (0,0,'4',0) : 0.0,
                (0,0,'4',1) : 0.0,
                (0,0,'4',2) : 1.0,
                (0,0,'5',0) : 0.0,
                (0,0,'5',1) : 0.0,
                (0,0,'5',2) : 1.0,
                (0,0,'6',0) : 0.0,
                (0,0,'6',1) : 0.0,
                (0,0,'6',2) : 1.0,
                (0,0,'7',0) : 0.0,
                (0,0,'7',1) : 1.0,
                (0,0,'7',2) : 0.0,
                (0,0,'8',0) : 0.0,
                (0,0,'8',1) : 0.0,
                (0,0,'8',2) : 1.0,
                (0,1,'1',0) : 0.0,
                (0,1,'1',1) : 0.0,
                (0,1,'1',2) : 1.0,
                (0,1,'2',0) : 0.0,
                (0,1,'2',1) : 0.0,
                (0,1,'2',2) : 1.0,
                (0,1,'3',0) : 0.0,
                (0,1,'3',1) : 0.0,
                (0,1,'3',2) : 1.0,
                (0,1,'4',0) : 0.0,
                (0,1,'4',1) : 0.0,
                (0,1,'4',2) : 1.0,
                (0,1,'5',0) : 0.0,
                (0,1,'5',1) : 0.0,
                (0,1,'5',2) : 1.0,
                (0,2,'1',0) : 0.0,
                (0,2,'1',1) : 1.0,
                (0,2,'1',2) : 0.0,
                (0,2,'2',0) : 0.0,
                (0,2,'2',1) : 1.0,
                (0,2,'2',2) : 0.0,
                (0,2,'3',0) : 0.0,
                (0,2,'3',1) : 1.0,
                (0,2,'3',2) : 0.0,
                (0,2,'4',0) : 0.0,
                (0,2,'4',1) : 1.0,
                (0,2,'4',2) : 0.0,
                (0,2,'5',0) : 0.0,
                (0,2,'5',1) : 1.0,
                (0,2,'5',2) : 0.0,
                (0,2,'6',0) : 1.0,
                (0,2,'6',1) : 0.0,
                (0,2,'6',2) : 0.0,
                (0,3,'1',0) : 0.0,
                (0,3,'1',1) : 0.0,
                (0,3,'1',2) : 1.0,
                (0,3,'2',0) : 1.0,
                (0,3,'2',1) : 0.0,
                (0,3,'2',2) : 0.0,
                (0,3,'3',0) : 0.0,
                (0,3,'3',1) : 1.0,
                (0,3,'3',2) : 0.0,
                (0,3,'4',0) : 0.0,
                (0,3,'4',1) : 1.0,
                (0,3,'4',2) : 0.0,
                (0,3,'5',0) : 0.0,
                (0,3,'5',1) : 1.0,
                (0,3,'5',2) : 0.0,
                (0,4,'1',0) : 0.0,
                (0,4,'1',1) : 1.0,
                (0,4,'1',2) : 0.0,
                (0,4,'2',0) : 0.0,
                (0,4,'2',1) : 1.0,
                (0,4,'2',2) : 0.0,
                (0,4,'3',0) : 0.0,
                (0,4,'3',1) : 1.0,
                (0,4,'3',2) : 0.0,
                (0,4,'4',0) : 0.0,
                (0,4,'4',1) : 0.0,
                (0,4,'4',2) : 1.0,
                (0,4,'5',0) : 1.0,
                (0,4,'5',1) : 0.0,
                (0,4,'5',2) : 0.0,
                (0,4,'6',0) : 0.0,
                (0,4,'6',1) : 1.0,
                (0,4,'6',2) : 0.0,
                (0,4,'7',0) : 0.0,
                (0,4,'7',1) : 1.0,
                (0,4,'7',2) : 0.0,
                (0,4,'8',0) : 0.0,
                (0,4,'8',1) : 1.0,
                (0,4,'8',2) : 0.0,
                (0,4,'9',0) : 0.0,
                (0,4,'9',1) : 1.0,
                (0,4,'9',2) : 0.0,                          
                        }
            return np.argmax([x_values[(0,self.id,task.task_number,s)] for s in range(self.numberofservers)])


    def weights_adjustement(self, broker_state_dict, coeff):
        # Get the current model's state_dict
        local_state_dict = self.simulator.servers[self.nearest_server].agent.agent.TrainNet.model.state_dict()

        # Create a new state_dict to store the averaged weights
        averaged_state_dict = {}

        # Iterate over the parameter names and values
        for name, local_param in local_state_dict.items():
            # Get the corresponding parameter from the broker's model
            broker_param = broker_state_dict[name]

            # Ensure both parameters are on the same device
            broker_param = broker_param.to(local_param.device)

            # Compute the averaged parameter
            averaged_param = (1 - coeff) * local_param + coeff * broker_param

            # Store the averaged parameter in the new state_dict
            averaged_state_dict[name] = averaged_param

        # Load the new averaged state_dict into the local model
        self.simulator.servers[self.nearest_server].agent.agent.TrainNet.model.load_state_dict(averaged_state_dict)

        # Update the target model if necessary
        self.simulator.servers[self.nearest_server].agent.agent.update_target_model()

    def weights_adjustement0(self,broker_weitghs,coeff):
        average_weights = [(wa*(1-coeff) + wb*coeff) for wa, wb in zip(self.simulator.servers[self.nearest_server].agent.agent.TrainNet.model.get_weights(), broker_weitghs)]
        self.simulator.servers[self.nearest_server].agent.agent.TrainNet.model.set_weights(average_weights)
        #self.agent.agent.TrainNet.model.set_weights(broker_weitghs)
        self.simulator.servers[self.nearest_server].agent.agent.update_target_model()

    def attach(self,simulator):
        self.simulator=simulator

    def run(self):
        #start_time = time.time()  # Start time for the iteration
        if self.arrival_time > self.env.now:
            yield self.env.timeout(self.arrival_time - self.env.now)
        timestep = 0.010
        task,observation=self.step(timestep) #user movement       
        self.time_step += 1
        done_list = False
        self.complete = False
        trajectory=[]
        self.sum_computinglatency=0
        self.sum_datatransferlatency=0
        self.sum_predlatency = 0
        self.sum_servicelatency = 0
        self.sum_waiting_latency = 0

        while (not self.complete):
            self.episode_decision_count += 1
            action = self.decision_on_server(observation,task)
            self.simulator.servers[action].numberofconnectedusers += 1
            #print(f'User {self.id} task {task.task_number} assigned to server {action} and the service is {task.service} and number of connected users is {self.simulator.servers[action].numberofconnectedusers}')
            if task.predecessors:
                task.result.pred_latency = max([self.done_tasks[t].result.finish_time + self.simulator.between_server_costs[self.done_tasks[t].assigned_server,action]*self.done_tasks[t].outputs_length[task.task_number] for t in task.predecessors])
            else:
                task.result.pred_latency = self.arrival_time

            if task.input_data_length>0:
                    task.result.data_transfer_latency = task.input_data_length* ((1.0/ self.rate_to_gateway)+self.simulator.between_server_costs[self.nearest_server,action])
            else:
                task.result.data_transfer_latency = 0 
            
            self.simulator.servers[action].execute_task(task)

            if task.completion_event is not None:
                yield task.completion_event
            else:
                while(not task.done):
                    yield self.env.timeout(1)

            self.sum_computinglatency+=task.result.computing_latency
            self.sum_datatransferlatency+=task.result.data_transfer_latency
            self.sum_predlatency +=task.result.pred_latency 
            self.sum_servicelatency += task.result.service_latency
            self.sum_waiting_latency += task.result.waiting_latency
            self.update_history_feedback(task)
            # add onr done task of server s to done tasks list of users
            if task.task_number in self.done_tasks:
                raise RuntimeError(
                    f"Task {task.task_number!r} completed more than once"
                )
            self.task_completion_counts[task.task_number] += 1
            self.done_tasks[task.task_number] = task
            for ts in task.successors:
                # remove predecessors
                self.tasks[ts].not_done_predecessors.remove(task.task_number)
                if (len(self.tasks[ts].not_done_predecessors)==0):
                        self.tasks[ts].result.ready_time = task.result.finish_time

            all_tasks_complete = (
                len(self.done_tasks) == len(self.tasks_init)
            )
            if all_tasks_complete:
                if set(self.done_tasks) != set(self.tasks_init):
                    raise RuntimeError(
                        "Completed task set does not match the application DAG"
                    )
                self.complete = True
                self.finish_time_of_application = max(
                    self.done_tasks[task_id].result.finish_time
                    for task_id in self.exit_task_ids
                )
                application_response_time = (
                    self.finish_time_of_application - self.arrival_time
                )

                if application_response_time < self.deadline:
                    base_reward = 1
                else:
                    base_reward = -1
                #print(f'User {self.id} finish time: {self.finish_time_of_application}')
                
                done = True
            else:
                #reward = - task.result.service_latency - task.result.waiting_latency - task.result.computing_latency - task.result.data_transfer_latency
                base_reward = 0
                done = False

            if self.critical_path_potential is not None:
                reward, _, potential_after = (
                    self.critical_path_potential.advance(
                        self.done_tasks,
                        terminal_finish_time=(
                            self.finish_time_of_application
                            if done
                            else None
                        ),
                    )
                )
                self.episode_potential_final = potential_after
                reward = compose_reward(
                    base_reward=base_reward,
                    potential_reward=reward,
                    reward_mode=self.simulator.reward_mode,
                    potential_weight=(
                        self.simulator.potential_reward_weight
                    ),
                )
            else:
                reward = base_reward

            self.episode_reward_sum += reward
            if (
                done
                and self.critical_path_potential is not None
            ):
                expected_return = (
                    self.critical_path_potential.expected_return(
                        self.finish_time_of_application
                    )
                )
                if (
                    self.simulator.reward_mode
                    == 'terminal_plus_potential'
                ):
                    expected_return = (
                        base_reward
                        + self.simulator.potential_reward_weight
                        * expected_return
                    )
                self.episode_reward_identity_error = abs(
                    self.episode_reward_sum - expected_return
                )

            if (
                done
                and self.simulator.training
                and self.simulator.updatedeadline
            ):
                self.adjust_deadline(application_response_time)

            task0=task
            task,next_observation= self.step(timestep)
            if (action>=0):
                trajectory.append(
                    {
                        's': observation,
                        'a': action,
                        'r': reward,
                        's2': next_observation,
                        'done': done,
                        'task_id': task0.task_number,
                        'node_features': self.node_features.clone(),
                        'edge_index': self.edge_index.clone(),
                        'embedding_index': int(task0.task_number) - 1,
                    }
                )
            observation = next_observation

        if any(
            count != 1
            for count in self.task_completion_counts.values()
        ):
            raise RuntimeError(
                "Every real task must complete exactly once"
            )
        
        if (
            trajectory
            and self.simulator.training
            and self.simulator.learning_enabled
        ):
            learning_agent = self.simulator.servers[
                self.nearest_server
            ].agent.agent
            if getattr(
                learning_agent,
                "uses_hindsight_critical_path_replay",
                False,
            ):
                posterior = posterior_critical_path(
                    self.done_tasks,
                    self.simulator.between_server_costs,
                    temperature=(
                        self.simulator.hcpr_temperature
                    ),
                )
                exact_path = set(posterior["path"])
                replay_scores = posterior["scores"]
                if (
                    getattr(
                        learning_agent,
                        "posterior_replay_mode",
                        "critical_path",
                    )
                    == "bottleneck_contribution"
                ):
                    replay_scores = (
                        bottleneck_contribution_scores(
                            posterior,
                            top_fraction=(
                                self.simulator.bcr_top_fraction
                            ),
                        )["scores"]
                    )
                for transition in trajectory:
                    task_id = transition["task_id"]
                    transition["posterior_criticality"] = (
                        replay_scores[task_id]
                    )
                self.episode_hcpr_exact_path_tasks = len(
                    exact_path
                )
                self.episode_hcpr_total_tasks = len(trajectory)
                self.episode_hcpr_criticality_sum = sum(
                    replay_scores[
                        transition["task_id"]
                    ]
                    for transition in trajectory
                )
                self.episode_hcpr_selected_tasks = sum(
                    replay_scores[transition["task_id"]] > 0.0
                    for transition in trajectory
                )
            if getattr(
                learning_agent,
                "uses_n_step_transitions",
                False,
            ):
                learning_agent.add_trajectory(trajectory)
            else:
                while trajectory:
                    return_gamma = (
                        1.0
                        if self.simulator.reward_mode
                        in {
                            'critical_path_potential',
                            'causal_critical_path',
                            'causal_makespan_increment',
                        }
                        else self.gamma
                    )
                    exp = trajectory.pop(-1)
                    if exp['done']:
                        G = exp['r']
                    else:
                        G = exp['r'] + return_gamma * G
                        exp['r'] = G
                        exp['done'] = True
                    replay_fields = (
                        learning_agent.TrainNet.experience
                    )
                    learning_agent.TrainNet.add_experience(
                        {
                            key: value
                            for key, value in exp.items()
                            if key in replay_fields
                        }
                    )



            #self.task_gcn.train()         # Add GCN training here
        #end_time = time.time()  # End time for the iteration
        #elapsed_time = end_time - start_time
        #print(f"user Iteration took {elapsed_time:.10f} seconds")

    """       Simulation_Time = self.time_step
        Average_of_finish_time = np.mean([self.users[m].finish_time_of_application for m in range(self.M)])
        minimum_of_finish_time = min([self.users[m].finish_time_of_application for m in range(self.M)])
        average_optimal_value = self.optimizer.optimal_objective
        optimal_values = [self.optimizer.optimal_finishtime[m] for m in range(self.M)]
        finish_times = [self.users[m].finish_time_of_application for m in range(self.M)]
        #print('Simulation Time: ', Simulation_Time)
        #print('Average of finish time: ', Average_of_finish_time)
        #print('minimum of finish time: ', minimum_of_finish_time)
        return optimal_values,finish_times 
    """    
