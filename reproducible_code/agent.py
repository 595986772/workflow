from dqn import DQN_Agent,GCN_DQN_Agent
from a2c import A2C_Agent
from task import Task
import numpy as np
import torch
import torch.nn as nn
from input import GCN_paramaters as gcn_paramaters
from critical_path_rl import (
    CAPQAgent,
    CP_RL_ALGORITHMS,
    CPQACAgent,
    CorrectDDQNAgent,
    PD3QNAgent,
    TELEMETRY_CP_RL_ALGORITHMS,
    structural_task_criticality,
    telemetry_channels_for_algorithm,
)
from discrete_sac import DiscreteSACAgent

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
except Exception:
    Data = None
    from_networkx = None

try:
    from lstm import LSTM_Agent
except Exception:
    LSTM_Agent = None

class Agent:
    def __init__(
        self,
        algorithm,
        learning_arguments,
        numberofservers,
        numberofservices,
        max_cpu_cycles,
        max_data_length,
        filename_png,
        task_dependency_features_enabled=True,
    ):
        self.alg = algorithm
        self.numberofservers = numberofservers
        self.numberofservices=numberofservices
        self.task_dependency_features_enabled = bool(
            task_dependency_features_enabled
        )
        self.action_size = numberofservers
        self. max_cpu_cycles = max_cpu_cycles
        self.max_data_length = max_data_length
        if self.alg in CP_RL_ALGORITHMS:
            self.state_size = (
                numberofservers * numberofservices
                + numberofservers
                + 2 * numberofservices
                + 4
            )
            self.telemetry_channels = (
                telemetry_channels_for_algorithm(self.alg)
            )
            self.state_size += (
                numberofservers * self.telemetry_channels
            )
            if self.alg.endswith('DDQN'):
                agent_class = CorrectDDQNAgent
            elif self.alg.endswith('CAPQ'):
                agent_class = CAPQAgent
            elif self.alg.endswith('PD3QN'):
                agent_class = PD3QNAgent
            elif self.alg.endswith('DiscreteSAC'):
                agent_class = DiscreteSACAgent
            else:
                agent_class = CPQACAgent
            self.agent = agent_class(
                num_states=self.state_size,
                num_actions=self.action_size,
                num_services=numberofservices,
                hidden_units=learning_arguments['hidden_units'],
                gamma=learning_arguments.get('gamma', 1.0),
                max_experiences=learning_arguments['max_experiences'],
                min_experiences=learning_arguments['min_experiences'],
                batch_size=learning_arguments['batch_size'],
                learning_rate=learning_arguments['learning_rate'],
                epsilon=learning_arguments['epsilon'],
                maximum_exploration=learning_arguments[
                    'maximum_exploration'
                ],
                n_step=learning_arguments.get('n_step', 3),
                num_quantiles=learning_arguments.get(
                    'num_quantiles',
                    16,
                ),
                risk_tail_fraction=learning_arguments.get(
                    'risk_tail_fraction',
                    0.25,
                ),
                entropy_coefficient=learning_arguments.get(
                    'entropy_coefficient',
                    0.02,
                ),
                sac_target_entropy_ratio=learning_arguments.get(
                    'sac_target_entropy_ratio',
                    0.98,
                ),
                sac_target_tau=learning_arguments.get(
                    'sac_target_tau',
                    0.005,
                ),
                priority_alpha=learning_arguments.get(
                    'priority_alpha',
                    0.6,
                ),
                priority_beta_start=learning_arguments.get(
                    'priority_beta_start',
                    0.4,
                ),
                priority_beta_anneal_steps=learning_arguments.get(
                    'priority_beta_anneal_steps',
                    2000,
                ),
                criticality_boost=learning_arguments.get(
                    'criticality_boost',
                    2.0,
                ),
                algorithm=self.alg,
            )
        elif self.alg == 'simpleDQN':
            self.state_size = 2
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'prev_servers_plus_service_per_serverDQN':
            self.state_size  = numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'prev_serversDQN':
            self.state_size  = numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'simpleA2C':   
            self.state_size = 2
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'prev_serversA2C':
            self.state_size  = numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'justserviceDQN':
            self.state_size  = 2*numberofservices+2+1  #current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'justserviceA2C':
            self.state_size  = 2*numberofservices+2+1  #current services+ next services + data + cpu+time
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'prev_serversLSTM':
            if LSTM_Agent is None:
                raise RuntimeError("The LSTM baseline requires a working TensorFlow/Keras installation.")
            self.state_size  = numberofservers+2*numberofservices+2+1  #previews servers + current services+ next services + data + cpu+time
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        
        elif self.alg == 'justserviceLSTM':
            if LSTM_Agent is None:
                raise RuntimeError("The LSTM baseline requires a working TensorFlow/Keras installation.")
            self.state_size  = 2*numberofservices+2+1  #current services+ next services + data + cpu+time
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'simpleLSTM':
            if LSTM_Agent is None:
                raise RuntimeError("The LSTM baseline requires a working TensorFlow/Keras installation.")
            self.state_size = 2
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'GAT':
            if from_networkx is None:
                raise RuntimeError("The GAT baseline requires a working torch-geometric installation.")
            self.state_size = 16
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverDQN':
            self.state_size  = numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverA2C':
            self.state_size  = numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  A2C_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverLSTM':
            if LSTM_Agent is None:
                raise RuntimeError("The LSTM baseline requires a working TensorFlow/Keras installation.")
            self.state_size  = numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  LSTM_Agent(num_states=self.state_size,num_actions=self.action_size,lstm_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'GCN_DQN':
            self.state_size  =gcn_paramaters['output_dim']   #previews servers + current services+ next services + data + cpu+time
            self.agent =  GCN_DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])

        elif self.alg == 'GCN':
            self.state_size = 4
            self.agent = GCN_Agent(num_states=self.state_size, num_actions=self.action_size, hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'], epsilon=learning_arguments['epsilon'], maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg == 'direction_nearestserver_prev_servers_plus_service_per_serverDQN':
            self.state_size  =numberofservers+numberofservers*numberofservices+numberofservers+2*numberofservices+2+1+1  #nearest server + previews servers + current services+ next services + data + cpu+time
            self.agent =  DQN_Agent(num_states=self.state_size,num_actions=self.action_size,hidden_units=learning_arguments['hidden_units'], gamma=learning_arguments['gamma'], max_experiences=learning_arguments['max_experiences'], min_experiences=learning_arguments['min_experiences'], batch_size=learning_arguments['batch_size'], lr=learning_arguments['learning_rate'],epsilon=learning_arguments['epsilon'],maximum_exploration=learning_arguments['maximum_exploration'])
        elif self.alg in {'random', 'nearest_server', 'nearest_with_service'}:
            self.state_size = 2
            self.agent = DQN_Agent(
                num_states=self.state_size,
                num_actions=self.action_size,
                hidden_units=learning_arguments['hidden_units'],
                gamma=learning_arguments['gamma'],
                max_experiences=learning_arguments['max_experiences'],
                min_experiences=learning_arguments['min_experiences'],
                batch_size=learning_arguments['batch_size'],
                lr=learning_arguments['learning_rate'],
                epsilon=learning_arguments['epsilon'],
                maximum_exploration=learning_arguments['maximum_exploration'],
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.alg}")

        #self.state_size  = 2  #current services+ next services + data + cpu #TODO STATE SIZE
        #self.agent.TrainNet.model.summary()
        #if filename_png:
           # plot_model(self.agent.TrainNet.model, to_file=filename_png+'/learning_model.png', show_shapes=True)

    
    def state(self,task,tasks,done_tasks,deadline,servers_service_info,gat=None,DAG=None,nearest_server_id=None,Embeddings=None,user_direction=None,server_quality=None):
        if self.alg in CP_RL_ALGORITHMS:
            state = torch.zeros(self.state_size)
            if task is None:
                return state

            server_service = torch.tensor(
                servers_service_info,
                dtype=torch.float,
            )
            predecessor_servers = torch.zeros(
                self.numberofservers,
                dtype=torch.float,
            )
            successor_services = torch.zeros(
                self.numberofservices,
                dtype=torch.float,
            )
            current_service = torch.zeros(
                self.numberofservices,
                dtype=torch.float,
            )
            if task.service > 0:
                current_service[task.service - 1] = 1.0
            if self.task_dependency_features_enabled:
                for successor_id in task.successors:
                    successor = tasks[successor_id]
                    if successor.service > 0:
                        successor_services[
                            successor.service - 1
                        ] = 1.0

            predecessor_finish = 0.0
            for predecessor_id in task.predecessors:
                predecessor = done_tasks[predecessor_id]
                if self.task_dependency_features_enabled:
                    predecessor_servers[
                        predecessor.assigned_server
                    ] = 1.0
                predecessor_finish = max(
                    predecessor_finish,
                    predecessor.result.finish_time,
                )

            if deadline > 0:
                state[0] = predecessor_finish / deadline
            state[1] = (
                task.cpu_cycle
                / self.max_cpu_cycles
                / (10**6)
            )
            state[2] = (
                task.input_data_length
                / self.max_data_length
            )
            index = 3
            state[index:index + server_service.numel()] = (
                server_service.view(-1)
            )
            index += server_service.numel()
            state[
                index:index + predecessor_servers.numel()
            ] = predecessor_servers
            index += predecessor_servers.numel()
            if self.alg in TELEMETRY_CP_RL_ALGORITHMS:
                telemetry_values = (
                    self.numberofservers * self.telemetry_channels
                )
                if server_quality is None:
                    quality = torch.ones(
                        telemetry_values,
                        dtype=torch.float,
                    )
                elif isinstance(server_quality, dict):
                    quality = torch.tensor(
                        [
                            value
                            for server_id
                            in range(self.numberofservers)
                            for value in np.atleast_1d(
                                server_quality[server_id]
                            )
                        ],
                        dtype=torch.float,
                    )
                else:
                    quality = torch.as_tensor(
                        server_quality,
                        dtype=torch.float,
                    )
                if quality.numel() != telemetry_values:
                    raise ValueError(
                        "server_quality must contain "
                        f"{self.telemetry_channels} value(s) per server"
                    )
                state[index:index + quality.numel()] = quality
                index += quality.numel()
            state[
                index:index + current_service.numel()
            ] = current_service
            index += current_service.numel()
            state[
                index:index + successor_services.numel()
            ] = successor_services
            if self.task_dependency_features_enabled:
                state[-1] = structural_task_criticality(
                    task.task_number,
                    tasks,
                )
            return state
        elif self.alg == 'simpleDQN' or self.alg == 'simpleA2C' or self.alg == 'simpleLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                state[0] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[1] = task.input_data_length / self.max_data_length
            return state

        elif self.alg == 'prev_servers_plus_service_per_serverDQN' or self.alg == 'prev_servers_plus_service_per_serverA2C'  or self.alg == 'prev_servers_plus_service_per_serverLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                server_service = torch.tensor(servers_service_info, dtype=torch.float)
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+server_service.numel()] = server_service.view(-1)
                state[3+server_service.numel():3+server_service.numel()+servers.numel()] = servers
                state[3+server_service.numel()+servers.numel():3+server_service.numel()+servers.numel()+service.numel()] = service
                state[3+server_service.numel()+servers.numel()+service.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()] = succ_services

            return state
        elif self.alg == 'prev_serversDQN' or self.alg == 'prev_serversA2C'  or self.alg == 'prev_serversLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+servers.numel()] = servers
                state[3+servers.numel():3+servers.numel()+service.numel()] = service
                state[3+servers.numel()+service.numel():3+servers.numel()+service.numel()+succ_services.numel()] = succ_services

            return state
        elif self.alg =='justserviceDQN' or self.alg =='justserviceA2C' or self.alg =='justserviceLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0

                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+service.numel()] = service
                state[3+service.numel():3+service.numel()+succ_services.numel()] = succ_services
            return state
        elif self.alg =='GAT':
            state = torch.zeros(self.state_size)
            if task is None:
                return state
            else:
                data = from_networkx(DAG)
                x = [[0, 0, 0, 0]]
                for t in tasks.values():
                    x.append([t.cpu_cycle * 1e-6 / self.max_cpu_cycles, t.input_data_length / self.max_data_length, t.outputlength / self.max_data_length, t.service / self.numberofservices])
                data.x = torch.tensor(x, dtype=torch.float)
                state_comp = gat.encode(data)
                keys_list = list(tasks.keys())
                state = state_comp[keys_list.index(task.task_number)]
                return state
        elif self.alg == 'GCN_DQN':
            if task is None:
                return torch.zeros((self.state_size,))
            else:
                return Embeddings[int(task.task_number)-1]
        elif self.alg == 'GCN':
            state = torch.zeros(self.state_size)
            if task is None:
                return state
            else:
                # Convert DAG to graph representation
                data = from_networkx(DAG)
                
                # Prepare node features
                x = []
                for t in tasks.values():
                    x.append([t.cpu_cycle * 1e-6 / self.max_cpu_cycles, t.input_data_length / self.max_data_length, t.outputlength / self.max_data_length, t.service / self.numberofservices])
                data.x = torch.tensor(x, dtype=torch.float)
                
                # Prepare edge index
                edge_index = data.edge_index
                self.agent.edge_index = edge_index
                
                # Use GCN model to get the state representation
                self.agent.TrainNet.eval()
                with torch.no_grad():
                    output = self.agent.TrainNet.model(data.x, edge_index)
                
                # Find the node index that corresponds to the task_number
                task_number = task.task_number
                node_index = None
                for i, (node, attr) in enumerate(DAG.nodes(data=True)):
                    if node == task_number:
                        node_index = i
                        break
                
                if node_index is None:
                    raise ValueError(f"Task number {task_number} not found in DAG nodes.")
                
                # Use the node index to get the state representation
                state = output[node_index]
                return state
  
        elif self.alg == 'nearestserver_prev_servers_plus_service_per_serverDQN' or self.alg == 'nearestserver_prev_servers_plus_service_per_serverA2C'  or self.alg == 'nearestserver_prev_servers_plus_service_per_serverLSTM':
            state = torch.zeros(self.state_size)
            if task is not None:
                server_service = torch.tensor(servers_service_info, dtype=torch.float)
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                nearestserver = torch.zeros(self.numberofservers, dtype=torch.float)
                
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                nearestserver[nearest_server_id] = 1.0
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+server_service.numel()] = server_service.view(-1)
                state[3+server_service.numel():3+server_service.numel()+servers.numel()] = servers
                state[3+server_service.numel()+servers.numel():3+server_service.numel()+servers.numel()+service.numel()] = service
                state[3+server_service.numel()+servers.numel()+service.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()] = succ_services
                state[3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel():] = nearestserver

            return state
        elif self.alg == 'direction_nearestserver_prev_servers_plus_service_per_serverDQN':
            state = torch.zeros(self.state_size)
            if task is not None:
                server_service = torch.tensor(servers_service_info, dtype=torch.float)
                servers = torch.zeros(self.numberofservers, dtype=torch.float)
                succ_services = torch.zeros(self.numberofservices, dtype=torch.float)
                service = torch.zeros(self.numberofservices, dtype=torch.float)
                nearestserver = torch.zeros(self.numberofservers, dtype=torch.float)
                if task.service > 0:
                    service[task.service-1] = 1.0
                for t in task.successors:
                    if tasks[t].service > 0:
                        succ_services[tasks[t].service-1] = 1.0

                max_prev_tasks = 0.0
                for t in task.predecessors:
                    if done_tasks:
                        servers[done_tasks[t].assigned_server] = 1.0
                        max_prev_tasks = max(max_prev_tasks, done_tasks[t].result.finish_time)
                    else:
                        max_prev_tasks = 0.0
                
                if deadline > 0:
                    state[0] = max_prev_tasks / deadline
                else:
                    state[0] = 0.0
                
                nearestserver[nearest_server_id] = 1.0
                if user_direction == None:
                    direction = 0.0
                else:
                    if user_direction == "+horizontal":
                        direction = 1.0
                    elif user_direction == "-horizontal":
                        direction = -1.0
                    elif user_direction == "+vertical":
                        direction = 1.0
                    else:
                        direction = -1.0
                state[1] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[2] = task.input_data_length / self.max_data_length
                state[3:3+server_service.numel()] = server_service.view(-1)
                state[3+server_service.numel():3+server_service.numel()+servers.numel()] = servers
                state[3+server_service.numel()+servers.numel():3+server_service.numel()+servers.numel()+service.numel()] = service
                state[3+server_service.numel()+servers.numel()+service.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()] = succ_services
                state[3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel():3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()+nearestserver.numel()] = nearestserver
                state[3+server_service.numel()+servers.numel()+service.numel()+succ_services.numel()+nearestserver.numel()] = direction
            return state
        else:
            #print("The algorithm is not IMPELEMENTED")
            state = torch.zeros(self.state_size)
            if task is not None:
                state[0] = task.cpu_cycle / self.max_cpu_cycles / (10**6)
                state[1] = task.input_data_length / self.max_data_length
            return state
