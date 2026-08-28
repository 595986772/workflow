import torch
from agent import Agent
from task import Task
import numpy as np
import random
import networkx as nx
import time
from input import GCN_paramaters as gcn_paramaters
from information_protocol import (
    CAUSAL_CACHE_INFORMATION_REGIME,
)
from critical_path_rl import (
    NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS,
)

try:
    import pulp
except ImportError:
    pulp = None

from critical_path_cache import (
    alternative_service_fetch_delays,
    coordinated_cache_decision,
    critical_service_values,
    dependency_locality_bonuses,
    exponential_moving_average,
    history_only_server_quality,
    hysteretic_cache_decision,
    workload_normalized_server_telemetry,
)

try:
    import tensorflow as tf
except Exception:
    tf = None


DAOC_PAPER_CACHE_POLICY = 'paper_popularity_cost_ema'


class Broker:
    def __init__(self,env,max_cpu_cycles,max_data_length,numberofservices,numberofservers,learning_arguments,algorithm,filename_png):
        self.simulator=None
        self.env=env
        learning_arguments_broker = learning_arguments.copy()
        learning_arguments_broker['learning_rate'] = 0.01
        self.agent = Agent(algorithm = algorithm,learning_arguments = learning_arguments_broker,numberofservers  = numberofservers,numberofservices = numberofservices,max_cpu_cycles = max_cpu_cycles,max_data_length=max_data_length,filename_png=filename_png)
        self.numberofservices = numberofservices
        self.numberofservers = numberofservers
        self.cache_replacements = 0
        self.cache_update_events = 0
        self.cache_decision_calls = 0
        self.cache_decision_wall_time_sec = 0.0
        self.last_cache_decision_wall_time_sec = 0.0
        self.cache_migration_events = 0
        self.cache_migration_time_sec = 0.0
        self.cache_migration_critical_time_sec = 0.0
        self.last_cache_migration_events = 0
        self.last_cache_migration_time_sec = 0.0
        self.last_cache_migration_critical_time_sec = 0.0
        self.cache_round = 0
        self.cache_observations = 0
        self.cache_window_observations = 0
        self.cache_window_values = {
            s: {
                q: 0.0
                for q in range(1, self.numberofservices + 1)
            }
            for s in range(self.numberofservers)
        }
        self.cache_information_regime = (
            CAUSAL_CACHE_INFORMATION_REGIME
        )
        self.cache_history_windows = 0
        self.cache_history_window_requests = 0
        self.cache_history_window_cpu_cycles = 0.0
        self.cache_history_window_server_latency = {
            s: 0.0 for s in range(self.numberofservers)
        }
        self.cache_history_window_server_samples = {
            s: 0 for s in range(self.numberofservers)
        }
        self.cache_history_window_server_compute_per_mcycle = {
            s: 0.0 for s in range(self.numberofservers)
        }
        self.cache_history_window_server_compute_samples = {
            s: 0 for s in range(self.numberofservers)
        }
        self.cache_history_window_server_waiting_latency = {
            s: 0.0 for s in range(self.numberofservers)
        }
        self.cache_expected_requests_ema = None
        self.cache_mean_cpu_cycles_ema = None
        self.cache_global_execution_latency_ema = None
        self.cache_global_compute_per_mcycle_ema = None
        self.cache_global_waiting_latency_ema = None
        self.cache_server_execution_latency_ema = {
            s: None for s in range(self.numberofservers)
        }
        self.cache_server_compute_per_mcycle_ema = {
            s: None for s in range(self.numberofservers)
        }
        self.cache_server_waiting_latency_ema = {
            s: None for s in range(self.numberofservers)
        }
        self.cache_server_sample_counts = {
            s: 0 for s in range(self.numberofservers)
        }
        self.cache_server_last_observed_window = {
            s: None for s in range(self.numberofservers)
        }
        self.last_server_telemetry_context = None
        self.last_cache_decision_context = None
        self.last_cache_change_round = {
            s: 0 for s in range(self.numberofservers)
        }
        #self.alltasks=[]
        self.H = {s: {q: 0 for q in range(1,self.numberofservices+1)} for s in range(self.numberofservers)}

        self.servergcn = None
        self.task_gcn = None
        if algorithm in ('GCN', 'GCN_DQN'):
            from gcn import ServerGCN, TaskGCN
            self.servergcn = ServerGCN(
                numberofservices=self.numberofservices,
                hidden_dim=[64, 64],
                output_dim=10,
                dropout=0.5,
            )
            self.task_gcn = TaskGCN(
                self.numberofservices,
                hidden_dim=gcn_paramaters['layers'],
                output_dim=gcn_paramaters['output_dim'],
                dropout=gcn_paramaters['dropout'],
            )
    def make_training_data(self,servers_info):
        inputs=[]
        outputs=[]
        selected_indices = random.sample(range(len(self.all_tasks)), min(1000, len(self.all_tasks)))
        #for index in selected_indices:
        for index in range(len(self.all_tasks)):
            tasks = self.all_tasks[index]
            # TODO GCN
            # DAG_task = self.DAGs[index]
            # node_features = self.task_gcn.generate_task_features(tasks, self.numberofservices)
            # edge_index = self.edgeindexes[index]
            # task_embeddings = self.task_gcn(node_features, edge_index).detach().numpy()
            task_embeddings = None
            for task in tasks.values():
                inputs.append(self.agent.state(task, tasks=tasks, done_tasks=None, deadline=-1, servers_service_info=self.simulator.server_service_info, gat=self.simulator.gat, DAG=task.DAG, Embeddings=task_embeddings))
                output = np.zeros(self.agent.numberofservers)
                for s in servers_info.values():
                    if task.service in s.services:
                        output[s.id] = 1.0
                outputs.append(output)
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        return inputs,outputs
    def make_training_tasks(self,DAGs,DAGsizeMAX,max_cpu_cycles,max_data_length):
        all_tasks=[]
        used_DAGs=[]
        batch_size = 128
        edge_indexes=[]
        for batch_index in range(batch_size):
            for service in range(1,self.numberofservices+1):
                while (True):
                    random_graph_key = random.choice(list(DAGs.keys()))
                    DAG = nx.DiGraph(DAGs[random_graph_key])
                    if(len(DAG.nodes.items())<=DAGsizeMAX):
                        break
                tasks={}
                edge_index=[]
                for edge in DAG.edges:                    
                    if (DAG[edge[0]][edge[1]]['datalength'] >1):
                        print(DAG[edge[0]][edge[1]]['datalength'] )
                    DAG[edge[0]][edge[1]]['datalength'] = int(DAG[edge[0]][edge[1]]['datalength'] *max_data_length)
                    edge_index.append([float(edge[0]), float(edge[1])])
                for i,node in DAG.nodes.items():
                    tasks[i] = Task(user_id = 0,tasknumber=i,cpu_cycles = int(max_cpu_cycles*node['cpucycle']),service=service,DAG=DAG)
                
                all_tasks.append(tasks)
                used_DAGs.append(DAG)
                edge_indexes.append(torch.tensor(edge_index, dtype=torch.long).t().contiguous())
        return all_tasks,used_DAGs,edge_indexes
    def learn0(self,inputs,outputs):
        if tf is None:
            raise RuntimeError("This legacy training path requires a working TensorFlow installation.")
        loss=1000
        while(True):
            loss_t= loss
            with tf.GradientTape() as tape:
                actual_values = self.agent.agent.TrainNet.predict(inputs)
                loss = tf.math.reduce_mean(tf.square(actual_values - outputs))
                variables = self.agent.agent.TrainNet.model.trainable_variables
                gradients = tape.gradient(loss, variables)
            self.agent.agent.TrainNet.optimizer.apply_gradients(zip(gradients, variables))
            #print(loss)
            #print('Diff:')
            #print((loss_t-loss))
            if (np.abs(loss_t-loss)) <1e-2:
                break
    def learn(self, inputs, outputs):
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        # Ensure inputs and outputs are PyTorch tensors and move to device
        #inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        #inputs = torch.stack(list(inputs)).to(dtype=torch.float32, device=device)
        inputs = torch.stack([torch.tensor(i, dtype=torch.float32, device=device) for i in inputs])



        outputs = torch.tensor(outputs, dtype=torch.float32).to(device)

        # Set the model to training mode and move to device
        self.agent.agent.TrainNet.model.train()
        self.agent.agent.TrainNet.model.to(device)

        loss = 1000.0  # Initialize loss
        loss_fn = torch.nn.MSELoss()

        while True:
            loss_t = loss

            # Zero the gradients
            self.agent.agent.TrainNet.optimizer.zero_grad()

            # Forward pass
            actual_values = self.agent.agent.TrainNet.model(inputs)

            # Compute loss
            loss = loss_fn(actual_values, outputs)

            # Backward pass
            loss.backward()

            # Update the weights
            self.agent.agent.TrainNet.optimizer.step()

            # Check for convergence
            #if abs(loss_t - loss.item()) < 1e-5:
            #    break
             # Check for convergence
            if abs(loss.item()) < 1e-2:
                break

    def weights0(self):
        return self.agent.agent.TrainNet.model.get_weights()
    def weights(self):
        return self.agent.agent.TrainNet.model.state_dict()
    def broadcast_weights_user(self,brokerweights):
        beta = self.simulator.beta
        for m in range(self.simulator.M):
            self.simulator.users[m].weights_adjustement(brokerweights,beta)    

    def broadcast_weights_server(self,broker_state_dict):
        coeff = self.simulator.beta
        for server in self.simulator.servers.values():
            local_state_dict = server.agent.agent.TrainNet.model.state_dict()

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
            server.agent.agent.TrainNet.model.load_state_dict(averaged_state_dict)
            # Update the target model if necessary
            server.agent.agent.update_target_model()  
    def attach(self,simulator):
        self.simulator=simulator
        self.agent.task_dependency_features_enabled = (
            simulator.task_dependency_features_enabled
        )
        self.all_tasks,self.DAGs,self.edgeindexes= self.make_training_tasks(DAGs = self.simulator.loaded_graphs,DAGsizeMAX = self.simulator.I,max_cpu_cycles=self.simulator.max_cpu_cycles,max_data_length=self.simulator.max_data_length)

    def causal_server_quality(self):
        """Return server quality derived only from completed-task EMAs."""
        if (
            self.simulator.alg
            in NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS
        ):
            return self._normalized_server_telemetry()
        compute_intensity = 0.0
        if self.cache_mean_cpu_cycles_ema is not None:
            maximum_cpu_cycles = (
                self.simulator.max_cpu_cycles * 1e6
            )
            compute_intensity = min(
                max(
                    self.cache_mean_cpu_cycles_ema
                    / maximum_cpu_cycles,
                    0.0,
                ),
                1.0,
            )
        return history_only_server_quality(
            server_execution_latency_ema=(
                self.cache_server_execution_latency_ema
            ),
            global_execution_latency_ema=(
                self.cache_global_execution_latency_ema
            ),
            compute_weight=self.simulator.cache_compute_weight,
            compute_intensity=compute_intensity,
        )

    def _normalized_server_telemetry(self):
        telemetry = workload_normalized_server_telemetry(
            server_compute_per_mcycle_ema=(
                self.cache_server_compute_per_mcycle_ema
            ),
            server_waiting_latency_ema=(
                self.cache_server_waiting_latency_ema
            ),
            global_compute_per_mcycle_ema=(
                self.cache_global_compute_per_mcycle_ema
            ),
            global_waiting_latency_ema=(
                self.cache_global_waiting_latency_ema
            ),
            server_sample_counts=self.cache_server_sample_counts,
            server_last_observed_window=(
                self.cache_server_last_observed_window
            ),
            current_window=self.cache_history_windows,
            min_samples=self.simulator.telemetry_min_samples,
            freshness_half_life=(
                self.simulator.telemetry_freshness_half_life
            ),
        )
        self.last_server_telemetry_context = {
            "mode": "workload_normalized_v1",
            "history_windows": self.cache_history_windows,
            "global_compute_per_mcycle_ema": (
                self.cache_global_compute_per_mcycle_ema
            ),
            "global_waiting_latency_ema": (
                self.cache_global_waiting_latency_ema
            ),
            "telemetry": dict(telemetry),
        }
        return telemetry

    def run(self):
        #start_time = time.time()  # Start time for the iteration
        while(self.simulator.notcomplete):
            self.simulator.notcomplete = False
            for m in range(self.simulator.M):
                self.simulator.notcomplete = self.simulator.notcomplete or (not self.simulator.users[m].complete)
            yield self.env.timeout(1)
        #end_time = time.time()  # End time for the iteration
        #elapsed_time = end_time - start_time
        #print(f"broker Iteration took {elapsed_time:.10f} seconds")
        if (
            self.simulator.update_caching
            and not self.simulator.periodic_cache_updates
        ):
            self.advance_cache_window()
        #self.federated_learning()

    def advance_cache_window(self):
        """Apply one causal online cache-update epoch."""
        if not self.simulator.update_caching:
            return False
        self.cache_round += 1
        should_update = (
            self.simulator.cache_policy
            in {'popularity_ema', DAOC_PAPER_CACHE_POLICY}
            or self.cache_round
            % self.simulator.cache_update_interval
            == 0
        )
        if not should_update:
            return False
        if (
            self.simulator.cache_policy
            in {
                'popularity_coordinated',
                'critical_path_hysteresis',
                'critical_path_coordinated',
                'critical_path_joint',
            }
        ):
            self._finalize_critical_cache_window()
        self.broadcast_caching_decisions()
        return True

        
    def caching_decisions_update(
        self,
        server,
        service,
        cost,
        task=None,
    ):
        if (
            self.simulator.cache_policy
            in {
                'popularity_coordinated',
                'critical_path_hysteresis',
                'critical_path_coordinated',
                'critical_path_joint',
            }
        ):
            if task is None:
                raise ValueError(
                    "critical-path caching requires the completed task"
                )
            if (
                self.simulator.cache_policy
                in {
                    'popularity_coordinated',
                    'critical_path_coordinated',
                    'critical_path_joint',
                }
            ):
                if (
                    self.simulator.cache_policy
                    == 'popularity_coordinated'
                    or not getattr(
                        self.simulator,
                        'cache_dependency_awareness_enabled',
                        True,
                    )
                ):
                    self._coordinated_popularity_update(task)
                else:
                    self._coordinated_cache_update(task)
            else:
                self._critical_path_cache_update(task)
            self._observe_causal_cache_history(task)
            self.cache_observations += 1
            return

        # The released DAOC code ranks pure popularity.  The paper protocol
        # ranks popularity multiplied by the local service-loading time.
        alpha = 0.1
        paper_loading_time = None
        if self.simulator.cache_policy == DAOC_PAPER_CACHE_POLICY:
            alpha = self.simulator.cache_score_alpha
            paper_loading_time = (
                self.simulator.service_data_length[service]
                / self.simulator.servers[server].rate_to_cloud
            )
        #self.H[server][service] += alpha * (self.simulator.service_data_length[service] *cost - self.H[server][service])
        #self.H[server][service] +=  (self.simulator.service_data_length[service] *cost)
        #self.H[server][service] += 1
        # Update Averaged Values
        for q in range(1,self.numberofservices+1):
            # Assuming I(q=eta_mi) is an indicator function that returns 1 if q equals eta_mi, else 0
            #self.H[server][q] += alpha * (int(q == service)*self.simulator.service_data_length[q] - self.H[server][q])
            #self.H[server][q] += alpha * (int(q == service)*self.simulator.service_data_length[q] *cost - self.H[server][q])
            observation = float(q == service)
            if paper_loading_time is not None:
                observation *= paper_loading_time
            self.H[server][q] += alpha * (
                observation - self.H[server][q]
            )
            #print(f"Updated H[{server}][{q}] = {self.H[server][q]}")
        self.cache_observations += 1

    def _critical_path_cache_update(self, task):
        user = self.simulator.users[task.user_id]
        criticality = (
            user.nominal_upward_ranks[task.task_number]
            / user.maximum_nominal_rank
        )
        locality_bonuses = dependency_locality_bonuses(
            task=task,
            done_tasks=user.done_tasks,
            number_of_servers=self.numberofservers,
            between_server_costs=(
                self.simulator.between_server_costs
            ),
        )
        fetch_delays = alternative_service_fetch_delays(
            service=task.service,
            service_size=(
                self.simulator.service_data_length[task.service]
            ),
            servers=self.simulator.servers,
            between_server_costs=(
                self.simulator.between_server_costs
            ),
        )
        values = critical_service_values(
            criticality=criticality,
            fetch_delays=fetch_delays,
            locality_bonuses=locality_bonuses,
            locality_weight=(
                self.simulator.cache_locality_weight
            ),
        )
        for server_id in range(self.numberofservers):
            self.cache_window_values[
                server_id
            ][task.service] += values[server_id]
        self.cache_window_observations += 1

    def _coordinated_cache_update(self, task):
        user = self.simulator.users[task.user_id]
        criticality = (
            user.nominal_upward_ranks[task.task_number]
            / user.maximum_nominal_rank
        )
        self.cache_window_values[
            task.assigned_server
        ][task.service] += criticality

        for predecessor_id in task.predecessors:
            predecessor = user.done_tasks[predecessor_id]
            output_fraction = (
                predecessor.outputs_length.get(
                    task.task_number,
                    0.0,
                )
                / self.simulator.max_data_length
            )
            self.cache_window_values[
                predecessor.assigned_server
            ][task.service] += (
                self.simulator.cache_locality_weight
                * criticality
                * output_fraction
            )
        self.cache_window_observations += 1

    def _coordinated_popularity_update(self, task):
        """Record one completed request without critical-path weighting."""
        self.cache_window_values[
            task.assigned_server
        ][task.service] += 1.0
        self.cache_window_observations += 1

    def _finalize_critical_cache_window(self):
        if self.cache_window_observations > 0:
            alpha = self.simulator.cache_score_alpha
            for server_id in range(self.numberofservers):
                for service_id in range(
                    1,
                    self.numberofservices + 1,
                ):
                    window_mean = (
                        self.cache_window_values[
                            server_id
                        ][service_id]
                        / self.cache_window_observations
                    )
                    self.H[server_id][service_id] += alpha * (
                        window_mean - self.H[server_id][service_id]
                    )
                    self.cache_window_values[
                        server_id
                    ][service_id] = 0.0
            self.cache_window_observations = 0
        self._finalize_causal_cache_history()

    def _observe_causal_cache_history(self, task):
        server_id = int(task.assigned_server)
        execution_latency = float(
            task.result.computing_latency
            + task.result.waiting_latency
        )
        computing_latency = float(task.result.computing_latency)
        waiting_latency = float(task.result.waiting_latency)
        cpu_cycles = float(task.cpu_cycle)
        if not np.isfinite(execution_latency) or execution_latency < 0:
            raise ValueError(
                "observed execution latency must be finite "
                "and non-negative"
            )
        if not np.isfinite(cpu_cycles) or cpu_cycles < 0:
            raise ValueError(
                "observed CPU cycles must be finite and non-negative"
            )

        self.cache_history_window_requests += 1
        self.cache_history_window_cpu_cycles += cpu_cycles
        self.cache_history_window_server_latency[
            server_id
        ] += execution_latency
        self.cache_history_window_server_samples[
            server_id
        ] += 1
        if cpu_cycles > 0.0:
            compute_per_mcycle = (
                computing_latency / (cpu_cycles / 1e6)
            )
            self.cache_history_window_server_compute_per_mcycle[
                server_id
            ] += compute_per_mcycle
            self.cache_history_window_server_compute_samples[
                server_id
            ] += 1
        self.cache_history_window_server_waiting_latency[
            server_id
        ] += waiting_latency

    def _finalize_causal_cache_history(self):
        request_count = self.cache_history_window_requests
        if request_count == 0:
            return

        alpha = self.simulator.cache_history_alpha
        self.cache_expected_requests_ema = (
            exponential_moving_average(
                self.cache_expected_requests_ema,
                request_count,
                alpha,
            )
        )
        self.cache_mean_cpu_cycles_ema = (
            exponential_moving_average(
                self.cache_mean_cpu_cycles_ema,
                self.cache_history_window_cpu_cycles
                / request_count,
                alpha,
            )
        )
        global_latency_sum = sum(
            self.cache_history_window_server_latency.values()
        )
        self.cache_global_execution_latency_ema = (
            exponential_moving_average(
                self.cache_global_execution_latency_ema,
                global_latency_sum / request_count,
                alpha,
            )
        )
        global_compute_samples = sum(
            self.cache_history_window_server_compute_samples.values()
        )
        if global_compute_samples > 0:
            global_compute_per_mcycle = (
                sum(
                    self.cache_history_window_server_compute_per_mcycle
                    .values()
                )
                / global_compute_samples
            )
            self.cache_global_compute_per_mcycle_ema = (
                exponential_moving_average(
                    self.cache_global_compute_per_mcycle_ema,
                    global_compute_per_mcycle,
                    alpha,
                )
            )
        global_waiting_latency = (
            sum(
                self.cache_history_window_server_waiting_latency
                .values()
            )
            / request_count
        )
        self.cache_global_waiting_latency_ema = (
            exponential_moving_average(
                self.cache_global_waiting_latency_ema,
                global_waiting_latency,
                alpha,
            )
        )
        for server_id in range(self.numberofservers):
            samples = (
                self.cache_history_window_server_samples[
                    server_id
                ]
            )
            if samples > 0:
                window_latency = (
                    self.cache_history_window_server_latency[
                        server_id
                    ]
                    / samples
                )
                self.cache_server_execution_latency_ema[
                    server_id
                ] = exponential_moving_average(
                    self.cache_server_execution_latency_ema[
                        server_id
                    ],
                    window_latency,
                    alpha,
                )
                compute_samples = (
                    self.cache_history_window_server_compute_samples[
                        server_id
                    ]
                )
                if compute_samples > 0:
                    window_compute_per_mcycle = (
                        self.cache_history_window_server_compute_per_mcycle[
                            server_id
                        ]
                        / compute_samples
                    )
                    self.cache_server_compute_per_mcycle_ema[
                        server_id
                    ] = exponential_moving_average(
                        self.cache_server_compute_per_mcycle_ema[
                            server_id
                        ],
                        window_compute_per_mcycle,
                        alpha,
                    )
                window_waiting_latency = (
                    self.cache_history_window_server_waiting_latency[
                        server_id
                    ]
                    / samples
                )
                self.cache_server_waiting_latency_ema[
                    server_id
                ] = exponential_moving_average(
                    self.cache_server_waiting_latency_ema[
                        server_id
                    ],
                    window_waiting_latency,
                    alpha,
                )
                self.cache_server_sample_counts[server_id] += samples
                self.cache_server_last_observed_window[
                    server_id
                ] = self.cache_history_windows + 1
            self.cache_history_window_server_latency[
                server_id
            ] = 0.0
            self.cache_history_window_server_samples[
                server_id
            ] = 0
            self.cache_history_window_server_compute_per_mcycle[
                server_id
            ] = 0.0
            self.cache_history_window_server_compute_samples[
                server_id
            ] = 0
            self.cache_history_window_server_waiting_latency[
                server_id
            ] = 0.0

        self.cache_history_windows += 1
        self.cache_history_window_requests = 0
        self.cache_history_window_cpu_cycles = 0.0

    def cache_history_state_dict(self):
        return {
            'information_regime': self.cache_information_regime,
            'history_windows': self.cache_history_windows,
            'window_requests': self.cache_history_window_requests,
            'window_cpu_cycles': (
                self.cache_history_window_cpu_cycles
            ),
            'window_server_latency': dict(
                self.cache_history_window_server_latency
            ),
            'window_server_samples': dict(
                self.cache_history_window_server_samples
            ),
            'window_server_compute_per_mcycle': dict(
                self.cache_history_window_server_compute_per_mcycle
            ),
            'window_server_compute_samples': dict(
                self.cache_history_window_server_compute_samples
            ),
            'window_server_waiting_latency': dict(
                self.cache_history_window_server_waiting_latency
            ),
            'expected_requests_ema': (
                self.cache_expected_requests_ema
            ),
            'mean_cpu_cycles_ema': (
                self.cache_mean_cpu_cycles_ema
            ),
            'global_execution_latency_ema': (
                self.cache_global_execution_latency_ema
            ),
            'global_compute_per_mcycle_ema': (
                self.cache_global_compute_per_mcycle_ema
            ),
            'global_waiting_latency_ema': (
                self.cache_global_waiting_latency_ema
            ),
            'server_execution_latency_ema': dict(
                self.cache_server_execution_latency_ema
            ),
            'server_compute_per_mcycle_ema': dict(
                self.cache_server_compute_per_mcycle_ema
            ),
            'server_waiting_latency_ema': dict(
                self.cache_server_waiting_latency_ema
            ),
            'server_sample_counts': dict(
                self.cache_server_sample_counts
            ),
            'server_last_observed_window': dict(
                self.cache_server_last_observed_window
            ),
            'last_server_telemetry_context': (
                None
                if self.last_server_telemetry_context is None
                else dict(self.last_server_telemetry_context)
            ),
            'last_decision_context': (
                None
                if self.last_cache_decision_context is None
                else dict(self.last_cache_decision_context)
            ),
        }

    def load_cache_history_state_dict(self, state):
        self.cache_information_regime = state[
            'information_regime'
        ]
        self.cache_history_windows = int(
            state['history_windows']
        )
        self.cache_history_window_requests = int(
            state['window_requests']
        )
        self.cache_history_window_cpu_cycles = float(
            state['window_cpu_cycles']
        )
        self.cache_history_window_server_latency = dict(
            state['window_server_latency']
        )
        self.cache_history_window_server_samples = dict(
            state['window_server_samples']
        )
        self.cache_history_window_server_compute_per_mcycle = dict(
            state.get(
                'window_server_compute_per_mcycle',
                {
                    server_id: 0.0
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.cache_history_window_server_compute_samples = dict(
            state.get(
                'window_server_compute_samples',
                {
                    server_id: 0
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.cache_history_window_server_waiting_latency = dict(
            state.get(
                'window_server_waiting_latency',
                {
                    server_id: 0.0
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.cache_expected_requests_ema = state[
            'expected_requests_ema'
        ]
        self.cache_mean_cpu_cycles_ema = state[
            'mean_cpu_cycles_ema'
        ]
        self.cache_global_execution_latency_ema = state[
            'global_execution_latency_ema'
        ]
        self.cache_global_compute_per_mcycle_ema = state.get(
            'global_compute_per_mcycle_ema'
        )
        self.cache_global_waiting_latency_ema = state.get(
            'global_waiting_latency_ema'
        )
        self.cache_server_execution_latency_ema = dict(
            state['server_execution_latency_ema']
        )
        self.cache_server_compute_per_mcycle_ema = dict(
            state.get(
                'server_compute_per_mcycle_ema',
                {
                    server_id: None
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.cache_server_waiting_latency_ema = dict(
            state.get(
                'server_waiting_latency_ema',
                {
                    server_id: None
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.cache_server_sample_counts = dict(
            state.get(
                'server_sample_counts',
                {
                    server_id: 0
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.cache_server_last_observed_window = dict(
            state.get(
                'server_last_observed_window',
                {
                    server_id: None
                    for server_id in range(self.numberofservers)
                },
            )
        )
        self.last_server_telemetry_context = state.get(
            'last_server_telemetry_context'
        )
        self.last_cache_decision_context = state[
            'last_decision_context'
        ]

    def cache_runtime_state_dict(self):
        return {
            'cache_replacements': self.cache_replacements,
            'cache_update_events': self.cache_update_events,
            'cache_decision_calls': self.cache_decision_calls,
            'cache_decision_wall_time_sec': (
                self.cache_decision_wall_time_sec
            ),
            'last_cache_decision_wall_time_sec': (
                self.last_cache_decision_wall_time_sec
            ),
            'cache_migration_events': getattr(
                self,
                'cache_migration_events',
                0,
            ),
            'cache_migration_time_sec': (
                getattr(self, 'cache_migration_time_sec', 0.0)
            ),
            'cache_migration_critical_time_sec': (
                getattr(
                    self,
                    'cache_migration_critical_time_sec',
                    0.0,
                )
            ),
            'last_cache_migration_events': (
                getattr(self, 'last_cache_migration_events', 0)
            ),
            'last_cache_migration_time_sec': (
                getattr(
                    self,
                    'last_cache_migration_time_sec',
                    0.0,
                )
            ),
            'last_cache_migration_critical_time_sec': (
                getattr(
                    self,
                    'last_cache_migration_critical_time_sec',
                    0.0,
                )
            ),
            'cache_round': self.cache_round,
            'cache_observations': self.cache_observations,
            'cache_window_observations': (
                self.cache_window_observations
            ),
            'cache_window_values': {
                server_id: dict(service_values)
                for server_id, service_values
                in self.cache_window_values.items()
            },
            'last_cache_change_round': dict(
                self.last_cache_change_round
            ),
        }

    def load_cache_runtime_state_dict(self, state):
        self.cache_replacements = int(
            state['cache_replacements']
        )
        self.cache_update_events = int(
            state['cache_update_events']
        )
        self.cache_decision_calls = int(
            state.get('cache_decision_calls', 0)
        )
        self.cache_decision_wall_time_sec = float(
            state.get('cache_decision_wall_time_sec', 0.0)
        )
        self.last_cache_decision_wall_time_sec = float(
            state.get('last_cache_decision_wall_time_sec', 0.0)
        )
        self.cache_migration_events = int(
            state.get('cache_migration_events', 0)
        )
        self.cache_migration_time_sec = float(
            state.get('cache_migration_time_sec', 0.0)
        )
        self.cache_migration_critical_time_sec = float(
            state.get('cache_migration_critical_time_sec', 0.0)
        )
        self.last_cache_migration_events = int(
            state.get('last_cache_migration_events', 0)
        )
        self.last_cache_migration_time_sec = float(
            state.get('last_cache_migration_time_sec', 0.0)
        )
        self.last_cache_migration_critical_time_sec = float(
            state.get(
                'last_cache_migration_critical_time_sec',
                0.0,
            )
        )
        self.cache_round = int(state['cache_round'])
        self.cache_observations = int(
            state['cache_observations']
        )
        self.cache_window_observations = int(
            state['cache_window_observations']
        )
        self.cache_window_values = {
            int(server_id): {
                int(service_id): float(value)
                for service_id, value in service_values.items()
            }
            for server_id, service_values
            in state['cache_window_values'].items()
        }
        self.last_cache_change_round = {
            int(server_id): int(cache_round)
            for server_id, cache_round
            in state['last_cache_change_round'].items()
        }

    def _historical_expected_requests(self):
        if self.cache_expected_requests_ema is None:
            return 1.0
        return max(float(self.cache_expected_requests_ema), 1.0)

    def caching_decisions(self,server):
        Ks = self.simulator.servers[server].capacity
        # Select Ks services with the highest values in H[q][s]
        sorted_dict = sorted(self.H[server].items(),key=lambda x: x[1], reverse=True)
        top_services = [item[0] for item in sorted_dict[:Ks]]
        if (
            self.simulator.cache_policy !=
            'critical_path_hysteresis'
            or self.cache_observations == 0
        ):
            return top_services

        current_services = [
            service_id
            for service_id
            in self.simulator.servers[server].services
            if service_id > 0
        ]
        minimum_residence = (
            self.simulator.cache_update_interval
            * self.simulator.cache_min_residence_updates
        )
        if (
            self.cache_round
            - self.last_cache_change_round[server]
            < minimum_residence
        ):
            return current_services

        switching_costs = {}
        for service_id in range(
            1,
            self.numberofservices + 1,
        ):
            switching_costs[service_id] = (
                alternative_service_fetch_delays(
                    service=service_id,
                    service_size=(
                        self.simulator
                        .service_data_length[service_id]
                    ),
                    servers=self.simulator.servers,
                    between_server_costs=(
                        self.simulator.between_server_costs
                    ),
                )[server]
            )
        expected_requests = self._historical_expected_requests()
        return hysteretic_cache_decision(
            scores=self.H[server],
            current_services=current_services,
            capacity=Ks,
            switching_costs=switching_costs,
            expected_requests=expected_requests,
            hysteresis_factor=(
                self.simulator.cache_hysteresis_factor
            ),
        )
    
    def broadcast_caching_decisions(self):
        decision_started = time.perf_counter()
        #C=self.caching_decision_solver(N=self.numberofservers,S=self.numberofservices,requestcounter=self.H,servicelength=self.simulator.service_data_length , between_server_costs=self.simulator.between_server_costs,servers_capacity=[self.simulator.input_dict['server capacity']  for _ in range(self.numberofservers)])
        # c={}
        # c[(1,0)] = 0.0
        # c[(1,1)] = 1.0
        # c[(1,2)] = 1.0
        # c[(2,0)] = 0.0
        # c[(2,1)] = 0.0
        # c[(2,2)] = 1.0
        # c[(3,0)] = 1.0
        # c[(3,1)] = 0.0
        # c[(3,2)] = 0.0
        # c[(4,0)] = 1.0
        # c[(4,1)] = 0.0
        # c[(4,2)] = 0.0
        # c[(5,0)] = 0.0
        # c[(5,1)] = 1.0
        # c[(5,2)] = 0.0

        # c[(1,0)] = 0.0
        # c[(1,1)] = 0.0
        # c[(1,2)] = 1.0
        # c[(2,0)] = 1.0
        # c[(2,1)] = 0.0
        # c[(2,2)] = 0.0
        # c[(3,0)] = 1.0
        # c[(3,1)] = 1.0
        # c[(3,2)] = 0.0
        # c[(4,0)] = 0.0
        # c[(4,1)] = 0.0
        # c[(4,2)] = 1.0
        # c[(5,0)] = 0.0
        # c[(5,1)] = 1.0
        # c[(5,2)] = 0.0
        caching_changed = False
        if (
            self.simulator.cache_policy
            in {
                'popularity_coordinated',
                'critical_path_coordinated',
                'critical_path_joint',
            }
            and self.cache_observations > 0
        ):
            new_decisions = (
                self.coordinated_caching_decisions()
            )
        else:
            new_decisions = {
                server_id: self.caching_decisions(server_id)
                for server_id in range(self.numberofservers)
            }
        current_services = {
            server_id: {
                service_id
                for service_id in server.services
                if service_id > 0
            }
            for server_id, server
            in self.simulator.servers.items()
        }
        service_locations = {
            service_id: [
                server_id
                for server_id, services
                in current_services.items()
                if service_id in services
            ]
            for service_id in range(
                1,
                self.numberofservices + 1,
            )
        }
        migration_times_by_server = {
            server_id: 0.0
            for server_id in range(self.numberofservers)
        }
        migration_events = 0
        for server_id, services in new_decisions.items():
            incoming = (
                set(services) - current_services[server_id]
            )
            migration_events += len(incoming)
            for service_id in incoming:
                source_costs = [
                    1.0
                    / self.simulator.servers[
                        server_id
                    ].rate_to_cloud
                ]
                source_costs.extend(
                    self.simulator.between_server_costs[
                        source_id,
                        server_id,
                    ]
                    for source_id
                    in service_locations[service_id]
                )
                migration_times_by_server[server_id] += (
                    self.simulator.service_data_length[service_id]
                    * min(source_costs)
                )
        migration_time = float(
            sum(migration_times_by_server.values())
        )
        migration_critical_time = float(
            max(migration_times_by_server.values(), default=0.0)
        )
        self.last_cache_migration_events = migration_events
        self.last_cache_migration_time_sec = migration_time
        self.last_cache_migration_critical_time_sec = (
            migration_critical_time
        )
        self.cache_migration_events += migration_events
        self.cache_migration_time_sec += migration_time
        self.cache_migration_critical_time_sec += (
            migration_critical_time
        )
        #z={}
        for s in range(self.numberofservers):
            new_decision = new_decisions[s]
            if (
                len(new_decision)
                > self.simulator.servers[s].capacity
            ):
                raise RuntimeError(
                    f"Cache decision exceeds server {s} capacity"
                )
            #new_decision = [i+1 for i in range(self.numberofservices) if C[0,s, i] == 1]
            #new_decision = [i+1 for i in range(self.numberofservices) if c[(i+1, s)] == 1]
            if self.simulator.servers[s].services != [0] + new_decision:
                old_services = set(self.simulator.servers[s].services) - {0}
                self.cache_replacements += len(old_services - set(new_decision))
                self.cache_update_events += 1
                self.last_cache_change_round[s] = self.cache_round
                self.simulator.servers[s].services = [0] + new_decision
                caching_changed = True
                for service in range(self.numberofservices):
                    self.simulator.server_service_info[s,service] = 0
                for service in self.simulator.servers[s].services:
                    if service>0:
                        self.simulator.server_service_info[s,service-1] = 1
        elapsed = time.perf_counter() - decision_started
        self.cache_decision_calls += 1
        self.cache_decision_wall_time_sec += elapsed
        self.last_cache_decision_wall_time_sec = elapsed
            #z[s]={} 
            #for q in range(1,self.numberofservices+1):
            #    if q in new_decision:
            #        z[s][q]=1
            #    else:
            #        z[s][q]=0
            #z[s][0]=1
        #TODO THERE IS NO TRANSFER LEARNING....
        #if caching_changed:
            #pass
            #self.simulator.dp.get_service_caching(z)
            #self.simulator.dp.solve()
            #inputs,outputs = self.make_training_data(servers_info = self.simulator.servers)

    def coordinated_caching_decisions(self):
        current_services = {
            server_id: [
                service_id
                for service_id in server.services
                if service_id > 0
            ]
            for server_id, server
            in self.simulator.servers.items()
        }
        minimum_residence = (
            self.simulator.cache_update_interval
            * self.simulator.cache_min_residence_updates
        )
        locked_servers = {
            server_id
            for server_id in range(self.numberofservers)
            if (
                self.cache_round
                - self.last_cache_change_round[server_id]
                < minimum_residence
            )
        }
        expected_requests = self._historical_expected_requests()
        server_quality = {
            server_id: 1.0
            for server_id in range(self.numberofservers)
        }
        compute_intensity = 0.0
        if (
            self.simulator.cache_policy == 'critical_path_joint'
            and getattr(
                self.simulator,
                'cache_server_quality_enabled',
                True,
            )
        ):
            if self.cache_mean_cpu_cycles_ema is not None:
                maximum_cpu_cycles = (
                    self.simulator.max_cpu_cycles * 1e6
                )
                compute_intensity = min(
                    max(
                        self.cache_mean_cpu_cycles_ema
                        / maximum_cpu_cycles,
                        0.0,
                    ),
                    1.0,
                )
            if (
                getattr(self.simulator, "alg", "")
                in NORMALIZED_TELEMETRY_CP_RL_ALGORITHMS
            ):
                normalized_telemetry = (
                    self._normalized_server_telemetry()
                )
                compute_weight = max(
                    float(self.simulator.cache_compute_weight),
                    0.0,
                )
                server_quality = {
                    server_id: (
                        (
                            telemetry[0] ** compute_weight
                            * telemetry[1]
                        )
                        ** (1.0 / (compute_weight + 1.0))
                    )
                    for server_id, telemetry
                    in normalized_telemetry.items()
                }
            else:
                server_quality = history_only_server_quality(
                    server_execution_latency_ema=(
                        self.cache_server_execution_latency_ema
                    ),
                    global_execution_latency_ema=(
                        self.cache_global_execution_latency_ema
                    ),
                    compute_weight=(
                        self.simulator.cache_compute_weight
                    ),
                    compute_intensity=compute_intensity,
                )
        self.last_cache_decision_context = {
            'information_regime': self.cache_information_regime,
            'history_windows': self.cache_history_windows,
            'expected_requests': expected_requests,
            'mean_cpu_cycles_ema': (
                self.cache_mean_cpu_cycles_ema
            ),
            'global_execution_latency_ema': (
                self.cache_global_execution_latency_ema
            ),
            'server_execution_latency_ema': dict(
                self.cache_server_execution_latency_ema
            ),
            'global_compute_per_mcycle_ema': (
                self.cache_global_compute_per_mcycle_ema
            ),
            'global_waiting_latency_ema': (
                self.cache_global_waiting_latency_ema
            ),
            'server_compute_per_mcycle_ema': dict(
                self.cache_server_compute_per_mcycle_ema
            ),
            'server_waiting_latency_ema': dict(
                self.cache_server_waiting_latency_ema
            ),
            'server_quality': dict(server_quality),
            'compute_intensity': compute_intensity,
        }
        return coordinated_cache_decision(
            demand=self.H,
            current_services=current_services,
            capacity={
                server_id: getattr(
                    server,
                    "capacity",
                    self.simulator.input_dict["server capacity"],
                )
                for server_id, server
                in self.simulator.servers.items()
            },
            service_sizes={
                service_id: (
                    self.simulator
                    .service_data_length[service_id]
                )
                for service_id in range(
                    1,
                    self.numberofservices + 1,
                )
            },
            cloud_costs={
                server_id: 1.0 / server.rate_to_cloud
                for server_id, server
                in self.simulator.servers.items()
            },
            between_server_costs=(
                self.simulator.between_server_costs
            ),
            expected_requests=expected_requests,
            hysteresis_factor=(
                self.simulator.cache_hysteresis_factor
            ),
            locked_servers=locked_servers,
            server_quality=server_quality,
            replica_diversity_regularization=(
                self.simulator.cache_policy == 'critical_path_joint'
            ),
            coverage_constraint=(
                self.simulator.cache_policy == 'critical_path_joint'
                and getattr(
                    self.simulator,
                    'cache_coverage_constraint',
                    False,
                )
            ),
        )
            #self.learn(inputs,outputs)
            #self.broadcast_weights_server(self.weights())

    def gcn_train(self):
        node_features = self.servergcn.generate_server_features(self.simulator.graph,self.numberofservices)
        edge_index = self.servergcn.generate_edge_index(self.simulator.graph)
        optimizer = torch.optim.Adam(self.servergcn.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.servergcn.train_gcn(node_features,edge_index, optimizer, loss_fn, epochs=100)
        self.servergcn.eval()
        return self.servergcn
    
    def federated_learning(self):
        client_updates = []
        for server in self.simulator.servers.values():  # Assume we have 3 clients
            client_state_dict = server.agent.agent.TrainNet.model.state_dict()  # Dummy state_dict
            coeff = random.random()  # Random coefficient for each client
            client_updates.append((client_state_dict, coeff))
        
        # Perform federated learning
        aggregated_state_dict = self.aggregate(client_updates)
        for server in self.simulator.servers.values():
            server.agent.agent.TrainNet.model.load_state_dict(aggregated_state_dict)

    def aggregate(self, client_updates):
        """
        Perform federated learning by aggregating client updates.

        Parameters
        ----------
        client_updates : list of tuples
            Each tuple contains (client_state_dict, coefficient) for a client.

        Returns
        -------
        None
        """
        # Initialize an empty state_dict to store the aggregated weights
        aggregated_state_dict = {}

        # Iterate over the keys of the first client's state_dict
        for key in client_updates[0][0].keys():
            # Initialize the aggregated parameter with zeros
            aggregated_state_dict[key] = torch.zeros_like(client_updates[0][0][key])

        # Aggregate the client updates
        for client_state_dict, coeff in client_updates:
            for key in client_state_dict.keys():
                aggregated_state_dict[key] += coeff * client_state_dict[key].to(aggregated_state_dict[key].device)

        # Normalize the aggregated weights by the sum of coefficients
        total_coeff = sum([coeff for _, coeff in client_updates])
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] /= total_coeff
        return aggregated_state_dict



    def solve_multi_time_caching_noB(self,T, N, S, K, servicelength, between_server_costs, cap=None):
        """
        Solve the multi-time caching problem without B-step retention:
        Minimize sum_{t,n,s} K[t,n,s] * sum_{n'} ( y[t,n,s,n'] * L[n,n'] ),
        subject to the constraints described above.
        
        Parameters:
        -----------
        T : int - number of time steps
        N : int - number of servers
        S : int - number of services
        K : 3D array-like, shape (T, N, S)
            K[t][n][s] = #requests from server n for service s at time t
        L : 2D array-like, shape (N, N)
            L[n][n'] = cost to fetch from n' to n
        cap : list of length N or None
            cap[n] = max #services that server n can hold at once, or None if unlimited
        
        Returns:
        --------
        status, C_sol, Y_sol, min_cost
        """
        if pulp is None:
            raise RuntimeError(
                "The legacy MILP cache solver requires PuLP."
            )
        prob = pulp.LpProblem("MultiTimeNoB", pulp.LpMinimize)
        # Decision vars: C[t,n,s], Y[t,n,s,n']
        C = {}
        Y = {}
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    C[(t,n,s)] = pulp.LpVariable(f"C_{t}_{n}_{s}", cat=pulp.LpBinary)
                    for nprime in range(N):
                        Y[(t,n,s,nprime)] = pulp.LpVariable(f"Y_{t}_{n}_{s}_{nprime}", cat=pulp.LpBinary)

        # Objective: sum_t,n,s [ K[t,n,s] * sum_{n'} ( Y[t,n,s,n'] * L[n][n'] ) ]
        prob += pulp.lpSum(K[n][s+1] * Y[(t,n,s,nprime)] * servicelength[s+1] * between_server_costs[n][nprime] for t in range(T) for n in range(N) for s in range(S) for nprime in range(N)), "TotalFetchCost"

        # Constraint (A): sum_{n'} Y[t,n,s,n'] = 1 - C[t,n,s]
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    prob += pulp.lpSum(Y[(t,n,s,nprime)] for nprime in range(N)) == (1 - C[(t,n,s)])

        # Constraint (B): Y[t,n,s,n'] <= C[t,n',s]
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    for nprime in range(N):
                        prob += Y[(t,n,s,nprime)] <= C[(t,nprime,s)]

        # Constraint (C): capacity (optional)
        if cap is not None:
            for t in range(T):
                for n in range(N):
                    prob += pulp.lpSum(C[(t,n,s)] for s in range(S)) <= cap[n]

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        status = pulp.LpStatus[prob.status]
        min_cost = pulp.value(prob.objective)
        
        # Extract solutions
        C_sol = { (t,n,s): int(pulp.value(C[(t,n,s)])) for t in range(T) for n in range(N) for s in range(S) }
        Y_sol = { (t,n,s,nprime): int(pulp.value(Y[(t,n,s,nprime)])) for t in range(T) for n in range(N) for s in range(S) for nprime in range(N) }

        return status, C_sol, Y_sol, min_cost

    def caching_decision_solver(self,N,S,requestcounter,servicelength, between_server_costs,servers_capacity):
        T=1
        status, C_sol, Y_sol, cost_val = self.solve_multi_time_caching_noB(T, N, S, K = requestcounter, servicelength= servicelength, between_server_costs = between_server_costs, cap=servers_capacity)
        #print("Solver Status:", status)
        #print("Minimized Cost:", cost_val)
        #for t in range(T):
        #    for n in range(N):
        #        cached = [s for s in range(S) if C_sol[(t,n,s)]==1]
        #        print(f"Time={t}, Server={n}, Cached Services:", cached)
        return C_sol
# Example usage
if __name__ == "__main__":
    # T=2, N=2, S=2
    T, N, S = 2, 2, 2
    # K[t][n][s]
    K = [
        [[10, 5],  [3, 8]],  # t=0
        [[2,  4],  [6, 1]]   # t=1
    ]
    # L[n][n']
    L = [
        [0, 2],
        [2, 0]
    ]
    # capacity?
    cap = [1, 1]  # each server can cache 1 service at a time

 
