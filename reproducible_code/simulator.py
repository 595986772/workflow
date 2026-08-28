# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
#from ortools.linear_solver import pywraplp
import numpy as np
import random

import torch
from user import User
from server import Server
from utils import plot_users_servers,generate_streets_with_users
from input import INPUT_DICT
from operator import attrgetter
import json
import hashlib
import networkx as nx
from solver import Joint_Optimizer,DynamicProgramming
from agent import Agent
from networkx.drawing.nx_pydot import graphviz_layout
from broker import Broker
import simpy 
import time
from pathlib import Path
from strict_environment import apply_strict_dag_stress
from capacity_protocol import resolve_server_capacities


DEFAULT_DAG_DATASET_PATH = (
    Path(__file__).resolve().parent / "dag_uniform.json"
)


def resolve_dag_dataset_path(value):
    """Resolve an explicit DAG dataset relative to the repository."""
    if value in (None, ""):
        return DEFAULT_DAG_DATASET_PATH
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def load_dag_dataset(path, number_of_tasks, number_of_services):
    """Load and validate a DAOC-compatible DAG dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DAG dataset does not exist: {path}")
    raw_bytes = path.read_bytes()
    dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        loaded_data = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid DAG dataset JSON: {path}") from error
    if not isinstance(loaded_data, dict) or not loaded_data:
        raise ValueError("DAG dataset must be a non-empty JSON object")

    loaded_graphs = {}
    eligible_keys = []
    for graph_name, graph_data in loaded_data.items():
        try:
            graph = nx.node_link_graph(graph_data)
        except (KeyError, TypeError, nx.NetworkXError) as error:
            raise ValueError(
                f"Invalid node-link graph {graph_name!r}"
            ) from error
        if not graph.is_directed() or not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f"Graph {graph_name!r} must be a DAG")
        if "0" not in graph:
            raise ValueError(
                f"Graph {graph_name!r} is missing dummy source node '0'"
            )
        for node_id, attributes in graph.nodes(data=True):
            if "service" not in attributes or "cpucycle" not in attributes:
                raise ValueError(
                    f"Graph {graph_name!r} node {node_id!r} "
                    "is missing service or cpucycle"
                )
            service_value = float(attributes["service"])
            if service_value > 0:
                service_id = (
                    int(number_of_services * (service_value - 1)) + 1
                )
                if not 1 <= service_id <= number_of_services:
                    raise ValueError(
                        f"Graph {graph_name!r} node {node_id!r} maps "
                        f"to invalid service {service_id}"
                    )
        for source, target, attributes in graph.edges(data=True):
            if "datalength" not in attributes:
                raise ValueError(
                    f"Graph {graph_name!r} edge "
                    f"{source!r}->{target!r} is missing datalength"
                )
        loaded_graphs[str(graph_name)] = graph
        if len(graph.nodes) <= int(number_of_tasks):
            eligible_keys.append(str(graph_name))

    if not eligible_keys:
        raise ValueError(
            "DAG dataset has no graph compatible with the configured "
            f"task limit {number_of_tasks}"
        )
    return loaded_graphs, tuple(eligible_keys), dataset_sha256

def shannon_capacity(d):
    B = 50e9  # Bandwidth in Hz (50 GHz)
    P_trans = 1e-3  # Transmitted power in W (1 mW)
    alpha_db_per_km = 0.2  # Attenuation in dB/km
    alpha = (alpha_db_per_km / 10) * np.log(10)  # Convert dB/km to linear
    N0 = 1e-20  # Noise power spectral density in W/Hz
    
    return B * np.log2(1 + (P_trans * np.exp(-alpha * d)) / (N0 * B))

class MEC_Simulator:
    def __init__(self,outputfile,Input_dict,learning_arguments,filename_png):
        self.env = simpy.Environment()
        self.networkgraph = nx.Graph()  # Create an empty graph
        self.input_dict = Input_dict
        self.notcomplete = True
        # Read parameters from input    
        self.filename_png=filename_png
        self.file = outputfile
        self.alg =Input_dict['alg']
        self.M = Input_dict['Number of users']
        self.S = Input_dict['Number of servers']
        self.I = Input_dict['Number of tasks for each user']
        self.Q = Input_dict['Number of services']
        self.dynamic_queueing = bool(
            Input_dict.get('dynamic queueing enabled', False)
        )
        self.periodic_cache_updates = bool(
            Input_dict.get('periodic cache updates', False)
        )
        configured_arrivals = Input_dict.get(
            'application arrival times'
        )
        if configured_arrivals is None:
            configured_arrivals = [0.0] * self.M
        if len(configured_arrivals) != self.M:
            raise ValueError(
                "application arrival times must contain one value "
                "per DAG instance"
            )
        self.application_arrival_times = [
            float(value) for value in configured_arrivals
        ]
        if any(
            not np.isfinite(value) or value < 0.0
            for value in self.application_arrival_times
        ):
            raise ValueError(
                "application arrival times must be finite and "
                "non-negative"
            )
        #self.Ks = INPUT_DICT['Ks']
        self.xlim=Input_dict['xlim']
        self.ylim=Input_dict['ylim']
        self.max_cpu_cycles=Input_dict['max_cpu_cycles']
        self.max_data_length=Input_dict['max_data_length']
        self.SimulationTime= Input_dict['SimulationTime']
        self.power = Input_dict['Power']
        self.BW = Input_dict['Bandwidth']
        self.min_cpu_freq = Input_dict['min_cpu_frquency']
        self.max_cpu_freq = Input_dict['max_cpu_frquency']
        self.min_service_data_length = 1e6*Input_dict['min_service_data_length']
        self.max_service_data_length = 1e6*Input_dict['max_service_data_length']
        self.minload = Input_dict['min_load_on_server']
        self.maxload = Input_dict['max_load_on_server']

        self.min_rate_between_servers = Input_dict['min_rate_between_servers']
        self.max_rate_between_servers = Input_dict['max_rate_between_servers']
        self.dag_depth_increment = int(
            Input_dict.get('dag depth increment', 0)
        )
        self.dependency_data_scale = float(
            Input_dict.get('dependency data scale', 1.0)
        )
        self.server_capacity = int(
            Input_dict.get('server capacity', 2)
        )
        self.server_capacity_multiset = Input_dict.get(
            'server capacity multiset'
        )
        self.server_capacities = resolve_server_capacities(
            input_config=Input_dict,
            number_of_servers=self.S,
            number_of_services=self.Q,
        )
        self.heterogeneous_capacity = (
            self.server_capacity_multiset is not None
        )
        self.total_server_capacity = sum(
            self.server_capacities.values()
        )
        self.baseline_server_capacity = int(
            Input_dict.get('baseline server capacity', 2)
        )
        if self.dag_depth_increment < 0:
            raise ValueError(
                "dag depth increment must be non-negative"
            )
        if self.dependency_data_scale < 1.0:
            raise ValueError(
                "dependency data scale must be at least one"
            )
        if not 0 <= self.server_capacity <= self.Q:
            raise ValueError(
                "server capacity must be between zero and "
                "the number of services"
            )
        maximum_active_capacity = max(
            self.server_capacities.values(),
            default=0,
        )
        if not maximum_active_capacity <= self.baseline_server_capacity <= self.Q:
            raise ValueError(
                "baseline server capacity must be at least the "
                "active capacity and no greater than the services"
            )
        self.filling_steps = Input_dict['filling steps']
        self.steps_b_updates = Input_dict['steps to updates']
        self.deadline = Input_dict['deadline']
        self.updatedeadline = Input_dict['update deadline']
        self.reward_scale = float(
            Input_dict.get('reward scale', self.deadline)
        )
        if self.reward_scale <= 0:
            raise ValueError("reward scale must be positive")

        self.maxratetocloud=Input_dict['Max rate to cloud']
        self.minratetocloud=Input_dict['Min rate to cloud']
        self.beta = Input_dict['beta']
        self.reward_mode = Input_dict.get(
            'reward mode',
            'terminal_binary',
        )
        if self.reward_mode not in {
            'terminal_binary',
            'critical_path_potential',
            'terminal_plus_potential',
            'causal_critical_path',
            'causal_makespan_increment',
        }:
            raise ValueError(
                f"Unknown reward mode: {self.reward_mode}"
            )
        self.potential_reward_weight = float(
            Input_dict.get('potential reward weight', 0.5)
        )
        if self.potential_reward_weight < 0:
            raise ValueError(
                "potential reward weight must be non-negative"
            )
        self.hcpr_temperature = float(
            Input_dict.get('hcpr temperature', 0.05)
        )
        if self.hcpr_temperature <= 0:
            raise ValueError("hcpr temperature must be positive")
        self.bcr_top_fraction = float(
            Input_dict.get('bcr top fraction', 0.25)
        )
        if not 0.0 < self.bcr_top_fraction <= 1.0:
            raise ValueError("bcr top fraction must be in (0, 1]")
        self.training = True
        self.update_caching = Input_dict.get('caching decision enabled', True)
        self.cache_policy = Input_dict.get(
            'cache policy',
            'popularity_ema',
        )
        if self.cache_policy not in {
            'popularity_ema',
            'paper_popularity_cost_ema',
            'popularity_coordinated',
            'critical_path_hysteresis',
            'critical_path_coordinated',
            'critical_path_joint',
        }:
            raise ValueError(
                f"Unknown cache policy: {self.cache_policy}"
            )
        self.cache_score_alpha = float(
            Input_dict.get('cache score alpha', 0.1)
        )
        self.cache_history_alpha = float(
            Input_dict.get(
                'cache history alpha',
                self.cache_score_alpha,
            )
        )
        self.cache_locality_weight = float(
            Input_dict.get('cache locality weight', 1.0)
        )
        self.cache_update_interval = int(
            Input_dict.get('cache update interval', 5)
        )
        self.cache_hysteresis_factor = float(
            Input_dict.get('cache hysteresis factor', 1.0)
        )
        self.cache_min_residence_updates = int(
            Input_dict.get('cache min residence updates', 2)
        )
        self.cache_compute_weight = float(
            Input_dict.get('cache compute weight', 1.0)
        )
        self.cache_server_quality_enabled = bool(
            Input_dict.get('cache server quality enabled', True)
        )
        self.cache_coverage_constraint = bool(
            Input_dict.get('cache coverage constraint', False)
        )
        self.task_dependency_features_enabled = bool(
            Input_dict.get(
                'task dependency features enabled',
                True,
            )
        )
        self.cache_dependency_awareness_enabled = bool(
            Input_dict.get(
                'cache dependency awareness enabled',
                True,
            )
        )
        self.telemetry_min_samples = int(
            Input_dict.get('telemetry min samples', 5)
        )
        self.telemetry_freshness_half_life = float(
            Input_dict.get('telemetry freshness half life', 10.0)
        )
        if not 0 < self.cache_score_alpha <= 1:
            raise ValueError(
                "cache score alpha must be in (0, 1]"
            )
        if not 0 < self.cache_history_alpha <= 1:
            raise ValueError(
                "cache history alpha must be in (0, 1]"
            )
        if self.cache_locality_weight < 0:
            raise ValueError(
                "cache locality weight must be non-negative"
            )
        if self.cache_update_interval < 1:
            raise ValueError(
                "cache update interval must be positive"
            )
        if self.cache_hysteresis_factor < 0:
            raise ValueError(
                "cache hysteresis factor must be non-negative"
            )
        if self.cache_min_residence_updates < 0:
            raise ValueError(
                "cache min residence updates must be non-negative"
            )
        if self.cache_compute_weight < 0:
            raise ValueError(
                "cache compute weight must be non-negative"
            )
        if self.telemetry_min_samples < 1:
            raise ValueError(
                "telemetry min samples must be positive"
            )
        if self.telemetry_freshness_half_life <= 0:
            raise ValueError(
                "telemetry freshness half life must be positive"
            )
        self.history_feedback_guidance = Input_dict.get(
            'historical feedback guidance',
            False,
        )
        self.adaptive_guidance_gate = Input_dict.get(
            'adaptive guidance gate',
            False,
        )
        self.history_feedback_alpha = float(
            Input_dict.get('history feedback alpha', 0.1)
        )
        self.history_feedback_min_samples = int(
            Input_dict.get('history feedback min samples', 3)
        )
        self.history_feedback_max_probability = float(
            Input_dict.get('history feedback max probability', 0.9)
        )
        self.history_feedback_fixed_probability = float(
            Input_dict.get('history feedback fixed probability', 0.1)
        )
        self.learning_enabled = self.alg not in {
            'random',
            'nearest_server',
            'nearest_with_service',
        }

        self.velocity = Input_dict['velocity']
        self.learning_arguments=learning_arguments
        #self.agent = Agent(algorithm = self.alg,learning_arguments = learning_arguments,numberofservers  = self.S,numberofservices = self.Q,max_cpu_cycles = self.max_cpu_cycles,max_data_length=self.max_data_length )
        self.tau_t_mis = {}
        self.tau_w_mis = {}
        self.tau_c_mis = {}
        self.time_step = 0
        self.users={}
        self.servers={}
        
        self.broker = Broker(self.env,max_cpu_cycles=self.max_cpu_cycles,max_data_length=self.max_data_length, numberofservices=self.Q, numberofservers=self.S,learning_arguments=learning_arguments,algorithm=self.alg,filename_png=self.filename_png)
        
        Numberofstreets_x = 5
        Numberofstreets_y = 5
        self.street_width = 24
        self.min_spacing = (self.ylim-Numberofstreets_x*24)/(Numberofstreets_x-1)
        self.max_spacing = self.min_spacing
        street_positions_y, street_positions_x, user_positions, user_directions = generate_streets_with_users(Numberofstreets_x,Numberofstreets_y, self.M, self.street_width, self.xlim, self.ylim, self.min_spacing, self.max_spacing)
        self.street_positions_y = street_positions_y
        self.street_positions_x = street_positions_x
        self.user_positions = user_positions
        self.user_directions = user_directions
        self.vmin=self.velocity/10
        self.vmax = self.velocity 
        configured_dag_path = Input_dict.get('dag dataset path')
        self.dag_dataset_path = resolve_dag_dataset_path(
            configured_dag_path
        )
        self.dag_dataset_is_default = (
            self.dag_dataset_path == DEFAULT_DAG_DATASET_PATH
        )
        (
            self.loaded_graphs,
            self.eligible_graph_keys,
            self.dag_dataset_sha256,
        ) = load_dag_dataset(
            self.dag_dataset_path,
            number_of_tasks=self.I,
            number_of_services=self.Q,
        )
        expected_dag_sha256 = Input_dict.get('dag dataset sha256')
        if (
            expected_dag_sha256 is not None
            and self.dag_dataset_sha256
            != str(expected_dag_sha256).lower()
        ):
            raise ValueError(
                "DAG dataset SHA-256 mismatch: expected "
                f"{expected_dag_sha256}, got {self.dag_dataset_sha256}"
            )
        self.dag_dataset_graph_count = len(self.loaded_graphs)
        self.dag_dataset_eligible_graph_count = len(
            self.eligible_graph_keys
        )
        self.application_graph_family = Input_dict.get(
            'application graph family'
        )
        if self.application_graph_family is None:
            self.graph_selection_keys = self.eligible_graph_keys
        else:
            self.application_graph_family = str(
                self.application_graph_family
            )
            self.graph_selection_keys = tuple(
                graph_key
                for graph_key in self.eligible_graph_keys
                if str(
                    self.loaded_graphs[graph_key].graph.get(
                        'source_family',
                        '',
                    )
                )
                == self.application_graph_family
            )
            if not self.graph_selection_keys:
                available_families = sorted(
                    {
                        str(graph.graph.get('source_family', ''))
                        for graph in self.loaded_graphs.values()
                        if graph.graph.get('source_family')
                    }
                )
                raise ValueError(
                    "DAG dataset has no eligible graph for family "
                    f"{self.application_graph_family!r}; available "
                    f"families: {available_families}"
                )
        self.user_graph_keys = {}
        self.user_graph_stress = {}
        for m in range(self.M):
            random_graph_key = random.choice(
                list(self.graph_selection_keys)
            )
            random_graph = self.loaded_graphs[random_graph_key]
            self.user_graph_keys[m] = random_graph_key
            random_graph, stress_metadata = apply_strict_dag_stress(
                random_graph,
                depth_increment=self.dag_depth_increment,
                dependency_data_scale=self.dependency_data_scale,
            )
            self.user_graph_stress[m] = stress_metadata
            #random_graph_key = list(loaded_graphs.keys())[m]
            #random_graph_key = 'j_13027'
            #TODO PYDOT ERROR
            #pos = graphviz_layout(random_graph, prog='dot')
            #plt.figure()
            #nx.draw_networkx(random_graph, pos, with_labels=True, node_color='lightblue')
            #plt.savefig(self.filename_png+f'/DAG_{m}.png')

            # initialzie users
            acc = (0.0,0.0)
            v0 = random.randint(-100,100)/100000
            self.users[m] = User(self.env,id=m, pos0 =user_positions[m] ,  application_graph =random_graph, max_cpu_cycles=self.max_cpu_cycles,
                         max_data_length=self.max_data_length, numberofservices=self.Q, acc=acc, v0=v0, power=self.power,
                         bandwidth=self.BW, numberofservers=self.S,deadline=self.deadline,learning_arguments=learning_arguments,algorithm=self.alg,filename_png=self.filename_png,
                         arrival_time=self.application_arrival_times[m])
        self.service_data_length={}
        self.service_data_length[0] = 0
        for q in range(self.Q):
            self.service_data_length[q+1] = self.min_service_data_length+random.random()*(self.max_service_data_length-self.min_service_data_length)
        
        self.server_latency=np.zeros((self.S,self.S))
        for s in range(0,self.S):
            for sp in range(s+1,self.S):
                    self.server_latency[s,sp]= 1e-3*random.randint(1,5)/10
                    self.server_latency[sp,s]= self.server_latency[s,sp]
        #self.servers[0] = Server(self.env,id=0, numberofservices=self.Q, min_freq=self.min_cpu_freq, max_freq=self.max_cpu_freq,
                                 #xlim=self.xlim, ylim=self.ylim, iscloud=True,minload=self.maxload-1,maxload=self.maxload)
        for s in range(self.S):
            # initialzie servers
            server = Server(
                self.env,
                id=s,
                numberofservices=self.Q,
                min_freq=self.min_cpu_freq,
                max_freq=self.max_cpu_freq,
                xlim=self.xlim,
                ylim=self.ylim,
                iscloud=False,
                minload=self.minload,
                maxload=self.maxload,
                minratetocloud=self.minratetocloud,
                maxratetocloud=self.maxratetocloud,
                capacity=self.server_capacities[s],
                random_draw_capacity=self.baseline_server_capacity,
            )
            self.servers[s] = server
            self.networkgraph.add_node(s, server=server)  # Add server as a node to the graph

        for m in range(self.M):
            self.users[m].attach(self)
        for n in range(self.S):
            self.servers[n].attach(self)
        self.broker.attach(self)

        self.server_rates=np.zeros((self.S,self.S))
        for s in range(self.S):
            for sp in range(self.S):

                if s==sp:
                    self.server_rates[s,sp] = np.inf
                else:
                    if np.linalg.norm(np.array(self.servers[s].pos) - np.array(self.servers[sp].pos)) < 5000:
                        #self.server_rates[s,sp]= shannon_capacity(np.linalg.norm(np.array(self.servers[s].pos) - np.array(self.servers[sp].pos)))
                        self.server_rates[s,sp]= 1e9*(self.max_rate_between_servers-(self.max_rate_between_servers-self.min_rate_between_servers)*np.linalg.norm(np.array(self.servers[s].pos) - np.array(self.servers[sp].pos))/5000)
                        self.networkgraph.add_edge(s, sp, weight=1.0/self.server_rates[s,sp])  # Add edge with rate parameter to represent connection between servers
        # Create a minimum spanning tree (MST) based on the networkgraph
        mst = nx.minimum_spanning_tree(self.networkgraph, weight='weight')
        self.networkgraph = mst
        # Draw the MST
        pos = {server.id: server.pos for server in self.servers.values()}  # Use server positions for the graph layout
        if Input_dict.get('save topology figure', True):
            plt.figure()
            nx.draw(mst, pos, with_labels=True, node_size=50, node_color='green', font_size=8, font_weight='bold')
            plt.title('Minimum Spanning Tree of Servers')
            mst_path = Path(self.filename_png) / 'mst_of_servers.png'
            plt.savefig(mst_path)
            plt.close()
        self.between_server_costs = np.zeros((self.S, self.S))
        for s in range(self.S):
            for sp in range(self.S):
                # Find the path between two nodes in the MST
                source_node = s  # Replace with your source node
                target_node = sp  # Replace with your target node
                path = nx.shortest_path(self.networkgraph, source=source_node, target=target_node, weight='weight')
                # Calculate the sum of weights along the path
                total_weight = sum(mst[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                self.between_server_costs[s, sp] = total_weight
        for m in range(self.M):
            #self.users[m].set_agent(self.agent)
            self.users[m].nearest_server =self.users[m].find_nearest_server(self.servers)
        
        #self.gat = GAT(self.users)
        self.gat = None

        #self.optimizer = Joint_Optimizer(M=self.M,num_s=self.S,tasks= [self.users[m].tasks_init for m in range(self.M)])
        #self.optimizer.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = [self.users[m].to_servers_rate for m in range(self.M)],nearest_server = [self.users[m].nearest_server for m in range(self.M)]) 
        #self.optimizer.solve_minlp()
        #self.optimizer.printvalues()
        
        #self.dp = DynamicProgramming(M=self.M,num_s=self.S,tasks= [self.users[m].tasks_init for m in range(self.M)])
        #self.dp.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = [self.users[m].to_servers_rate for m in range(self.M)],nearest_server = [self.users[m].nearest_server for m in range(self.M)]) 
        self.server_service_info = np.zeros((self.S,self.Q))
        for s in self.servers.values():
            for service in s.services:
                if service>0:
                    self.server_service_info[s.id,service-1] = 1

        if caching_decision := Input_dict.get('caching decision enabled', True):    
            self.broker.broadcast_caching_decisions()

        self._processes_started = False
        
        """
        
        for m in range(self.M):
           services={}
           for n in range(self.users[m].numberofservers):
               services[n] = [key for key, value in self.optimizer.optimal_z[int(n)].items() if value >= 0.9]
           pos={}
           listofnodes=list(self.users[m].DAG.nodes())
           for n in listofnodes:
               if n=='0':
                   pos[n]=(10*self.users[m].nearest_server,0)
               else:
                   y = -10*list(nx.topological_sort(self.users[m].DAG)).index(n)
                   x= 10*np.argmax([self.optimizer.optimal_x[m,int(n),s] for s in range(self.users[m].numberofservers)])
                   pos[n]=(x,y)
           plt.figure(3)
           nx.draw_networkx(self.users[m].DAG, pos, with_labels=True, node_color='lightblue')
           plt.grid(True)
           for t in self.users[m].tasks.values():
               plt.text(pos[t.task_number][0]+5, pos[t.task_number][1]+5, t.service, ha='center', fontsize=10, color='red')

            # Create a graph
           servers_graph = nx.Graph()
           pos_s={}
           for n in range(self.S):
                   servers_graph.add_node(n, service = services[n])
                   pos_s[n]=(n *10, 10)

                # Draw the graph with the grid layout
           nx.draw_networkx(servers_graph, pos=pos_s, with_labels=True, node_size=300, node_color='red', font_size=10)
           for n in range(self.S):
               plt.text(pos_s[n][0], pos_s[n][1]+5, services[n][1], ha='center', fontsize=10, color='red')

           plt.savefig(self.filename_png+f'/DAG_sol{m}.png')
        #for n in range(self.users[m].numberofservers):
        #    self.servers[n].service_caching(services[n])
        """    
    def set_training(self, training, update_caching=None):
        self.training = training
        if update_caching is None:
            update_caching = training and self.input_dict.get(
                'caching decision enabled', True
            )
        self.update_caching = update_caching

        for server in self.servers.values():
            model = server.agent.agent.TrainNet.model
            model.train(mode=training)

    def reset(self):
        for m in range(self.M):
            # initialzie users
            acc = (0,0)
            v0 = (0,0)
            #self.select_random_graph(m,acc,v0)
            #acc = (10 * (random.random() - 0.5), 10 * (random.random() - 0.5))
            #v0 = (10 * (random.random() - 0.5), 10 * (random.random() - 0.5))
            self.users[m].reset()

    def select_random_graph_and_find_optimal_offloading_decision(self,m,acc,v0):
            
            while (True):
                random_graph_key = random.choice(list(self.loaded_graphs.keys()))
                random_graph = self.loaded_graphs[random_graph_key]
                if(len(random_graph.nodes.items())<=self.I):
                    break
            pos = graphviz_layout(random_graph, prog='dot')
            plt.figure()
            nx.draw_networkx(random_graph, pos, with_labels=True, node_color='lightblue')
            plt.savefig(self.filename_png+f'/DAG_{m}/{random_graph_key}.png')
            numberoftasks = self.users[m].task_generate(DAG = random_graph,
                             max_cpu_cycles=self.max_cpu_cycles,
                             max_data_length=self.max_data_length, numberofservices=self.Q, acc=acc, v0=v0,
                             power=self.power,
                             bandwidth=self.BW, numberofservers=self.S)
            
            self.users[m].optimizer = Optimizer(num_v =numberoftasks,num_s=self.S,tasks= self.users[m].tasks_init)
            self.users[m].optimizer.get_server_and_service_parameters(servers =self.servers,service_lengths = self.service_data_length,server_rates = self.server_rates,server_latencies=self.server_latency,to_servers_rate = self.users[m].to_servers_rate,nearest_server = self.users[m].nearest_server)
            self.users[m].optimizer.solve_minlp()
            #self.users[m].optimizer.printvalues()
    
            pos={}
            listofnodes=list(self.users[m].DAG.nodes())
            for n in listofnodes:
                if n=='0':
                    pos[n]=(10*self.users[m].nearest_server,0)
                else:
                    y = -10*list(nx.topological_sort(self.users[m].DAG)).index(n)
                    x= 10*np.argmax([self.users[m].optimizer.optimal_x[int(n),s] for s in range(self.users[m].numberofservers)])
                    pos[n]=(x,y)
            plt.figure(2)
            nx.draw_networkx(self.users[m].DAG, pos, with_labels=True, node_color='lightblue')
            plt.grid(True)
            for t in self.users[m].tasks.values():
                plt.text(pos[t.task_number][0]+5, pos[t.task_number][1]+5, t.service, ha='center', fontsize=10, color='red')

            # Create a graph
            servers_graph = nx.Graph()
            pos_s={}
            for n in range(self.users[m].numberofservers):
                    servers_graph.add_node(n, service = self.servers[n].services)
                    self.servers[n].services
                    pos_s[n]=(n *10, 10)

                # Draw the graph with the grid layout
            nx.draw_networkx(servers_graph, pos=pos_s, with_labels=True, node_size=300, node_color='red', font_size=10)
            for n in range(self.users[m].numberofservers):
                plt.text(pos_s[n][0], pos_s[n][1]+5, self.servers[n].services[1], ha='center', fontsize=10, color='red')

            plt.savefig(self.filename_png+f'/DAG_{m}/{random_graph_key}_sol.png')
    def test(self,epoc):
        text='~~~~~~~\n#run'+str(epoc)+'\n'
        Q_text='~~~~~~~\n#run'+str(epoc)+'\n'
        observations, tasks = self.get_tasks_and_observations(self.time_step)
        done_list={}
        for m in range(self.M):
            done_list[m] = False
            self.users[m].agent.epsilon=0
        complete = False
        Q = {}
        self.time_step=0

        while (not complete):
            actions = self.select_actions(observations,tasks)
            for m in range(self.M):
                Q[m] = self.users[m].agent.agent.TrainNet.predict(np.atleast_2d(observations[m]))[0]
            Q_text += 'Time: '+str(self.time_step) + '\n Q: '+str(Q)+'\n'
            rewards,done_list,complete = self.step(self.time_step,actions,tasks,done_list)
            next_observations, tasks = self.get_tasks_and_observations(self.time_step)
            Q_text +=str(observations)+'\n'+str(actions)+'\n'+str(next_observations)+'\n'+str(rewards)+'\n#########################\n'
            observations = next_observations
            self.time_step += 1
        text+='Deadline: '+str([self.users[m].deadline for m in range(self.M)])+'\n'
        Average_of_finish_time = np.mean([self.users[m].finish_time_of_application for m in range(self.M)])
        minimum_of_finish_time = min([self.users[m].finish_time_of_application for m in range(self.M)])
        text+='Average of finish time: '+ str(Average_of_finish_time)+'\n'
        text+='minimum of finish time: '+ str(minimum_of_finish_time)+'\n'
        finish_times = [self.users[m].finish_time_of_application for m in range(self.M)]
        
        for m in range(self.M):
            pos = {}
            listofnodes=list(self.users[m].DAG.nodes())
            for n in listofnodes:
                if n=='0':
                    pos[n]=(10*self.users[m].nearest_server,0)
                else:
                    y = -10*list(nx.topological_sort(self.users[m].DAG)).index(n)
                    x= 10*self.users[m].done_tasks[n].assigned_server
                    pos[n]=(x,y)
            plt.figure(4)
            nx.draw_networkx(self.users[m].DAG, pos, with_labels=True, node_color='lightblue')
            plt.grid(True)
            for t in self.users[m].tasks.values():
                plt.text(pos[t.task_number][0]+5, pos[t.task_number][1]+5, t.service, ha='center', fontsize=10, color='red')
            # Create a graph
            servers_graph = nx.Graph()
            pos_s={}
            for n in range(self.users[m].numberofservers):
                    servers_graph.add_node(n, service = self.servers[n].services)
                    self.servers[n].services
                    pos_s[n]=(n *10, 10)
                # Draw the graph with the grid layout
            nx.draw_networkx(servers_graph, pos=pos_s, with_labels=True, node_size=300, node_color='red', font_size=10)
            for n in range(1, self.users[m].numberofservers):
                plt.text(pos_s[n][0], pos_s[n][1]+5, self.servers[n].services[1], ha='center', fontsize=10, color='red')
            plt.savefig(self.filename_png+f'/DAG_{m}_RL.png')    
        return text,Q_text,Average_of_finish_time,minimum_of_finish_time,finish_times
    def start_processes(self):
        """Schedule simulator actors once and allow external observers."""
        if self._processes_started:
            raise RuntimeError("simulator processes were already started")
        self.notcomplete=True
        for m in range(self.M):
            self.env.process(self.users[m].run())
        for n in range(self.S):
            self.env.process(self.servers[n].run())
        self.env.process(self.broker.run())
        self._processes_started = True

    def run(self):
        #start_time = time.time()  # Start time for the iteration
        self.start_processes()
        self.env.run()
        self._processes_started = False
        Simulation_Time = self.time_step
        Average_of_finish_time = np.mean([self.users[m].finish_time_of_application for m in range(self.M)])
        minimum_of_finish_time = min([self.users[m].finish_time_of_application for m in range(self.M)])
        #average_optimal_value = self.optimizer.optimal_objective
        #optimal_values = [self.optimizer.optimal_finishtime[m] for m in range(self.M)]
        optimal_values = [self.users[m].finish_time_of_application for m in range(self.M)]
        #dp_values = [self.dp.application_finish_time[m] for m in range(self.M)]
        dp_values = [self.users[m].finish_time_of_application for m in range(self.M)]


        finish_times = [self.users[m].finish_time_of_application for m in range(self.M)]
        

        #print('Simulation Time: ', Simulation_Time)
        #print('Average of finish time: ', Average_of_finish_time)
        #print('minimum of finish time: ', minimum_of_finish_time)
        self.calculate_latency_shares()
        #end_time = time.time()  # End time for the iteration
        #elapsed_time = end_time - start_time
        #print(f"simulator Iteration took {elapsed_time:.10f} seconds")

        return optimal_values,finish_times,dp_values

    def plot_positions(self,iter=0):
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 8))
        server_positions = []
        user_positions = []

        # Plot horizontal streets with spacing
        for y in self.street_positions_y:
            ax.plot([0, self.xlim], [y, y], color='green', linewidth=1, linestyle='--', label='Horizontal Street' if y == self.street_positions_y[0] else "")
            # Show the width of each street
            ax.fill_between([0, self.xlim], y, y + self.street_width, color='green', alpha=0.2)

        # Plot vertical streets with spacing
        for x in self.street_positions_x:
            ax.plot([x, x], [0, self.ylim], color='purple', linewidth=1, linestyle='--', label='Vertical Street' if x == self.street_positions_x[0] else "")
            # Show the width of each street
            ax.fill_betweenx([0, self.ylim], x, x + self.street_width, color='purple', alpha=0.2)

        # Collect server and user positions
        for s in self.servers.values():
            server_positions.append(s.pos)
        for u in self.users.values():
            user_positions.append(u.pos)

        # Plot servers (red circles) and users (blue squares)
        if server_positions:
            ax.scatter(*zip(*server_positions), color='red', marker='o', label='Servers')
        if user_positions:
            ax.scatter(*zip(*user_positions), color='blue', marker='s', label='Users')

        # Set plot limits, labels, and title
        ax.set_xlim(0, self.xlim)
        ax.set_ylim(0, self.ylim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Server and User Positions')
        ax.grid(True)

        # Add legend
        ax.legend()
        fig.savefig(f'locations.png')

        pos = {server.id: server.pos for server in self.servers.values()}  # Use server positions for the graph layout
        nx.draw(self.networkgraph, pos, with_labels=True, ax=ax, node_size=5, node_color='red', font_size=1, font_weight='bold')
        #fig.savefig('graph_of_servers.png')
        #plt.close(fig)
                # Save the figure as 'locations.png'
        # Create a minimum spanning tree (MST) based on the server distances
        #mst = nx.minimum_spanning_tree(self.networkgraph, weight='weight')
        #nx.draw(mst, pos, with_labels=True, ax=ax, node_size=5, node_color='red', font_size=1, font_weight='bold')
        # Save the MST figure as 'mst_of_servers.png'
        fig.savefig(self.filename_png+'/graph_of_servers.pdf', dpi=300)  # Save as PDF with tight bounding box

        plt.close(fig)

        # plt.figure()
        # plt.title("Graph of Servers")
        # plt.savefig('graph_of_servers.png')  # Save the graph as an image file
        # plt.close()
        #plt.show()  


    def calculate_latency_shares(self):
        # Collect latency components from all users
        computing_latencies = [self.users[m].sum_computinglatency for m in range(self.M)]
        data_transfer_latencies = [self.users[m].sum_datatransferlatency for m in range(self.M)]
        pred_latencies = [self.users[m].sum_predlatency for m in range(self.M)]
        service_latencies = [self.users[m].sum_servicelatency for m in range(self.M)]
        waiting_latencies = [self.users[m].sum_waiting_latency for m in range(self.M)]

        # Compute the average latency components across all users
        self.avg_computing_latency = np.mean(computing_latencies)
        self.avg_data_transfer_latency = np.mean(data_transfer_latencies)
        self.avg_pred_latency = np.mean(pred_latencies)
        self.avg_service_latency = np.mean(service_latencies)
        self.avg_waiting_latency = np.mean(waiting_latencies)

        # Calculate the total latency for each user and the average of total latencies
        total_latencies = [comp + dt + pred + serv + wait for comp, dt, pred, serv, wait in zip(
            computing_latencies, data_transfer_latencies, 
            pred_latencies, service_latencies, 
            waiting_latencies)]
        avg_total_latency = np.mean(total_latencies)
        # Calculate the contribution share for each latency component
        if avg_total_latency > 0:
            self.computing_share = self.avg_computing_latency / avg_total_latency
            self.data_transfer_share = self.avg_data_transfer_latency / avg_total_latency
            self.pred_share = self.avg_pred_latency / avg_total_latency
            self.service_share = self.avg_service_latency / avg_total_latency
            self.waiting_share = self.avg_waiting_latency / avg_total_latency
        else:
            self.computing_share = 0.0
            self.data_transfer_share = 0.0
            self.pred_share = 0.0
            self.service_share = 0.0
            self.waiting_share = 0.0

        # Optional: print or store the values for debugging
        #print(f"Avg Computing Latency Share: {self.computing_share}")
        #print(f"Avg Data Transfer Latency Share: {self.data_transfer_share}")
        #print(f"Avg Prediction Latency Share: {self.pred_share}")
        #print(f"Avg Service Latency Share: {self.service_share}")
        #print(f"Avg Waiting Latency Share: {self.waiting_share}")
