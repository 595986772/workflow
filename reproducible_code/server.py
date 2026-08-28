import random

from utils import generate_random_pos
import numpy as np
from task import Task
from agent import Agent
import time
import torch
import simpy
from input import GCN_paramaters as gcn_paramaters
from input import INPUT_DICT as input_dict
from sample_efficient_guidance import HistoricalFeedbackGuide


class Server:
    """Server Class

    This class defines a server.

    Attributes
    ----------
        id : int
            the ID of the server
        pos : tuple(float,float)
            server's position
        services : list[int]
            The list of the services on server
        task_queue : list[Task]
            The list of the task assigned to the server
        frequency : float
            CPU cycle frequency of the server
        ongoing_task: Task
            indicate the ongoing task
    .. note::
        for now, server's position, server's frequency and the services on the server is assigned randomly.
    """

    def __init__(
        self,
        env,
        id,
        numberofservices,
        xlim,
        ylim,
        min_freq,
        max_freq,
        iscloud,
        minload,
        maxload,
        minratetocloud,
        maxratetocloud,
        capacity=None,
        random_draw_capacity=None,
    ):
        """Initiates a Server object.

        Arg
        ----------
            id : int
                The id of the server
            numberofservices : int
                The number of services
            xlim :
                Maximum of x
            ylim
                Maximum of y
            min_freq
                minimum of cpu frequency
            max_freq
                maximum of cpu frequency

        Returns
        -------
        None

        """
        self.simulator=None
        self.env=env
        self.id = id
        if capacity is None:
            capacity = input_dict['server capacity']
        self.capacity = int(capacity)
        if random_draw_capacity is None:
            random_draw_capacity = self.capacity
        random_draw_capacity = int(random_draw_capacity)
        if not 0 <= self.capacity <= numberofservices:
            raise ValueError(
                "capacity must be between zero and numberofservices"
            )
        if not self.capacity <= random_draw_capacity <= numberofservices:
            raise ValueError(
                "random_draw_capacity must be at least capacity and "
                "no greater than numberofservices"
            )
        # define server position

        #assign random service for the server
        self.numberofservices = numberofservices
        self.task_queue=[]
        self.load = np.random.randint(low=5*minload, high=5*maxload)/5.0 #TODO set PARAMETERS
        self.ongoing_task=None
        if iscloud:
            self.services = list(range(0, numberofservices+1))
            self.frequency = max_freq * (10 ** 9)
            self.pos = (250*xlim,250*ylim)
            self.rate_to_cloud = 10000 * (10 ** 9)
        else:
            sampled_services = random.sample(
                range(1, numberofservices + 1),
                random_draw_capacity,
            )
            self.services = [0] + sampled_services[:self.capacity]
            #self.services=[0]  #TODO set PARAMETERS
            self.frequency = np.random.randint(low=5*min_freq, high=5*max_freq)/5.0 * (10 ** 9)
            self.pos = generate_random_pos(xlim, ylim)
            
            self.rate_to_cloud = np.random.randint(low=minratetocloud, high=maxratetocloud) * (10 ** 9)  # TODO set PARAMETERS
        self.statistic={}
        self.statistic['service required counter']={}
        self.statistic['service required counter'][0]=0
        self.statistic['service required counter'][1]=0
        self.time_step = 0
        self.numberofconnectedusers = 0
        self.servers_with_service=[]
        self.policy_inference_calls = 0
        self.policy_inference_wall_time_sec = 0.0
    def attach(self,simulator):
        self.simulator=simulator
        self.dynamic_compute_resource = (
            simpy.Resource(self.env, capacity=1)
            if self.simulator.dynamic_queueing
            else None
        )
        self.agent = Agent(
            algorithm=self.simulator.alg,
            learning_arguments=self.simulator.learning_arguments,
            numberofservers=self.simulator.S,
            numberofservices=self.simulator.Q,
            max_cpu_cycles=self.simulator.max_cpu_cycles,
            max_data_length=self.simulator.max_data_length,
            filename_png=None,
            task_dependency_features_enabled=(
                self.simulator.task_dependency_features_enabled
            ),
        )
        self.history_feedback_guide = None
        if self.simulator.history_feedback_guidance:
            self.history_feedback_guide = HistoricalFeedbackGuide(
                number_of_servers=self.simulator.S,
                alpha=self.simulator.history_feedback_alpha,
                min_samples=self.simulator.history_feedback_min_samples,
            )
        
        self.task_gcn = None
        self.gcn_optimizer = None
        if self.simulator.alg in ('GCN', 'GCN_DQN'):
            from gcn import TaskGCN
            self.task_gcn = TaskGCN(
                self.numberofservices,
                gcn_paramaters['layers'],
                gcn_paramaters['output_dim'],
                dropout=gcn_paramaters['dropout'],
            )
            self.gcn_optimizer = torch.optim.Adam(self.task_gcn.parameters(), lr=0.01)

    def update_weights(self, client_state_dict, coeff):
        # Get the server's current model state_dict
        server_state_dict = self.agent.agent.TrainNet.model.state_dict()
        
        # Create a new state_dict to store the updated (averaged) weights
        averaged_state_dict = {}
        
        # Iterate over the parameters and perform weighted averaging
        for key in server_state_dict.keys():
            server_param = server_state_dict[key]
            client_param = client_state_dict[key].to(server_param.device)
            
            # Compute the weighted average
            averaged_param = (1 - coeff) * server_param + coeff * client_param
            
            # Store the averaged parameter
            averaged_state_dict[key] = averaged_param
        
        # Load the averaged weights into the server's model
        self.agent.agent.TrainNet.model.load_state_dict(averaged_state_dict)
        
        # Update the server's target network if applicable
        #self.agent.agent.update_target_model()

    def add_task(self,task: Task,time):
        """add a task to the task queue of the server.

        Parameters
        ----------
            task : Task
                The task that will be added to the task queue
            time : int
                The time slot when the task is appended to the queue

        Returns
        -------
        None

        """
        self.task_queue.append(task)

        task.result.waiting_time.set_start(time)
        #print(f"Task added to server {self.id}. New task_queue size: {len(self.task_queue)}")


    def execute_task(self,task):
        servers_with_service = [
            server
            for server in self.simulator.servers.values()
            if task.service in server.services
        ]
        self.servers_with_service = servers_with_service
        cache_hit = task.service in self.services
        if cache_hit:
            alpha = 0
            self.statistic['service required counter'][0]+=1
            serviceloadingcost = 0
        else:
            alpha = 1
            self.statistic['service required counter'][1]+=1
            if len(servers_with_service) == 0:
                serviceloadingcost = 1.0/self.rate_to_cloud
            else:
                serviceloadingcost = min(
                    self.simulator.between_server_costs[
                        server.id,
                        self.id,
                    ]
                    for server in servers_with_service
                )
        task.assigned_server = self.id
        task.cache_hit = bool(cache_hit)
        task.remote_service_loaded = not cache_hit
        background_wait = self.load * (10**6) / self.frequency
        task.result.service_latency = (
            alpha
            * self.simulator.service_data_length[task.service]
            * serviceloadingcost
        )
        task.result.computing_latency = task.cpu_cycle / self.frequency
        fetch_cost = min(
            [1.0 / self.rate_to_cloud]
            + [
                self.simulator.between_server_costs[
                    server.id,
                    self.id,
                ]
                for server in servers_with_service
            ]
        )
        if self.simulator.dynamic_queueing:
            task.done = False
            task.completion_event = self.env.event()
            self.env.process(
                self._execute_dynamic_task(
                    task,
                    background_wait=background_wait,
                    fetch_cost=fetch_cost,
                )
            )
            return

        task.result.waiting_latency = background_wait
        task.result.finish_time = (
            task.result.pred_latency
            + task.result.data_transfer_latency
            + task.result.service_latency
            + task.result.computing_latency
            + task.result.waiting_latency
        )
        task.done = True
        self._finalize_task(task, fetch_cost)

    def _execute_dynamic_task(self, task, background_wait, fetch_cost):
        queue_ready_time = (
            max(float(task.result.pred_latency), float(self.env.now))
            + float(task.result.data_transfer_latency)
            + float(task.result.service_latency)
            + float(background_wait)
        )
        if queue_ready_time > self.env.now:
            yield self.env.timeout(queue_ready_time - self.env.now)
        queued_at = float(self.env.now)
        with self.dynamic_compute_resource.request() as request:
            yield request
            queue_wait = float(self.env.now) - queued_at
            task.queue_waiting_latency = queue_wait
            task.result.waiting_latency = (
                float(background_wait) + queue_wait
            )
            yield self.env.timeout(task.result.computing_latency)
        task.result.finish_time = float(self.env.now)
        task.done = True
        self._finalize_task(task, fetch_cost)
        if not task.completion_event.triggered:
            task.completion_event.succeed(task)

    def _finalize_task(self, task, fetch_cost):
        self.numberofconnectedusers -= 1
        if self.simulator.update_caching:
            self.simulator.broker.caching_decisions_update(
                server=self.id,
                service=task.service,
                cost=fetch_cost,
                task=task,
            )

    def run(self):
        self.time_step += 1
        #start_time = time.time()  # Start time for the iteration
        while(self.simulator.notcomplete):
            yield self.env.timeout(1)
                #print(f"Task run on server {self.id}. New task_queue size: {len(self.task_queue)}")
                
        
        #end_time = time.time()  # End time for the iteration
        #elapsed_time = end_time - start_time
        #print(f"server Iteration took {elapsed_time:.10f} seconds")
        if self.simulator.training and self.simulator.learning_enabled:
            self.agent.agent.decay_epsilon()
            if self.time_step % self.simulator.filling_steps == 0:
                self.update_target_model()

            if self.time_step % self.simulator.steps_b_updates == 0:
                self.update_predict_model()
    """                 if (self.file!=None):
                        self.file.write('task: ' +str(m)+'-'+str(tasks[m].task_number) + '\n')
                        self.file.write('assigned server: ' +str(m)+'-'+str(tasks[m].assigned_server) + '\n')
                        self.file.write('pred_latency: '+str(tasks[m].result.pred_latency)+'\n')
                        self.file.write('service_latency: ' + str(tasks[m].result.service_latency) + '\n')
                        self.file.write('data_transfer_latency: ' + str(tasks[m].result.data_transfer_latency) + '\n')
                        self.file.write('computing_latency: ' + str(tasks[m].result.computing_latency)+ '\n')
                        self.file.write('waiting_latency: ' +str(tasks[m].result.waiting_latency) + '\n')
                        self.file.write('finish_time: ' +str(tasks[m].result.finish_time) + '\n#########################################\n')
    """        
        #print('rewards:', rewards)
        #print('done_list:', rewards)
        #print('complete:', complete)
            

    def update_target_model(self):
        self.agent.agent.update_target_model()
    def update_predict_model(self):
        self.agent.agent.replay()
        #self.agent.agent.replay(self.task_gcn,self.gcn_optimizer)
        #self.task_embeddings = self.task_gcn(self.node_features, self.edge_index)
    def offloading_desicion(self,observation,task):
        self.servers_with_service = [server for server in self.simulator.servers.values()  if task.service in server.services]
        if (
            self.simulator.training
            and getattr(
                self.agent.agent,
                "stochastic_policy",
                False,
            )
        ):
            started = time.perf_counter()
            action = self.agent.agent.sample_action(observation)
            self.policy_inference_calls += 1
            self.policy_inference_wall_time_sec += (
                time.perf_counter() - started
            )
            return action
        if (
            self.simulator.training
            and np.random.random() < self.agent.agent.epsilon
        ):
            return np.random.choice(self.agent.agent.TrainNet.num_actions)
        else:
            started = time.perf_counter()
            action = self.agent.agent.TrainNet.get_action(observation)
            self.policy_inference_calls += 1
            self.policy_inference_wall_time_sec += (
                time.perf_counter() - started
            )
            return action
