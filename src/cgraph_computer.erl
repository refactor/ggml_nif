-module(cgraph_computer).
-behaviour(gen_statem).

-include_lib("kernel/include/logger.hrl").

%% API.
-export([start_link/1]).
-export([do_compute/1]).

%% gen_statem.
-export([callback_mode/0]).
-export([init/1]).
-export([graph_waiting/3]).
-export([node_forward/3]).
-export([node_partition_computing/3]).
-export([terminate/3]).
-export([code_change/4]).
-export([compute_node_partition/3]).

-record(state, {
    tensor,
    cgraph,
    current_node,
    n_tasks,
    compute_params,
    from
}).

%% API.

-spec start_link(any()) -> {ok, pid()}.
start_link(Tensor) ->
	gen_statem:start_link(?MODULE, Tensor, []).

do_compute(Pid) ->
    gen_statem:call(Pid, do_compute).

%% gen_statem.

callback_mode() ->
	[state_functions,state_enter].

init(Tensor) ->
    ?LOG_DEBUG("nthread: ~p", [erlang:system_info(dirty_cpu_schedulers)]),
    Graph = ggml_nif:build_forward(Tensor),
    ggml_nif:graph_init_workbuf(Graph),
	{ok, graph_waiting, #state{tensor=Tensor, cgraph=Graph}, []}.

graph_waiting(enter, OldState, _StateData) ->
    ?LOG_DEBUG("enter... ~p, waiting for call", [OldState]),
    keep_state_and_data;
graph_waiting({call,From}, OldState, StateData) ->
    ?LOG_DEBUG("graph_waiting... state: old: ~p called from: ~p~n", [OldState, From]),
    {next_state, node_forward, StateData#state{from=From}}.

node_forward(enter, OldState, #state{cgraph=Graph, from=From, tensor=Tensor} = StateData) ->
    ?LOG_DEBUG("enter... from oldstate: ~p From: ~p", [OldState, From]),
    case ggml_nif:graph_iter_node(Graph) of
        {ok, Node} ->
            {keep_state, StateData#state{current_node=Node}, {timeout,0,do_comp}};
        {error,_} ->
            {stop_and_reply, normal, {reply, From, ggml_nif:get_data(Tensor)}}
    end;
node_forward(EventType, EventContent, #state{current_node=Node}=StateData) ->
    ComputeParams = ggml_nif:node_compute_params(Node),
    ?LOG_DEBUG("node_forward: type=~p, content=~p, ComputeParams: ~p", [EventType,EventContent,ComputeParams]),
    {next_state, node_partition_computing, StateData#state{n_tasks=length(ComputeParams), compute_params=ComputeParams}}.

node_partition_computing(enter, OldState, #state{cgraph=Graph,current_node=Node,compute_params=ComputeParams}) ->
    ?LOG_DEBUG("node_partition_computing: ~p", [OldState]),
    [erlang:monitor(process, spawn(?MODULE, compute_node_partition, [Graph, {I,NT}, Node])) || {I,NT} <- ComputeParams],
    keep_state_and_data;
node_partition_computing(info, {'DOWN', _Ref, process, _Pid, normal}, #state{n_tasks=1}=StateData) ->
    ?LOG_DEBUG("node_partition"),
    {next_state, node_forward, StateData, {timeout, 0, part_comp}};
node_partition_computing(info, {'DOWN', _Ref, process, _Pid, normal}, #state{n_tasks=N}=StateData) ->
    ?LOG_DEBUG("node_partition: ~p", [N]),
    {keep_state, StateData#state{n_tasks=N-1}};
node_partition_computing(EventType, EventContent, _StateData) ->
    ?LOG_WARNING("type: ~p, content: ~p", [EventType,EventContent]),
    keep_state_and_data.


terminate(_Reason, _StateName, _StateData) ->
	ok.

code_change(_OldVsn, StateName, StateData, _Extra) ->
	{ok, StateName, StateData}.

compute_node_partition(Graph, {I,N}, Node) ->
    Params = ggml_nif:create_compute_params(Graph, {I, N}),
    ggml_nif:compute_forward(Params, Node).
