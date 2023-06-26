-module(cgraph_computer).
-behaviour(gen_statem).

%% API.
-export([start_link/1]).
-export([do_compute/1]).

%% gen_statem.
-export([callback_mode/0]).
-export([init/1]).
-export([forward_waiting/3]).
-export([node_forward/3]).
-export([node_part_computed/3]).
-export([handle_event/4]).
-export([terminate/3]).
-export([code_change/4]).
-export([compute_node_partition/4]).

-record(state, {
    tensor,
    cgraph,
    current_node,
    node_partitions = [],
    n_tasks,
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
    %Graph = ggml_nif:graph_build(Tensor),
    N = erlang:system_info(dirty_cpu_schedulers),
    io:format("nthread: ~p~n", [N]),

    Graph = ggml_nif:build_forward(Tensor),
    ggml_nif:graph_init_workbuf(Graph),

	{ok, forward_waiting, #state{n_tasks=N, tensor=Tensor, cgraph=Graph}, []}.

forward_waiting(enter, OldState, _StateData) ->
    io:format("#~p: enter... ~p, waiting for call", [?LINE, OldState]),
    keep_state_and_data;
forward_waiting({call,From}, OldState, StateData) ->
    io:format("#~p: forward_waiting... state: old: ~p called from: ~p~n", [?LINE, OldState, From]),
    {next_state, node_forward, StateData#state{from=From}}.

node_forward(enter, OldState, #state{cgraph=Graph, from=From, tensor=Tensor} = StateData) ->
    io:format("#~p: enter... from state: old: ~p From: ~p~n", [?LINE, OldState, From]),
    case ggml_nif:graph_iter_node(Graph) of
        {ok, Node} ->
%            gen_statem:send_request(self(), foo),
            {keep_state, StateData#state{current_node=Node}, {timeout,0,do_comp}};
        {error,_} ->
            {stop_and_reply, normal, {reply, From, ggml_nif:get_data(Tensor)}}
    end;
node_forward(EventType, EventContent, #state{cgraph=Graph,current_node=Node}=StateData) ->
    ComputeParams = ggml_nif:node_compute_params(Node),
    io:format("#~p, node_forward: type=~p, content=~p, ComputeParams: ~p~n", [?LINE, EventType,EventContent,ComputeParams]),
    Pid = self(),
    [spawn(?MODULE, compute_node_partition, [Pid, Graph, {I,NT}, Node]) || {I,NT} <- ComputeParams],
    {next_state, node_part_computed, StateData#state{n_tasks=length(ComputeParams)}}.

node_part_computed(info, {node_partition, I}, #state{node_partitions=NP,n_tasks=N}=StateData) ->
    io:format("~p: node_partition: ~p/~p~n", [?LINE,I, NP]),
    NewNP = [I|NP],
    if length(NewNP) == N -> {next_state, node_forward, StateData#state{node_partitions=[]}, {timeout,0,part_comp}};
        true -> {next_state, node_part_computed, StateData#state{node_partitions=NewNP}}
    end;
node_part_computed(EventType, EventContent, StateData) ->
    io:format("~p: type: ~p, content: ~p~n", [?LINE, EventType,EventContent]),
	{next_state, node_part_computed, StateData}.

handle_event(EventType, EventContent, StateName, StateData) ->
    io:format("#~p: handle_event: state=~p, type=~p, data=~p~n", [?LINE,StateName, EventType,EventContent]),
	{next_state, StateName, StateData}.

terminate(_Reason, _StateName, _StateData) ->
	ok.

code_change(_OldVsn, StateName, StateData, _Extra) ->
	{ok, StateName, StateData}.

compute_node_partition(Pid, Graph, {I,N}, Node) ->
    Params = ggml_nif:create_compute_params(Graph, {I, N}),
    ggml_nif:compute_forward(Params, Node),
    Pid ! {node_partition, I}.
