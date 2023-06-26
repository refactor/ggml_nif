-module(cgraph_computer).
-behaviour(gen_statem).

%% API.
-export([start_link/1]).
-export([do_compute/1]).

%% gen_statem.
-export([callback_mode/0]).
-export([init/1]).
-export([next_node/3]).
-export([node_part_computed/3]).
-export([handle_event/4]).
-export([terminate/3]).
-export([code_change/4]).
-export([node_part_compute/4]).

-record(state, {
    tensor,
    cgraph,
    current_node,
    node_partitions = [],
    nthread,
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
	{ok, next_node, #state{nthread=N, tensor=Tensor, cgraph=Graph}, []}.

node_part_computed(info, {node_partition, I}, #state{node_partitions=NP,nthread=N}=StateData) ->
    io:format("~p: node_partition: ~p/~p~n", [?LINE,I, NP]),
    NewNP = [I|NP],
    if length(NewNP) == N -> {next_state, next_node, StateData#state{node_partitions=[]}};
        true -> {next_state, node_part_computed, StateData#state{node_partitions=NewNP}}
    end;
node_part_computed(EventType, EventContent, StateData) ->
    io:format("~p: type: ~p, content: ~p~n", [?LINE, EventType,EventContent]),
	{next_state, node_part_computed, StateData}.

next_node(enter, OldState, #state{cgraph=Graph, from=From, tensor=Tensor} = StateData) ->
    io:format("enter... state: old: ~p~n", [OldState]),
    case ggml_nif:graph_iter_node(Graph) of
        {ok, Node} ->
%            gen_statem:send_request(self(), foo),
            {keep_state, StateData#state{current_node=Node}};
        {error,_} ->
            {stop_and_reply, normal, {reply, From, ggml_nif:get_data(Tensor)}}
    end;
next_node({call, From}, EventContent, #state{cgraph=Graph,current_node=Node,nthread=N}=StateData) ->
    io:format("#~p, next_node: type=call ~p, data=~p~n", [?LINE, From,EventContent]),
    Pid = self(),
    [spawn(?MODULE, node_part_compute, [Pid, Graph, {I,N}, Node]) || I <- lists:seq(0, N-1)],
    {next_state, node_part_computed, StateData#state{from=From}};
next_node(EventType, EventContent, StateData) ->
    io:format("#~p, next_node: type=~p, data=~p~n", [?LINE, EventType,EventContent]),
    {next_state, node_part_computed, StateData}.

handle_event(EventType, EventContent, StateName, StateData) ->
    io:format("#~p: handle_event: state=~p, type=~p, data=~p~n", [?LINE,StateName, EventType,EventContent]),
	{next_state, StateName, StateData}.

terminate(_Reason, _StateName, _StateData) ->
	ok.

code_change(_OldVsn, StateName, StateData, _Extra) ->
	{ok, StateName, StateData}.

node_part_compute(Pid, Graph, {I,N}, Node) ->
    Params = ggml_nif:create_compute_params(Graph, {I, N}),
    ggml_nif:compute_forward(Params, Node),
    Pid ! {node_partition, I}.
