-module(cgraph_computer).
-behaviour(gen_statem).

-include_lib("kernel/include/logger.hrl").

%% API.
-export([start_link/1]).
-export([stop/1]).
-export([do_compute/1]).
-export([do_compute/2]).

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
    input,
    cgraph,
    probs,
    current_node,
    n_tasks,
    compute_params,
    from
}).

%% API.

-spec start_link(any()) -> {ok, pid()}.
start_link(NN) ->
	gen_statem:start_link(?MODULE, NN, []).

stop(Pid) ->
    gen_statem:stop(Pid).

do_compute(Pid) ->
    gen_statem:call(Pid, do_compute).

do_compute(Pid, NewInput) ->
    gen_statem:call(Pid, {do_compute, NewInput}).

%% gen_statem.

callback_mode() ->
	[state_functions,state_enter].

init({InputTensor, ProbsTensor}) ->
    ?LOG_DEBUG("nthread: ~p", [erlang:system_info(dirty_cpu_schedulers)]),
    Graph = ggml_nif:build_forward(ProbsTensor),
    ggml_nif:graph_init_workbuf(Graph),
	{ok, graph_waiting, #state{input=InputTensor, probs=ProbsTensor, cgraph=Graph}, []}.

graph_waiting(enter, OldState, _StateData) ->
    ?LOG_DEBUG("enter from old-state: ~p, waiting for call...", [OldState]),
    keep_state_and_data;
graph_waiting({call,From}, {do_compute, DigitBin}, #state{input=InputTensor} = StateData) ->
    ?LOG_DEBUG("prepare for new digit input"),
    ggml_nif:tensor_load(InputTensor, DigitBin),
    {next_state, node_forward, StateData#state{from=From}};
graph_waiting({call,From}, do_compute, StateData) ->
    ?LOG_DEBUG("graph_waiting... called from: ~p~n", [From]),
    {next_state, node_forward, StateData#state{from=From}};
graph_waiting(EventType, EventContent, StateData) ->
    ?LOG_DEBUG("graph_waiting: ~p, ~p, ~p", [EventType, EventContent, StateData]),
    keep_state_and_data.

node_forward(enter, OldState, #state{from=From, probs=ProbsTensor}) ->
    ?LOG_DEBUG("enter... from oldstate: ~p From: ~p, probs: ~p", [OldState, From, ProbsTensor]),
    {keep_state_and_data, {timeout, 0, part_comp_done}};
node_forward(timeout, part_comp_done, #state{cgraph=Graph, from=From, probs=ProbsTensor}=StateData) ->
    ?LOG_DEBUG("node_partition_computing over, lets move on......", []),
    case ggml_nif:graph_iter_node(Graph) of
        {ok, Node} ->
            {keep_state, StateData#state{current_node=Node}, {timeout,0,do_comp}};
        {error,_} ->
            ?LOG_DEBUG("DONE graph_compute! replying"),
            %gen_statem:reply(From, ggml_nif:get_data(ProbsTensor)),
            {next_state, graph_waiting, StateData#state{from=undefined}, {reply, From, ggml_nif:get_data(ProbsTensor)}}
            %{keep_state, StateData#state{from=undefined}, {reply, From, ggml_nif:get_data(ProbsTensor)}}
%            {stop_and_reply, normal, {reply, From, ggml_nif:get_data(Tensor)}}
    end;
node_forward(timeout, do_comp, #state{current_node=Node}=StateData) ->
    ComputeParams = ggml_nif:node_compute_params(Node),
    ?LOG_DEBUG("node_forward:  ComputeParams: ~p", [ComputeParams]),
    {next_state, node_partition_computing, StateData#state{n_tasks=length(ComputeParams), compute_params=ComputeParams}};
node_forward(EventType, EventContent, #state{from=undefined}=StateData) ->
    ?LOG_DEBUG("node_forward... type: ~p, content: ~p, state: ~p", [EventType, EventContent, StateData]),
    {next_state, graph_waiting, StateData}.

node_partition_computing(enter, OldState, #state{cgraph=Graph,current_node=Node,compute_params=ComputeParams}) ->
    ?LOG_DEBUG("node_partition_computing: ~p", [OldState]),
    [erlang:monitor(process, spawn(?MODULE, compute_node_partition, [Graph, {I,NT}, Node])) || {I,NT} <- ComputeParams],
    keep_state_and_data;
node_partition_computing(info, {'DOWN', _Ref, process, _Pid, normal}, #state{n_tasks=1}=StateData) ->
    ?LOG_DEBUG("node_partition"),
    {next_state, node_forward, StateData#state{compute_params=undefined}, {timeout, 0, part_comp_done}};
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
