-module(model_predict_fun).

-export[start/1].
-export([do_compute/1]).
-export([compute_node_partition/3]).

start(PredTensor) ->
    Graph = ggml_nif:build_forward(PredTensor),
    ggml_nif:graph_init_workbuf(Graph),
    do_compute(Graph),
    {Time, _Res} =timer:tc(?MODULE, do_compute, [Graph], millisecond),
    io:format("docompute time: ~p~n", [Time]),
    ggml_nif:get_data(PredTensor).

do_compute(Graph) ->
    node_forward(Graph, ggml_nif:graph_iter_node(Graph)).

node_forward(Graph, {ok, Node}) ->
    ComputeParams = ggml_nif:node_compute_params(Node),
    [erlang:monitor(process, spawn(?MODULE, compute_node_partition, [Graph, {I,NT}, Node])) || {I,NT} <- ComputeParams],
    node_partitions(length(ComputeParams)),
    node_forward(Graph, ggml_nif:graph_iter_node(Graph));
node_forward(_Graph, {error, _}) ->
    done.

node_partitions(0) ->
    over;
node_partitions(NTasks) ->
    receive 
        {'DOWN', _Ref, process, _Pid, normal} ->
            node_partitions(NTasks - 1);
        Other ->
            io:format("something wrong: ~p~n", [Other])
    end.

compute_node_partition(Graph, {I,N}, Node) ->
    Params = ggml_nif:create_compute_params(Graph, {I, N}),
    ggml_nif:compute_forward(Params, Node).
