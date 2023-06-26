-module(ggml_nif).

-export([load/0]).
-export([hello/1]).
-export([init/1]).
-export([new_tensor_f32_3d/4]).
-export([new_tensor_f32_2d/3]).
-export([new_tensor_f32_1d/2]).
-export([tensor_load/2]).
-export([tensor_set_f32/2]).
-export([grad_set_f32/2]).
-export([add/3]).
-export([mul/3]).
-export([mul_mat/3]).
-export([relu/2]).
-export([sum/2]).
-export([soft_max/2]).
-export([graph_compute/1]).
-export([graph_init_workbuf/1]).
-export([build_forward/1]).
-export([build_backward/3]).
-export([graph_reset/1]).
-export([graph_iter_node/1]).
-export([create_compute_params/2]).
-export([compute_forward/2]).
-export([graph_dump_dot/2]).
-export([set_param/1, set_param/2]).
-export([nbytes/1]).
-export([set_name/2]).
-export([get_data/1]).
-export([get_grad/1]).
-export([f32_sizef/0]).

-export_type([my_context/0, my_tensor/0, my_graph/0, my_compute_params/0]).

-nifs([hello/1]).
-nifs([new_tensor_f32_3d/4]).
-nifs([new_tensor_f32_2d/3]).
-nifs([new_tensor_f32_1d/2]).
-nifs([tensor_load/2]).
-nifs([tensor_set_f32/2]).
-nifs([grad_set_f32/2]).
-nifs([add/3]).
-nifs([mul/3]).
-nifs([mul_mat/3]).
-nifs([relu/2]).
-nifs([sum/2]).
-nifs([soft_max/2]).
-nifs([graph_compute/1]).
-nifs([graph_init_workbuf/1]).
-nifs([build_forward/1]).
-nifs([build_backward/3]).
-nifs([graph_reset/1]).
-nifs([graph_iter_node/1]).
-nifs([create_compute_params/2]).
-nifs([compute_forward/2]).
-nifs([graph_dump_dot/2]).
-nifs([set_param/1, set_param/2]).
-nifs([nbytes/1]).
-nifs([set_name/2]).
-nifs([get_data/1]).
-nifs([get_grad/1]).
-nifs([f32_sizef/0]).
-nifs([init/1]).

-deprecated({graph_compute, 1, "just for test, using gen_statem instead"}).

-opaque my_context() :: reference().
-opaque my_tensor() :: reference().
-opaque my_graph() :: reference().
-opaque my_compute_params() :: reference().

-on_load(load/0).

load() ->
    erlang:load_nif("priv/ggml_enif", 0).

-spec init(integer()) -> my_context().
init(_ByteSize) ->
    erlang:nif_error("NIF library not loaded").

-spec new_tensor_f32_3d(my_context(), non_neg_integer(), non_neg_integer(), non_neg_integer()) -> my_tensor().
new_tensor_f32_3d(_Ctx, _NE0, _NE1, _NE2) ->
    erlang:nif_error("NIF library not loaded").

-spec new_tensor_f32_2d(my_context(), non_neg_integer(), non_neg_integer()) -> my_tensor().
new_tensor_f32_2d(_Ctx, _NE0, _NE1) ->
    erlang:nif_error("NIF library not loaded").

-spec new_tensor_f32_1d(my_context(), non_neg_integer()) -> my_tensor().
new_tensor_f32_1d(_Ctx, _NE) ->
    erlang:nif_error("NIF library not loaded").

-spec tensor_load(my_tensor(), binary()) -> ok.
tensor_load(_T, _Bin) ->
    erlang:nif_error("NIF library not loaded").

-spec tensor_set_f32(my_tensor(), float()) -> my_tensor().
tensor_set_f32(_T, _V) ->
    erlang:nif_error("NIF library not loaded").

-spec grad_set_f32(my_tensor(), float()) -> my_tensor().
grad_set_f32(_T, _V) ->
    erlang:nif_error("NIF library not loaded").

-spec add(my_context(), my_tensor(), my_tensor()) -> my_tensor().
add(_Ctx, _A, _B) ->
    erlang:nif_error("NIF library not loaded").

-spec mul(my_context(), my_tensor(), my_tensor()) -> my_tensor().
mul(_Ctx, _A, _B) ->
    erlang:nif_error("NIF library not loaded").

-spec mul_mat(my_context(), my_tensor(), my_tensor()) -> my_tensor().
mul_mat(_Ctx, _A, _B) ->
    erlang:nif_error("NIF library not loaded").

-spec relu(my_context(), my_tensor()) -> my_tensor().
relu(_Ctx, _T) ->
    erlang:nif_error("NIF library not loaded").

-spec soft_max(my_context(), my_tensor()) -> my_tensor().
soft_max(_Ctx, _T) ->
    erlang:nif_error("NIF library not loaded").

-spec sum(my_context(), my_tensor()) -> my_tensor().
sum(_Ctx, _T) ->
    erlang:nif_error("NIF library not loaded").

-spec graph_compute(my_graph()) -> my_tensor().
graph_compute(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec graph_init_workbuf(my_tensor()) -> my_tensor().
graph_init_workbuf(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec build_forward(my_tensor()) -> my_graph().
build_forward(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec build_backward(my_context(), my_tensor(), boolean()) -> my_graph().
build_backward(_Ctx, _T, _Keep) ->
    erlang:nif_error("NIF library not loaded").

-spec graph_reset(my_graph()) -> ok.
graph_reset(_Graph) ->
    erlang:nif_error("NIF library not loaded").

-spec graph_iter_node(my_graph()) -> {ok,my_tensor()}|{error,not_exists}.
graph_iter_node(_Graph) ->
    erlang:nif_error("NIF library not loaded").

-spec create_compute_params(my_graph(), {non_neg_integer(),non_neg_integer()}) -> my_compute_params().
create_compute_params(_G, {_I,_N}) ->
    erlang:nif_error("NIF library not loaded").

-spec compute_forward(my_compute_params(), my_tensor()) -> ok.
compute_forward(_Params, _Tensor) ->
    erlang:nif_error("NIF library not loaded").

-spec graph_dump_dot(my_tensor(), string()) -> ok.
graph_dump_dot(_T, _F) ->
    erlang:nif_error("NIF library not loaded").

-spec set_param(my_context(), my_tensor()) -> ok.
set_param(_Ctx, _T) ->
    erlang:nif_error("NIF library not loaded").

-spec set_param(my_tensor()) -> ok.
set_param(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec nbytes(my_tensor()) -> non_neg_integer().
nbytes(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec set_name(my_tensor(), string()) -> ok.
set_name(_T, _Name) ->
    erlang:nif_error("NIF library not loaded").

-spec get_data(my_tensor()) -> binary().
get_data(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec get_grad(my_tensor()) -> binary().
get_grad(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec f32_sizef() -> float().
f32_sizef() ->
    erlang:nif_error("NIF library not loaded").

hello(_T) ->
    erlang:nif_error("NIF library not loaded").
