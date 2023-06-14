-module(ggml_nif).

-export([load/0]).
-export([hello/1]).
-export([init/1]).
-export([new_tensor_f32_2d/3]).
-export([new_tensor_f32_1d/2]).
-export([tensor_set_f32/2]).
-export([add/3]).
-export([mul/3]).
-export([mul_mat/3]).
-export([relu/2]).
-export([soft_max/2]).
-export([graph_compute/1]).
-export([graph_dump_dot/2]).
-export([set_param/1]).
-export([nbytes/1]).
-export([set_name/2]).

-export_type([my_context/0, my_tensor/0]).

-nifs([hello/1]).
-nifs([new_tensor_f32_2d/3]).
-nifs([new_tensor_f32_1d/2]).
-nifs([tensor_set_f32/2]).
-nifs([add/3]).
-nifs([mul/3]).
-nifs([mul_mat/3]).
-nifs([relu/2]).
-nifs([soft_max/2]).
-nifs([graph_compute/1]).
-nifs([graph_dump_dot/2]).
-nifs([set_param/1]).
-nifs([nbytes/1]).
-nifs([set_name/2]).
-nifs([init/1]).

-opaque my_context() :: reference().
-opaque my_tensor() :: reference().

-on_load(load/0).

load() ->
    erlang:load_nif("priv/ggml_enif", 0).

-spec init(integer()) -> my_context().
init(_ByteSize) ->
    erlang:nif_error("NIF library not loaded").

-spec new_tensor_f32_2d(my_context(), non_neg_integer(), non_neg_integer()) -> my_tensor().
new_tensor_f32_2d(_Ctx, _NE0, _NE1) ->
    erlang:nif_error("NIF library not loaded").

-spec new_tensor_f32_1d(my_context(), non_neg_integer()) -> my_tensor().
new_tensor_f32_1d(_Ctx, _NE) ->
    erlang:nif_error("NIF library not loaded").

-spec tensor_set_f32(my_tensor(), float()) -> my_tensor().
tensor_set_f32(_T, _V) ->
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

-spec graph_compute(my_tensor()) -> my_tensor().
graph_compute(_F) ->
    erlang:nif_error("NIF library not loaded").

-spec graph_dump_dot(my_tensor(), string()) -> ok.
graph_dump_dot(_T, _F) ->
    erlang:nif_error("NIF library not loaded").

-spec set_param(my_tensor()) -> ok.
set_param(_F) ->
    erlang:nif_error("NIF library not loaded").

-spec nbytes(my_tensor()) -> non_neg_integer().
nbytes(_T) ->
    erlang:nif_error("NIF library not loaded").

-spec set_name(my_tensor(), string()) -> ok.
set_name(_T, _Name) ->
    erlang:nif_error("NIF library not loaded").

hello(_T) ->
    erlang:nif_error("NIF library not loaded").
