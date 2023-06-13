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

-on_load(load/0).

load() ->
    erlang:load_nif("priv/ggml_enif", 0).

init(_ByteSize) ->
    erlang:nif_error("NIF library not loaded").

new_tensor_f32_2d(_Ctx, _NE0, _NE1) ->
    erlang:nif_error("NIF library not loaded").

new_tensor_f32_1d(_Ctx, _NE) ->
    erlang:nif_error("NIF library not loaded").

tensor_set_f32(_T, _V) ->
    erlang:nif_error("NIF library not loaded").

add(_Ctx, _A, _B) ->
    erlang:nif_error("NIF library not loaded").

mul(_Ctx, _A, _B) ->
    erlang:nif_error("NIF library not loaded").

mul_mat(_Ctx, _A, _B) ->
    erlang:nif_error("NIF library not loaded").

relu(_Ctx, _T) ->
    erlang:nif_error("NIF library not loaded").

soft_max(_Ctx, _T) ->
    erlang:nif_error("NIF library not loaded").

graph_compute(_F) ->
    erlang:nif_error("NIF library not loaded").

graph_dump_dot(_T, _F) ->
    erlang:nif_error("NIF library not loaded").

set_param(_F) ->
    erlang:nif_error("NIF library not loaded").

nbytes(_T) ->
    erlang:nif_error("NIF library not loaded").

set_name(_T, _Name) ->
    erlang:nif_error("NIF library not loaded").

hello(_T) ->
    erlang:nif_error("NIF library not loaded").
