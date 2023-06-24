-module(adder).
-export([do_compute/3]).
-export([do_compute/2]).
-export([build_adder/2, build_adder/3]).
-export([build_mul/2]).

-spec do_compute(float(),float()) -> binary().
do_compute(A, B) ->
    do_compute(1_000_000, A, B).

-spec do_compute(non_neg_integer(),float(),float()) -> binary().
do_compute(L, A, B) ->
    Ctx = ggml_nif:init(3 * L * 4 + 1024),
    V1 = ggml_nif:new_tensor_f32_1d(Ctx, L),
    ggml_nif:tensor_set_f32(V1, A),
    V2 = ggml_nif:new_tensor_f32_1d(Ctx, L),
    ggml_nif:tensor_set_f32(V2, B),
    V = ggml_nif:add(Ctx, V1, V2),
    ggml_nif:graph_compute(V),
    Result = ggml_nif:get_data(V),
    Result.

build_adder(A, B) ->
    build_adder(1_000_000, A, B).

build_adder(L, A, B) ->
    Ctx = ggml_nif:init(3 * L * 4 + 1024),
    V1 = ggml_nif:new_tensor_f32_1d(Ctx, L),
    ggml_nif:tensor_set_f32(V1, A),
    V2 = ggml_nif:new_tensor_f32_1d(Ctx, L),
    ggml_nif:tensor_set_f32(V2, B),
    V = ggml_nif:add(Ctx, V1, V2),
    ggml_nif:graph_build(V).

build_mul(A, B) ->
    NInput = 100,
    NHidden = 40,
    Ctx = ggml_nif:init(3 * NInput*NHidden * 4 + 1024),
    V1 = ggml_nif:new_tensor_f32_1d(Ctx, NInput),
    ggml_nif:tensor_set_f32(V1, A),
    Weight = ggml_nif:new_tensor_f32_2d(Ctx, NInput, NHidden),
    ggml_nif:tensor_set_f32(Weight, B),
    V = ggml_nif:mul_mat(Ctx, Weight, V1),

    {ok, Pid} = cgraph_computer:start_link(V),
    Bin = cgraph_computer:do_compute(Pid),
    Bin.
    %ggml_nif:graph_build(V).
