-module(model_predictor_tests).
-include_lib("eunit/include/eunit.hrl").

do_compute_test() ->
    NInput = 784,
    NHidden = 500,
    Ctx = ggml_nif:init(3 * NInput*NHidden * 4 + 1024),
    Input = ggml_nif:new_tensor_f32_1d(Ctx, NInput),
    ggml_nif:tensor_set_f32(Input, 1.0),
    Weight1 = ggml_nif:new_tensor_f32_2d(Ctx, NInput, NHidden),
    ggml_nif:tensor_set_f32(Weight1, 1.0),
    Bias1 = ggml_nif:new_tensor_f32_1d(Ctx, NHidden),
    ggml_nif:tensor_set_f32(Bias1, 0.0),
    V = ggml_nif:add(Ctx, ggml_nif:mul_mat(Ctx, Weight1, Input), Bias1),

    {ok, Pid} = model_predictor:start_link({Input,V}),
    Bin = model_predictor:do_compute(Pid),
    model_predictor:stop(Pid),
    ?assertNot(is_process_alive(Pid)),

    ?assertEqual(500 * 4, size(Bin)),
    Res = [X || <<X:32/native-float>> <= Bin],
    ?assertEqual(500, length(Res)),
    ?assertMatch([X | _] when X == 784.0, Res).
    %ggml_nif:graph_build(V).

