-module(cgraph_computer_tests).
-include_lib("eunit/include/eunit.hrl").

do_compute_test() ->
    NInput = 100,
    NHidden = 40,
    Ctx = ggml_nif:init(3 * NInput*NHidden * 4 + 1024),
    V1 = ggml_nif:new_tensor_f32_1d(Ctx, NInput),
    ggml_nif:tensor_set_f32(V1, 1.0),
    Weight = ggml_nif:new_tensor_f32_2d(Ctx, NInput, NHidden),
    ggml_nif:tensor_set_f32(Weight, 1.0),
    V = ggml_nif:mul_mat(Ctx, Weight, V1),

    {ok, Pid} = cgraph_computer:start_link(V),
    Bin = cgraph_computer:do_compute(Pid),
    ?assertNot(is_process_alive(Pid)),

    ?assertEqual(40 * 4, size(Bin)),
    Res = [X || <<X:32/native-float>> <= Bin],
    ?assertEqual(40, length(Res)),
    ?assertMatch([X | _] when X == 100.0, Res).
    %ggml_nif:graph_build(V).

