-module(simple1_tests).
-include_lib("eunit/include/eunit.hrl").

simple1_1_test() ->
    Ctx = ggml_nif:init(128 * 1024 * 1024),
    X = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    ggml_nif:set_param(Ctx, X),

    A = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    B = ggml_nif:mul(Ctx, X, X),
    F = ggml_nif:mul(Ctx, B, A),

    GF = ggml_nif:build_forward(F),
    GB = ggml_nif:build_backward(Ctx, GF, false),

    ggml_nif:tensor_set_f32(X, 2.0),
    ggml_nif:tensor_set_f32(A, 3.0),

    ggml_nif:graph_reset(GF),
    ggml_nif:grad_set_f32(F, 1.0),

    ggml_nif:graph_compute(GB),

    <<FV:32/native-float>> = ggml_nif:get_data(F),
    ?assertEqual(12.0, FV),

    <<GV:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X)),
    ?assertEqual(12.0, GV),

    ggml_nif:tensor_load(X, <<3.0:32/native-float>>),
    ggml_nif:graph_reset(GF),
    ggml_nif:grad_set_f32(F, 1.0),
    ggml_nif:graph_compute(GB),
    <<FV1:32/native-float>> = ggml_nif:get_data(F),
    ?assertEqual(27.0, FV1),
    <<GV1:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X)),
    ?assertEqual(18.0, GV1).

simple1_2_test() ->
    Ctx = ggml_nif:init(128 * 1024 * 1024),
    X1 = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    X2 = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    X3 = ggml_nif:new_tensor_f32_1d(Ctx, 1),

    ggml_nif:tensor_load(X1, <<3.0:32/native-float>>),
    ggml_nif:tensor_load(X2, <<1.0:32/native-float>>),
    ggml_nif:tensor_load(X3, <<0.0:32/native-float>>),

    ggml_nif:set_param(Ctx, X1),
    ggml_nif:set_param(Ctx, X2),

    Y = ggml_nif:add(Ctx, ggml_nif:mul(Ctx, X1,X1), ggml_nif:mul(Ctx, X1,X2)),

    GF = ggml_nif:build_forward(Y),
    GB = ggml_nif:build_backward(Ctx, GF, false),

    ggml_nif:graph_reset(GF),
    ggml_nif:grad_set_f32(Y, 1.0),

    ggml_nif:graph_compute(GB),

    <<Y0:32/native-float>> = ggml_nif:get_data(Y),
    <<XG1:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X1)),
    <<XG2:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X2)),
    ?debugFmt("Y      = ~p", [Y0]),
    ?debugFmt("df/dx1 = ~p", [XG1]),
    ?debugFmt("df/dx2 = ~p", [XG2]),
    ?assertEqual(12.0, Y0),
    ?assertEqual(7.0, XG1),
    ?assertEqual(3.0, XG2),

    G1 = ggml_nif:get_grad(X1),
    G2 = ggml_nif:get_grad(X2),

    GBB = ggml_nif:build_backward(Ctx, GB, true),

    ggml_nif:graph_reset(GB),
    ggml_nif:tensor_load(ggml_nif:get_grad(G1), <<1.0:32/native-float>>),
    ggml_nif:tensor_load(ggml_nif:get_grad(G2), <<1.0:32/native-float>>),

    ggml_nif:graph_compute(GBB),
    <<X1grad:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X1)),
    <<X2grad:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X2)),
    ?debugFmt("H * [1, 1] = [ ~p ~p ]", [X1grad, X2grad]),
    ?assertEqual(3.0, X1grad),
    ?assertEqual(1.0, X2grad).

simple1_3_test() ->
    Ctx = ggml_nif:init(128 * 1024 * 1024),
    X1 = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    X2 = ggml_nif:new_tensor_f32_1d(Ctx, 1),

    ggml_nif:set_param(Ctx, X1),
    ggml_nif:set_param(Ctx, X2),

    Y = ggml_nif:mul(Ctx, ggml_nif:add(Ctx, ggml_nif:mul(Ctx, X1,X1), ggml_nif:mul(Ctx, X1, X2)), X1),

    GF = ggml_nif:build_forward(Y),
    GB = ggml_nif:build_backward(Ctx, GF, false),

    ggml_nif:tensor_load(X1, <<3.0:32/native-float>>),
    ggml_nif:tensor_load(X2, <<4.0:32/native-float>>),

    ggml_nif:graph_reset(GF),
    ggml_nif:tensor_load(ggml_nif:get_grad(Y), <<1.0:32/native-float>>),

    ggml_nif:graph_compute(GB),

    <<Y0:32/native-float>> = ggml_nif:get_data(Y),
    <<XG1:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X1)),
    <<XG2:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X2)),
    ?debugFmt("Y      = ~p", [Y0]),
    ?debugFmt("df/dx1 = ~p", [XG1]),
    ?debugFmt("df/dx2 = ~p", [XG2]),
    ?assertEqual(63.0, Y0),
    ?assertEqual(51.0, XG1),
    ?assertEqual(9.0, XG2).
    
simple1_4_test() ->
    Ctx = ggml_nif:init(128 * 1024 * 1024),
    X1 = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    X2 = ggml_nif:new_tensor_f32_1d(Ctx, 1),
    X3 = ggml_nif:new_tensor_f32_1d(Ctx, 1),

    ggml_nif:set_param(Ctx, X1),
    ggml_nif:set_param(Ctx, X2),
    ggml_nif:set_param(Ctx, X3),

    Y = ggml_nif:mul(Ctx, ggml_nif:mul(Ctx, ggml_nif:mul(Ctx, X1,X1), ggml_nif:mul(Ctx, X2,X2)), X3),

    GF = ggml_nif:build_forward(Y),
    GB = ggml_nif:build_backward(Ctx, GF, false),

    ggml_nif:tensor_load(X1, <<1.0:32/native-float>>),
    ggml_nif:tensor_load(X2, <<2.0:32/native-float>>),
    ggml_nif:tensor_load(X3, <<3.0:32/native-float>>),

    ggml_nif:graph_reset(GF),
    ggml_nif:tensor_load(ggml_nif:get_grad(Y), <<1.0:32/native-float>>),

    ggml_nif:graph_compute(GB),

    <<Y0:32/native-float>> = ggml_nif:get_data(Y),
    <<XG1:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X1)),
    <<XG2:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X2)),
    <<XG3:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X3)),
    ?debugFmt("Y      = ~p", [Y0]),
    ?debugFmt("df/dx1 = ~p", [XG1]),
    ?debugFmt("df/dx2 = ~p", [XG2]),
    ?debugFmt("df/dx3 = ~p", [XG3]),
    ?assertEqual(12.0, Y0),
    ?assertEqual(24.0, XG1),
    ?assertEqual(12.0, XG2),
    ?assertEqual(4.0, XG3),

    G1 = ggml_nif:get_grad(X1),
    G2 = ggml_nif:get_grad(X2),
    G3 = ggml_nif:get_grad(X3),
    
    GBB = ggml_nif:build_backward(Ctx, GB, true),
    ggml_nif:graph_reset(GB),
    ggml_nif:tensor_load(ggml_nif:get_grad(G1), <<1.0:32/native-float>>),
    ggml_nif:tensor_load(ggml_nif:get_grad(G2), <<1.0:32/native-float>>),
    ggml_nif:tensor_load(ggml_nif:get_grad(G3), <<1.0:32/native-float>>),

    ggml_nif:graph_compute(GBB),
    <<XGrad1:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X1)),
    <<XGrad2:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X2)),
    <<XGrad3:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X3)),
    ?debugFmt("H * [1, 1, 1] = [ ~p ~p ~p ]", [XGrad1, XGrad2, XGrad3]),
    ?assertEqual(56.0, XGrad1),
    ?assertEqual(34.0, XGrad2),
    ?assertEqual(12.0, XGrad3).

simple1_5_test() ->
    Ctx = ggml_nif:init(128 * 1024 * 1024),
    X1 = ggml_nif:new_tensor_f32_1d(Ctx, 3),
    X2 = ggml_nif:new_tensor_f32_1d(Ctx, 3),

    ggml_nif:set_param(Ctx, X1),
    ggml_nif:set_param(Ctx, X2),

    Y = ggml_nif:sum(Ctx, ggml_nif:mul(Ctx, X1,X2)),

    GF = ggml_nif:build_forward(Y),
    GB = ggml_nif:build_backward(Ctx, GF, false),

    ggml_nif:tensor_set_f32(X1, 3.0),
    ggml_nif:tensor_set_f32(X2, 5.0),

    ggml_nif:graph_reset(GF),
    ggml_nif:tensor_set_f32(ggml_nif:get_grad(Y), 1.0),

    ggml_nif:graph_compute(GB),

    Y0 = [E || <<E:32/native-float>> <= ggml_nif:get_data(Y)],
    XG1 = [E || <<E:32/native-float>> <= ggml_nif:get_data(ggml_nif:get_grad(X1))],
    XG2 = [E || <<E:32/native-float>> <= ggml_nif:get_data(ggml_nif:get_grad(X2))],
    ?debugFmt("Y      = ~p", [Y0]),
    ?debugFmt("df/dx1 = ~p", [XG1]),
    ?debugFmt("df/dx2 = ~p", [XG2]),
    ?assertMatch([E | _] when E == 45.0, Y0),
    ?assertMatch([E | _] when E == 5.0, XG1),
    ?assertMatch([E | _] when E == 3.0, XG2).
