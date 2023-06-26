-module(demo).
-compile(export_all).


exec() ->
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

    <<GV:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X)),

    ggml_nif:tensor_load(X, <<3.0:32/native-float>>),
    ggml_nif:graph_reset(GF),
    ggml_nif:grad_set_f32(F, 1.0),
%    ggml_nif:graph_compute(GB),
%    <<FV1:32/native-float>> = ggml_nif:get_data(F),
%    <<GV1:32/native-float>> = ggml_nif:get_data(ggml_nif:get_grad(X)),
    ok.
