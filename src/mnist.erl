-module(mnist).

-export([model_load/1]).
-export([model_eval/2]).
-export([digit_load/1]).
-export([digit_render/1]).

-type mnist_hparams() :: #{n_input   := non_neg_integer(),
                           n_hidden  := non_neg_integer(),
                           n_classes := non_neg_integer()}.

-type mnist_model() :: #{hparams := mnist_hparams(),
                         fc1_weight := ggml_nif:my_tensor(),
                         fc1_bias   := ggml_nif:my_tensor(),
                         fc2_weight := ggml_nif:my_tensor(),
                         fc2_bias   := ggml_nif:my_tensor(),
                         ctx := ggml_nif:my_context() }.

-define(F32_SZ, 4).

-spec calc_size(mnist_hparams()) -> non_neg_integer().
calc_size(#{n_input := NInput, n_hidden := NHidden, n_classes := NClasses}) ->
    CtxSize = NInput * NHidden + NHidden + NHidden * NClasses + NClasses,
    trunc(CtxSize * ggml_nif:f32_sizef()).

-spec model_load(file:filename()) -> mnist_model().
model_load(FN) ->
    Hparams = #{n_input => 784, n_hidden => 500, n_classes => 10},
    io:format("hparams: ~p~n", [Hparams]),
    {ok, FileBin} = file:read_file(FN),
    Magic = 16#67676d6c,
    <<Magic:32/little-unsigned, RestBin/binary>> = FileBin,

    CtxSize = calc_size(Hparams),
    MemSize = CtxSize + 1024 * 1024,
    Ctx = ggml_nif:init(MemSize),

    <<NDims:32/little-integer, RestBin1/binary>> = RestBin,
    io:format("n_dims: ~p~n", [NDims]),
    <<NInput:32/little-integer, NHidden:32/little-integer, W01:(NInput*NHidden*?F32_SZ)/binary, RestBin2/binary>> = RestBin1,
    io:format("n_input: ~p, n_hidden: ~p, byte_size: ~p~n", [NInput, NHidden, byte_size(W01)]),

%    <<W0:(NInput*NHidden)/binary, RestBin3/binary>> = RestBin2,
    Fc1Weight = ggml_nif:new_tensor_f32_2d(Ctx, NInput, NHidden),
    ggml_nif:tensor_load(Fc1Weight, W01),
    ggml_nif:set_name(Fc1Weight, "fc1_weight"),

    <<NEBias0:32/little-integer, NEBias1:32/little-integer, Bias1:(NEBias0*NEBias1*?F32_SZ)/binary, RestBin3/binary>> = RestBin2,
    Fc1Bias = ggml_nif:new_tensor_f32_1d(Ctx, NHidden),
    ggml_nif:tensor_load(Fc1Bias, Bias1),
    ggml_nif:set_name(Fc1Bias, "fc1_bias"),

    <<_NDims2:32/little-integer, RestBin4/binary>> = RestBin3,
    <<NInput2:32/little-integer, NHidden2:32/little-integer, W12:(NInput2*NHidden2*?F32_SZ)/binary, RestBin5/binary>> = RestBin4,
    Fc2Weight = ggml_nif:new_tensor_f32_2d(Ctx, NHidden, NHidden2),
    ggml_nif:tensor_load(Fc2Weight, W12),
    ggml_nif:set_name(Fc2Weight, "fc2_weight"),

    <<NEBias10:32/little-integer, NEBias11:32/little-integer, Bias2:(NEBias10*NEBias11*?F32_SZ)/binary, _RestBin6/binary>> = RestBin5,
    Fc2Bias = ggml_nif:new_tensor_f32_1d(Ctx, NHidden2), 
    ggml_nif:tensor_load(Fc2Bias, Bias2),
    ggml_nif:set_name(Fc2Bias, "fc2_bias"),

    #{hparams => Hparams, fc1_weight => Fc1Weight, fc1_bias => Fc1Bias,
      n_classes => NHidden2, fc2_weight => Fc2Weight, fc2_bias => Fc2Bias,
      ctx => Ctx}.

model_eval(Model, Digit) ->
    Ctx0 = ggml_nif:init(4 * 784 * 8),
    Input = ggml_nif:new_tensor_f32_1d(Ctx0, 784),
    ggml_nif:tensor_load(Input, Digit),
    ggml_nif:set_name(Input, "input"),

    ggml_nif:set_param(Input),

    Fc1 = ggml_nif:add(Ctx0, ggml_nif:mul_mat(Ctx0, maps:get(fc1_weight, Model), Input), maps:get(fc1_bias, Model)),
    Fc2 = ggml_nif:add(Ctx0, ggml_nif:mul_mat(Ctx0, maps:get(fc2_weight, Model), ggml_nif:relu(Ctx0, Fc1)), maps:get(fc2_bias, Model)),
    Probs = ggml_nif:soft_max(Ctx0, Fc2),
    ggml_nif:set_name(Probs, "probs"),

    ggml_nif:graph_compute(Probs),

    ggml_nif:graph_dump_dot(Probs, "mnist.dot"),

    Data = ggml_nif:get_data(Probs),
    io:format("probs: ~p~n", [[E || <<E:32/float>> <= Data]]),
    argmax(Data).

argmax(<<P:32/float, RestBin/binary>>) ->
    argmax({0,0}, P, RestBin).

argmax({I,_C}, _P, <<>>) ->
    I;
argmax({I,C}, P, <<P1:32/float, RestBin/binary>>) ->
    if P1 > P -> argmax({C+1, C+1}, P1, RestBin);
       true -> argmax({I,C+1}, P, RestBin)
    end.

digit_load(FN) ->
    {ok, FileBin} = file:read_file(FN),
    <<_:16/binary, DigitBin/binary>> = FileBin,
    ByteInput = binary:part(DigitBin, 784 * rand:uniform(1000), 784),
    << <<E:32/native-float>> || <<E:8>> <= ByteInput>>.

digit_render(Digit) ->
    digit_render(Digit, 0, 0).

digit_render(_Digit, 27, 27) ->
    ok;
digit_render(Digit, Row, 27) ->
    io:format("~n"),
    digit_render(Digit, Row + 1, 0);
digit_render(Digit, Row, Col) ->
    <<D:32/native-float>> = binary:part(Digit, 4*(Row * 28 + Col), 4),
    if D > 230 -> io:format("*");
           true -> io:format("_")
    end,
    digit_render(Digit, Row, Col + 1).
