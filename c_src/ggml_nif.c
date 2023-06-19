#include <erl_nif.h>
#include <string.h>

#define PRINT(...) enif_fprintf(stdout, __VA_ARGS__) 

#include "ggml.h"

struct my_context {
    struct ggml_context* ctx;
};
struct my_tensor {
    struct my_context* ctx;
    struct ggml_tensor* tensor;
};

static ErlNifResourceType* GGML_CONTEXT_RESOURCE_TYPE;
static ErlNifResourceType* GGML_TENSOR_RESOURCE_TYPE;

static void context_dtor(ErlNifEnv* env, void* obj) {
    struct my_context* myctx = (struct my_context*)obj;
    enif_fprintf(stdout, "free ggml_context......\n");
    ggml_free(myctx->ctx);
}
static void tensor_dtor(ErlNifEnv* env, void* obj) {
    struct my_tensor* mytensor = (struct my_tensor*)obj;
    enif_fprintf(stdout, "free ggml_tensor(%s)......\n", mytensor->tensor->name);
    enif_release_resource(mytensor->ctx);
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info) {
    enif_fprintf(stdout, "loading......\n");
    GGML_CONTEXT_RESOURCE_TYPE =  enif_open_resource_type(env, "ggml", "ggml_context", context_dtor, ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER, NULL);
    if (GGML_CONTEXT_RESOURCE_TYPE == NULL) return -1;
    GGML_TENSOR_RESOURCE_TYPE =  enif_open_resource_type(env, "ggml", "ggml_tensor", tensor_dtor, ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER, NULL);
    if (GGML_TENSOR_RESOURCE_TYPE == NULL) return -1;
    return 0;
}

static ERL_NIF_TERM init(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    int64_t sz;
    if (argc != 1 || !enif_get_int64(env, argv[0], &sz)) {
        return enif_make_badarg(env);
    }
    struct ggml_init_params params = {
        .mem_size = sz,
        .mem_buffer = NULL,
    };
    struct ggml_context* ctx = ggml_init(params);
    struct my_context* myctx = enif_alloc_resource(GGML_CONTEXT_RESOURCE_TYPE, sizeof(*myctx));
    *myctx = (struct my_context) {
        .ctx = ctx
    };
    ERL_NIF_TERM gtx = enif_make_resource(env, myctx);
    enif_release_resource(myctx);

    return gtx;
}

static ERL_NIF_TERM new_tensor_1d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], enum ggml_type type) {
    struct my_context* myctx;
    if (argc != 2 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    int64_t ne0;
    if (!enif_get_int64(env, argv[1], &ne0)) {
        return enif_make_badarg(env);
    }
    enif_fprintf(stdout, "new_tensor_f32_1d......myctx: %p\n", myctx);
    struct ggml_tensor* t = ggml_new_tensor_1d(myctx->ctx, type, ne0);

    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor){
        .ctx = myctx,
        .tensor = t
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM new_tensor_f32_1d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return new_tensor_1d(env, argc, argv, GGML_TYPE_F32);
}

static ERL_NIF_TERM new_tensor_2d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], enum ggml_type type) {
    struct my_context* myctx;
    if (argc != 3 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    int64_t ne0;
    if (!enif_get_int64(env, argv[1], &ne0)) {
        return enif_make_badarg(env);
    }
    int64_t ne1;
    if (!enif_get_int64(env, argv[2], &ne1)) {
        return enif_make_badarg(env);
    }
    enif_fprintf(stdout, "new_tensor_f32_2d......myctx: %p\n", myctx);
    struct ggml_tensor* t = ggml_new_tensor_2d(myctx->ctx, type, ne0, ne1);

    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor){
        .ctx = myctx,
        .tensor = t
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM new_tensor_f32_2d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return new_tensor_2d(env, argc, argv, GGML_TYPE_F32);
}

static ERL_NIF_TERM add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (argc != 3 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensora;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensorb;
    if (!enif_get_resource(env, argv[2], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorb)) {
        return enif_make_badarg(env);
    }
    
    struct ggml_tensor* m = ggml_add(myctx->ctx, mytensora->tensor, mytensorb->tensor);
    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor){
        .ctx = myctx,
        .tensor = m
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM mul(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (argc != 3 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensora;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensorb;
    if (!enif_get_resource(env, argv[2], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorb)) {
        return enif_make_badarg(env);
    }
    
    struct ggml_tensor* m = ggml_mul(myctx->ctx, mytensora->tensor, mytensorb->tensor);
    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor){
        .ctx = myctx,
        .tensor = m
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM mul_mat(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (argc != 3 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensora;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensorb;
    if (!enif_get_resource(env, argv[2], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorb)) {
        return enif_make_badarg(env);
    }
    
    struct ggml_tensor* m = ggml_mul_mat(myctx->ctx, mytensora->tensor, mytensorb->tensor);
    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor){
        .ctx = myctx,
        .tensor = m
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM relu(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (argc != 2 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensora;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* result = ggml_relu(myctx->ctx, mytensora->tensor);
    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor) {
        .ctx = myctx,
        .tensor = result
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM soft_max(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (argc != 2 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensora;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* result = ggml_soft_max(myctx->ctx, mytensora->tensor);
    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor) {
        .ctx = myctx,
        .tensor = result
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM graph_compute(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensorf;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorf)) {
        return enif_make_badarg(env);
    }

    struct ggml_cgraph gf = ggml_build_forward(mytensorf->tensor);
    ggml_graph_compute(mytensorf->ctx->ctx, &gf);
    return argv[0];
}

static ERL_NIF_TERM graph_dump_dot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensorf;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorf)) {
        return enif_make_badarg(env);
    }
    struct ggml_cgraph gf = ggml_build_forward(mytensorf->tensor);
    char filename[128] = {};
    if (!enif_get_string(env, argv[1], filename, sizeof(filename), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }
    ggml_graph_dump_dot(&gf, NULL, filename);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM tensor_load(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensor;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensor)) {
        return enif_make_badarg(env);
    }
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, argv[1], &bin)) {
        return enif_make_badarg(env);
    }

    struct ggml_tensor* tensor = mytensor->tensor;
    memcpy(tensor->data, bin.data, bin.size);
    return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM tensor_set_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensor;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensor)) {
        return enif_make_badarg(env);
    }
    double value;
    if (!enif_get_double(env, argv[1], &value)) {
        return enif_make_badarg(env);
    }

    struct ggml_tensor* tensor = mytensor->tensor;
    ggml_set_f32(tensor, (float)value);
    return argv[0];
}

static ERL_NIF_TERM set_param(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* tensor = mytensora->tensor;
    ggml_set_param(mytensora->ctx->ctx, tensor);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_data(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* tensor = mytensora->tensor;
    void* data = ggml_get_data(tensor);
    ErlNifBinary bin = {
        .size = ggml_nbytes(tensor),
        .data = data
    };
    return enif_make_binary(env, &bin);
}

static ERL_NIF_TERM nbytes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* tensor = mytensora->tensor;
    size_t sz = ggml_nbytes(tensor);
    return enif_make_ulong(env, sz);
}

static ERL_NIF_TERM set_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 2 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    char name[GGML_MAX_NAME] = {};
    if (!enif_get_string(env, argv[1], name, sizeof(name), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }
    enif_fprintf(stdout, "name: %s\r\n", name);
    struct ggml_tensor* tensor = mytensora->tensor;
    ggml_set_name(tensor, name);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM f32_sizef(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return enif_make_double(env, ggml_type_sizef(GGML_TYPE_F32));
}
static ERL_NIF_TERM hello(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    ggml_print_objects(mytensora->ctx->ctx);
    struct ggml_tensor* tensor = mytensora->tensor;
    for (int i = 0; i < tensor->n_dims; ++i) {
        enif_fprintf(stdout, "data(%p)[%d]: [", tensor->data, tensor->nb[0]);
        for (int j = 0; j < tensor->ne[i]; ++j) {
            enif_fprintf(stdout, "%f, ", *(float*)(tensor->data + j * tensor->nb[i]));
        }
        enif_fprintf(stdout, "]\r\n");
    }
    return enif_make_atom(env, "ok");
}

static ErlNifFunc nif_funcs[] = {
    {"init", 1, init, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"new_tensor_f32_1d", 2, new_tensor_f32_1d},
    {"new_tensor_f32_2d", 3, new_tensor_f32_2d},
    {"tensor_set_f32", 2, tensor_set_f32},
    {"tensor_load", 2, tensor_load},
    {"add", 3, add},
    {"mul", 3, mul},
    {"mul_mat", 3, mul_mat},
    {"relu", 2, relu},
    {"soft_max", 2, soft_max},
    {"set_param", 1, set_param},
    {"nbytes", 1, nbytes},
    {"set_name", 2, set_name},
    {"get_data", 1, get_data},
    {"graph_compute", 1, graph_compute, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"graph_dump_dot", 2, graph_dump_dot},
    {"f32_sizef", 0, f32_sizef},
    {"hello", 1, hello}
};

ERL_NIF_INIT(ggml_nif,nif_funcs,load,NULL,NULL,NULL)
