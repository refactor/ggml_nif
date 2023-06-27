#include <erl_nif.h>
#include <string.h>

#include "common.h"

#include "ggml.h"
#include "my_ggml.h"
#include "mylog.h"

struct my_context {
    struct ggml_context* ctx;
};
struct my_tensor {
    struct my_context* myctx;
    struct ggml_tensor* tensor;
};
struct my_cgraph {
    struct my_context* workctx;
    struct my_tensor* output;
    struct ggml_cgraph cgraph;
    int current_node;
};

static ErlNifResourceType* GGML_CONTEXT_RESOURCE_TYPE;
static ErlNifResourceType* GGML_TENSOR_RESOURCE_TYPE;
static ErlNifResourceType* GGML_CGRAPH_RESOURCE_TYPE;
static ErlNifResourceType* GGML_CPARAMS_RESOURCE_TYPE;

ERL_NIF_TERM OK;
ERL_NIF_TERM ERROR;
ERL_NIF_TERM NOT_EXISTS;
ERL_NIF_TERM TRUE;
ERL_NIF_TERM FALSE;

static void context_dtor(ErlNifEnv* env, void* obj) {
    struct my_context* myctx = (struct my_context*)obj;
    DBG("FREE ggml_context......%p", myctx);
    ggml_free(myctx->ctx);
}
static void tensor_dtor(ErlNifEnv* env, void* obj) {
    struct my_tensor* mytensor = (struct my_tensor*)obj;
    DBG("free ggml_tensor(%s: %p), in myctx(%p)......", mytensor->tensor->name, mytensor->tensor, mytensor->myctx);
    if (mytensor->myctx) enif_release_resource(mytensor->myctx);
}
static void cgraph_dtor(ErlNifEnv* env, void* obj) {
    struct my_cgraph* cgraph = (struct my_cgraph*)obj;
    DBG("free cgraph...");
    enif_release_resource(cgraph->output);
    if (cgraph->workctx) enif_release_resource(cgraph->workctx);
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info) {
    DBG("loading......");
    OK = enif_make_atom(env, "ok");
    ERROR = enif_make_atom(env, "error");
    NOT_EXISTS = enif_make_atom(env, "not_exists");
    TRUE = enif_make_atom(env, "true");
    FALSE = enif_make_atom(env, "false");

    GGML_CONTEXT_RESOURCE_TYPE =  enif_open_resource_type(env, "ggml", "ggml_context", context_dtor, ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER, NULL);
    if (GGML_CONTEXT_RESOURCE_TYPE == NULL) return -1;
    GGML_TENSOR_RESOURCE_TYPE =  enif_open_resource_type(env, "ggml", "ggml_tensor", tensor_dtor, ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER, NULL);
    if (GGML_TENSOR_RESOURCE_TYPE == NULL) return -1;
    GGML_CGRAPH_RESOURCE_TYPE =  enif_open_resource_type(env, "ggml", "ggml_cgraph", cgraph_dtor, ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER, NULL);
    GGML_CPARAMS_RESOURCE_TYPE =  enif_open_resource_type(env, "ggml", "ggml_compute_params", NULL, ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER, NULL);
    if (GGML_CGRAPH_RESOURCE_TYPE == NULL) return -1;
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

static ERL_NIF_TERM new_tensor_(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], enum ggml_type type) {
    struct my_context* myctx;
    if (!enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }

    int64_t ne[argc - 1];
    for (int i = 0; i < argc - 1; ++i) {
        if (!enif_get_int64(env, argv[i+1], &ne[i])) {
            return enif_make_badarg(env);
        }
    }
    DBG("new_tensor_f32_<%d>d......myctx: %p", argc-1, myctx);
    struct ggml_tensor* t = NULL;
    switch (argc) {
        case 2:
            t = ggml_new_tensor_1d(myctx->ctx, type, ne[0]);
            break;
        case 3:
            t = ggml_new_tensor_2d(myctx->ctx, type, ne[0], ne[1]);
            break;
        case 4:
            t = ggml_new_tensor_3d(myctx->ctx, type, ne[0], ne[1], ne[2]);
            break;
        default:
            return enif_make_badarg(env);
    }

    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor){
        .myctx = myctx,
        .tensor = t
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM new_tensor_f32_1d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) return enif_make_badarg(env);
    return new_tensor_(env, argc, argv, GGML_TYPE_F32);
}

static ERL_NIF_TERM new_tensor_f32_2d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 3) return enif_make_badarg(env);
    return new_tensor_(env, argc, argv, GGML_TYPE_F32);
}

static ERL_NIF_TERM new_tensor_f32_3d(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 4) return enif_make_badarg(env);
    return new_tensor_(env, argc, argv, GGML_TYPE_F32);
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
        .myctx = myctx,
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
        .myctx = myctx,
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
        .myctx = myctx,
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
        .myctx = myctx,
        .tensor = result
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM sum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (argc != 2 || !enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensora;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* result = ggml_sum(myctx->ctx, mytensora->tensor);
    struct my_tensor* mytensor = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mytensor));
    *mytensor = (struct my_tensor) {
        .myctx = myctx,
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
        .myctx = myctx,
        .tensor = result
    };
    enif_keep_resource(myctx);
    ERL_NIF_TERM etensor = enif_make_resource(env, mytensor);
    enif_release_resource(mytensor);
    return etensor;
}

static ERL_NIF_TERM build_forward(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensorf;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorf)) {
        return enif_make_badarg(env);
    }

    ErlDrvSysInfo sysinfo;
    enif_system_info(&sysinfo, sizeof(sysinfo));
    DBG("enif_system_info: scheduler_threads=%d", sysinfo.scheduler_threads);
    struct my_cgraph* mygraph = enif_alloc_resource(GGML_CGRAPH_RESOURCE_TYPE, sizeof(*mygraph));
    *mygraph = (struct my_cgraph) {
        .output = mytensorf,
        .current_node = 0,
        .cgraph = {
            .n_threads = sysinfo.scheduler_threads
        }
    };
    ggml_build_forward_expand(&mygraph->cgraph, mytensorf->tensor);
    DBG(" for cgraph: scheduler_threads=%d", mygraph->cgraph.n_threads);
    enif_keep_resource(mytensorf);
    ERL_NIF_TERM cg = enif_make_resource(env, mygraph);
    enif_release_resource(mygraph);
    return cg;
}

static ERL_NIF_TERM build_backward(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_context* myctx;
    if (!enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
        return enif_make_badarg(env);
    }
    struct my_cgraph* mygraph;
    if (!enif_get_resource(env, argv[1], GGML_CGRAPH_RESOURCE_TYPE, (void**)&mygraph)) {
        return enif_make_badarg(env);
    }

    bool keep = enif_is_identical(TRUE, argv[2]); 

    struct my_cgraph* mygb = enif_alloc_resource(GGML_CGRAPH_RESOURCE_TYPE, sizeof(*mygraph));
    *mygb = (struct my_cgraph) {
        .current_node = 0,
        .workctx = NULL,
        .output = NULL,
    };
    mygb->cgraph = ggml_build_backward(myctx->ctx, &mygraph->cgraph, keep);
    enif_keep_resource(mygb);
    ERL_NIF_TERM cg = enif_make_resource(env, mygb);
    enif_release_resource(mygb);
    return cg;
}

static ERL_NIF_TERM graph_reset(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_cgraph* mygraph;
    if (!enif_get_resource(env, argv[0], GGML_CGRAPH_RESOURCE_TYPE, (void**)&mygraph)) {
        return enif_make_badarg(env);
    }
    ggml_graph_reset(&mygraph->cgraph);
    return OK;
}


static ERL_NIF_TERM graph_init_workbuf(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_cgraph* mygraph;
    if (!enif_get_resource(env, argv[0], GGML_CGRAPH_RESOURCE_TYPE, (void**)&mygraph)) {
        return enif_make_badarg(env);
    }

    struct ggml_cgraph* cgraph = &mygraph->cgraph;
    my_init_task_and_workbuf(cgraph);
    if (cgraph->work_size > 0 && cgraph->work == NULL) {
        if (mygraph->workctx == NULL) {
            struct ggml_context* ctx = ggml_init((struct ggml_init_params){.mem_size=1024 + cgraph->work_size});
            struct my_context* myctx = enif_alloc_resource(GGML_CONTEXT_RESOURCE_TYPE, sizeof(*myctx));
            *myctx = (struct my_context) {
                .ctx = ctx
            };
            enif_make_resource(env, myctx);
            enif_release_resource(myctx);

            mygraph->workctx = myctx;
        }
        struct ggml_context* ctx = mygraph->workctx->ctx;
        cgraph->work = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, cgraph->work_size);
    }
    return argv[0];
}

static ERL_NIF_TERM graph_iter_node(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_cgraph* mygraph;
    if (!enif_get_resource(env, argv[0], GGML_CGRAPH_RESOURCE_TYPE, (void**)&mygraph)) {
        return enif_make_badarg(env);
    }
    if (mygraph->current_node < mygraph->cgraph.n_nodes) {
        struct ggml_tensor * node = mygraph->cgraph.nodes[mygraph->current_node];
        mygraph->current_node++;

        struct my_tensor* mynode = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mynode));
        *mynode = (struct my_tensor) {
            .tensor = node,
            .myctx = NULL
        };
        ERL_NIF_TERM etensor = enif_make_resource(env, mynode);
        enif_release_resource(mynode);
        return enif_make_tuple2(env, OK, etensor);
    }

    mygraph->current_node = 0;
    return enif_make_tuple2(env, ERROR, NOT_EXISTS);
}

static ERL_NIF_TERM node_compute_params(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* node;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&node)) {
        return enif_make_badarg(env);
    }
    int n_tasks = node->tensor->n_tasks;
    DBG("node->n_tasks: %d", node->tensor->n_tasks);
    ERL_NIF_TERM arr[n_tasks];
    ERL_NIF_TERM NTASKS = enif_make_int(env, n_tasks);
    for (int i = 0; i < n_tasks; ++i) {
        arr[i] = enif_make_tuple2(env, enif_make_int(env, i), NTASKS);
    }
    return enif_make_list_from_array(env, arr, n_tasks);
}

static ERL_NIF_TERM create_compute_params(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_cgraph* mygraph;
    if (!enif_get_resource(env, argv[0], GGML_CGRAPH_RESOURCE_TYPE, (void**)&mygraph)) {
        return enif_make_badarg(env);
    }
    const ERL_NIF_TERM* tinfo = NULL;
    int len;
    if (!enif_get_tuple(env, argv[1], &len, &tinfo) || len != 2) {
        return enif_make_badarg(env);
    }
    int ith = 0;
    int nth = 0;
    if (!enif_get_int(env, tinfo[0], &ith) || !enif_get_int(env, tinfo[1], &nth)) {
        return enif_make_badarg(env);
    }
    struct ggml_cgraph* cgraph = &mygraph->cgraph;
    struct ggml_compute_params* compute_params = enif_alloc_resource(GGML_CPARAMS_RESOURCE_TYPE, sizeof(*compute_params));
    *compute_params = (struct ggml_compute_params) {
        .type = GGML_TASK_COMPUTE,
        .ith = ith,
        .nth = nth,
        .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
        .wdata = cgraph->work ? cgraph->work->data : NULL,
    };
    ERL_NIF_TERM params = enif_make_resource(env, compute_params);
    enif_release_resource(compute_params);
    return params;
}

static ERL_NIF_TERM compute_forward(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct ggml_compute_params* params;
    if (!enif_get_resource(env, argv[0], GGML_CPARAMS_RESOURCE_TYPE, (void**)&params)) {
        return enif_make_badarg(env);
    }
    struct my_tensor* mytensorf;
    if (!enif_get_resource(env, argv[1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensorf)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* tensor = mytensorf->tensor;
    DBG("params: %d, %d, tensor: %p", params->ith, params->nth, tensor);
    ggml_compute_forward(params, mytensorf->tensor);
    return OK;
}

static ERL_NIF_TERM graph_compute(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_cgraph* mygraph;
    if (!enif_get_resource(env, argv[0], GGML_CGRAPH_RESOURCE_TYPE, (void**)&mygraph)) {
        return enif_make_badarg(env);
    }

    struct ggml_cgraph* cgraph = &mygraph->cgraph;
    ggml_graph_compute(NULL, &mygraph->cgraph);

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
    return OK;
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
    return OK;
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

static ERL_NIF_TERM grad_set_f32(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensor;
    if (!enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensor)) {
        return enif_make_badarg(env);
    }
    double value;
    if (!enif_get_double(env, argv[1], &value)) {
        return enif_make_badarg(env);
    }
    ggml_set_f32(mytensor->tensor->grad, (float)value);
    return OK;
}

static ERL_NIF_TERM set_param(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 && argc != 2) {
        enif_fprintf(stderr, "wrong argc: %d", argc);
        return enif_make_badarg(env);
    }
    if (!enif_get_resource(env, argv[argc - 1], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct my_context* myctx = mytensora->myctx;
    if (argc == 2) {
        if (!enif_get_resource(env, argv[0], GGML_CONTEXT_RESOURCE_TYPE, (void**)&myctx)) {
            return enif_make_badarg(env);
        }
    }
    struct ggml_tensor* tensor = mytensora->tensor;
    ggml_set_param(myctx->ctx, tensor);
    return OK;
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

static ERL_NIF_TERM get_grad(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    struct ggml_tensor* tensor = mytensora->tensor;
    struct my_tensor* mygrad = enif_alloc_resource(GGML_TENSOR_RESOURCE_TYPE, sizeof(*mygrad));
    *mygrad = (struct my_tensor) {
        .tensor = tensor->grad,
        .myctx = NULL
    };
    ERL_NIF_TERM etensor = enif_make_resource(env, mygrad);
    enif_release_resource(mygrad);
    return etensor;
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
    DBG("name: %s", name);
    struct ggml_tensor* tensor = mytensora->tensor;
    ggml_set_name(tensor, name);
    return OK;
}

static ERL_NIF_TERM f32_sizef(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return enif_make_double(env, ggml_type_sizef(GGML_TYPE_F32));
}
static ERL_NIF_TERM hello(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct my_tensor* mytensora;
    if (argc != 1 || !enif_get_resource(env, argv[0], GGML_TENSOR_RESOURCE_TYPE, (void**)&mytensora)) {
        return enif_make_badarg(env);
    }
    ggml_print_objects(mytensora->myctx->ctx);
    struct ggml_tensor* tensor = mytensora->tensor;
    for (int i = 0; i < tensor->n_dims; ++i) {
        DBG("data(%p)[%d]: [", tensor->data, tensor->nb[0]);
        for (int j = 0; j < tensor->ne[i]; ++j) {
            DBG("%f, ", *(float*)(tensor->data + j * tensor->nb[i]));
        }
        DBG("]");
    }
    return OK;
}

static ErlNifFunc nif_funcs[] = {
    {"init", 1, init, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"new_tensor_f32_1d", 2, new_tensor_f32_1d},
    {"new_tensor_f32_2d", 3, new_tensor_f32_2d},
    {"new_tensor_f32_3d", 4, new_tensor_f32_3d},
    {"tensor_set_f32", 2, tensor_set_f32},
    {"grad_set_f32", 2, grad_set_f32},
    {"tensor_load", 2, tensor_load},
    {"add", 3, add},
    {"mul", 3, mul},
    {"mul_mat", 3, mul_mat},
    {"relu", 2, relu},
    {"sum", 2, sum},
    {"soft_max", 2, soft_max},
    {"set_param", 1, set_param},
    {"set_param", 2, set_param},
    {"nbytes", 1, nbytes},
    {"set_name", 2, set_name},
    {"get_data", 1, get_data},
    {"get_grad", 1, get_grad},
    {"build_forward", 1, build_forward},
    {"build_backward", 3, build_backward},
    {"graph_reset", 1, graph_reset},
    {"graph_init_workbuf", 1, graph_init_workbuf},
    {"graph_iter_node", 1, graph_iter_node},
    {"node_compute_params", 1, node_compute_params},
    {"create_compute_params", 2, create_compute_params},
    {"compute_forward", 2, compute_forward, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"graph_compute", 1, graph_compute, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"graph_dump_dot", 2, graph_dump_dot},
    {"f32_sizef", 0, f32_sizef},
    {"hello", 1, hello}
};

ERL_NIF_INIT(ggml_nif,nif_funcs,load,NULL,NULL,NULL)
