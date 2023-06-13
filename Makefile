PROJECT = ggml_enif
PROJECT_DESCRIPTION = New project
PROJECT_VERSION = 0.1.0

LDFLAGS+=-lm
CFLAGS+=-mavx2 -mfma

include erlang.mk
