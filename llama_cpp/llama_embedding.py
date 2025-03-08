from __future__ import annotations

import os
import ctypes
import pathlib

from typing import (
    Callable,
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    byref,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp._ctypes_extensions import (
        CtypesCData,
        CtypesArray,
        CtypesPointer,
        CtypesVoidPointer,
        CtypesRef,
        CtypesPointerOrRef,
        CtypesFuncPointer,
    )
from llama_cpp.llama_cpp import llama_context_p_ctypes,llama_context_p
from llama_cpp.llama_cpp import llama_batch

# Specify the base name of the shared library to load
_lib_base_name = "llama-embedding"
_override_base_path = os.environ.get("LLAMA_CPP_LIB_PATH")
_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _override_base_path is None else pathlib.Path(_override_base_path)
# Load the library
_lib = load_shared_library(_lib_base_name, _base_path)

ctypes_function = ctypes_function_for_shared_library(_lib)

# // llama_batch_decode
@ctypes_function(
    "llama_batch_decode", [llama_context_p_ctypes, llama_batch, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)], ctypes.c_bool
)
def llama_batch_decode(
    ctx: llama_context_p, 
    batch: llama_batch, 
    n_seq: ctypes.c_int32, 
    n_embd:ctypes.c_int32, 
    n_norm:ctypes.c_int32, 
    output:CtypesArray[ctypes.c_float], /) -> ctypes.c_bool:
    """llama_batch_decode
    """
    ...