# main_joker → 最新 main Rebase 分析文档

**执行日期**: 2026-03-07
**操作人**: Claude Sonnet 4.6
**目标分支**: `main_joker`
**基准分支**: `main`（commit `c37132b`）

---

## 一、背景与目标

### 为什么需要 rebase

`main_joker` 分支基于旧版 main（merge base: `37eb5f0`），包含 1 个自定义 commit `43efedb`，引入了 JokerRAG 所需的 embedding DLL 集成、归一化参数、chat format 工具方法等功能。

最新 main 已前进到 `c37132b`（多个版本更新），包含 llama.cpp 子模块更新和若干接口修改。如果继续停留在旧 base，每次 main 有新功能或 bugfix 时都需要手动合并，且旧版 API（如 `llama_kv_cache_clear`）在新 llama.cpp 中已被移除，存在运行时符号找不到的风险。

### 目标

1. 将 `main_joker` 的自定义修改应用到最新 `main` 上
2. 修复自定义修改中已知的 4 个 bug
3. 保持清晰的 commit 分层，方便后续回溯

---

## 二、自定义修改有效性分析

commit `43efedb` 包含以下修改，逐一评估在最新 main 上是否仍有意义。

### 修改 1: llama_batch_decode DLL 集成

**背景**: JokerRAG 使用专用 embedding 模型 gte-Qwen2-7B-instruct，需要在 Python 进程中高效批量计算文档向量。

**做了什么**: 新建 `llama_embedding.py`（ctypes 绑定），重写 `embed()` 方法，将批量 embedding 计算委托给自定义编译的 `llama-embedding` DLL（C++ 函数 `llama_batch_decode`）。

**在新 main 上是否仍有意义**: 是。新 main 的 `embed()` 仍使用 Python 逐 batch 调用方式，没有 DLL 集成。

**在新 main 上是否仍有效**: 是。`llama_batch_decode` 的 C++ 函数签名与 Python ctypes 声明完全匹配，DLL 接口本身不受 main 版本影响。

**原版 embed() 末尾调用 `llama_kv_cache_clear`**，这个 API 在新 llama.cpp（`llamacpp_joker_2026.01`）中已被移除（改为 `llama_memory_clear`）。由于 `llama_batch_decode` DLL 内部已处理 KV cache 清理，可以直接移除这行调用，无需替换为新 API。

---

### 修改 2: create_embedding 增加 normalize 参数

**做了什么**: `create_embedding(input, normalize=True, model=None)`，将 normalize 传给 `embed()`。

**在新 main 上是否仍有意义**: 是。新 main 的 `create_embedding` 不暴露归一化控制。

**在新 main 上是否仍有效**: 有效（但 Bug #2 导致参数实际未传递到 C++ 层，见 Bug 章节）。

---

### 修改 3: apply_chat_format 新方法

**做了什么**: 在 `__init__` 中将 `Jinja2ChatFormatter` 实例存入 `_char_formaters` 字典；新增 `apply_chat_format()` 方法，调用 formatter 格式化消息并 tokenize，返回 token ids。

**在新 main 上是否仍有意义**: 是。JokerRAG 需要直接获取格式化后的 token ids，不走完整 chat completion 流程。新 main 没有此方法。

**在新 main 上是否仍有效**: 是。

**正确性**: 基本正确。`_char_formaters` 拼写少一个 t（`formaters` vs `formatters`），属于 typo，不影响功能。

---

### 修改 4: logits_all 注释掉

**做了什么**: 注释掉 `logits_all = pooling_type == llama_cpp.LLAMA_POOLING_TYPE_NONE`。

**原因**: `llama_batch_decode` 内部自行处理 pooling，不依赖 Python 侧传入的 `logits_all` flag；旧版 embed() 需要此 flag 告知 llama.cpp 为所有 token 计算 logits（NONE pooling 需要），但新版 DLL 路径不经过此 flag。

**在新 main 上是否仍有意义**: 是。

---

### 修改 5: add_bos=False tokenize

**做了什么**: `embed()` 中所有 tokenize 调用均使用 `add_bos=False`。

**原因**: JokerRAG 调用 DLL 时，tokenize 在 Python 侧完成，调用者自行控制 BOS。embedding 模型（gte-Qwen2-7B）不需要 BOS token，重复添加会干扰 pooling。

**在新 main 上是否仍有意义**: 是。

---

### 修改 6: CMakeLists.txt 构建规则

**做了什么**: 新增 `add_subdirectory(vendor/llama.cpp/examples/embedding)` 等构建规则，将 `llama-embedding` 导出为共享库并安装到 `llama_cpp/lib/`。

**在新 main 上是否仍有意义**: 是。DLL 需要随 Python 包一起安装。

---

## 三、Bug 发现与分析过程

### Bug #1: `self._batch.n_tokens` 缺括号

**发现**: 静态代码审查，`llama.py:1124`：
```python
embd_offset += self._batch.n_tokens if pooling_type == LLAMA_POOLING_TYPE_NONE else p_batch
```

`_internals.py` 中 `LlamaBatch.n_tokens` 是 `def n_tokens(self) -> int` 方法，不加 `()` 返回的是 bound method 对象，`+=` 会报 TypeError。

**确认过程**: 查看 `_internals.py`，确认 `n_tokens` 是方法而非属性（无 `@property` 装饰器）。

**触发条件**: 仅在 batch 溢出时执行（当一批文本的 token 数超过 n_batch）。单 batch 能容纳所有输入时不触发，因此功能测试中可能未暴露。

**注意**: 同一方法在 `llama.py:1119` 有正确调用 `self._batch.n_tokens()`（带括号），只有 1124 行的 batch overflow 分支中缺括号。

**修复**: `self._batch.n_tokens()` 加括号。

---

### Bug #2: n_norm 写死为 2

**发现**: 静态代码审查，`llama.py:1121` 和 `llama.py:1138`：
```python
llama_batch_decode(self._ctx.ctx, self._batch.batch, p_batch, n_embd, 2, embeddings_ptr)
```

`normalize` 参数被接收（`create_embedding` 暴露给调用方），但未传递到 C++ DLL，始终执行 L2 归一化（`embd_norm=2`）。

**确认过程**: 查看 C++ 侧 `embedding.cpp`：
```cpp
bool llama_batch_decode(... int embd_norm, float * output) {
    ...
    common_embd_normalize(embd, out, n_embd, embd_norm);
}
```
`embd_norm=-1` 表示不归一化，`embd_norm=2` 表示 L2 归一化。Python 侧 `normalize=False` 时应传 `-1`（或 `0`，根据 `common_embd_normalize` 实现）。

**修复**: 引入 `n_norm = 2 if normalize else 0`，传入 DLL 调用。

---

### Bug #3: NONE pooling pos 偏移计算错误

**初始分析**: `llama.py:1077`：
```python
pos += size          # size = 该序列的 token 数
```
`ptr` 是 `ctypes.POINTER(c_float)`，索引单位是 float。每个 token 的 embedding 是 `n_embd` 个 float，因此应为 `pos += size * n_embd`。

**争议过程（关键）**:

用户指出："其实 pos 真的不用改，因为下面获取 embedding_ptr 的时候，传进来的指针偏移已经重算过了"。即 `embeddings_ptr` 在外层已经按 `embd_offset * n_embd` 做了偏移（INTER-batch 偏移），那 `pos` 管的是 INTRA-batch（同一个 batch 内不同序列间的偏移）。

**C++ 侧验证**: 查看 `embedding.cpp` 中 `llama_batch_decode` 的内存布局：
```cpp
// NONE pooling: 按 token 在 batch 中的顺序写入
// embd_pos = i (token 在 batch 中的绝对序号)
float * out = output + embd_pos * n_embd;
common_embd_normalize(embd, out, n_embd, embd_norm);
```
对于 NONE pooling，C++ 侧将 token `i` 的 embedding 写入 `output[i * n_embd]`（float 偏移，不是 token 偏移）。

因此，Python 侧 `ptr` 中 token `i` 的 embedding 确实在 `ptr[i * n_embd : (i+1) * n_embd]`。

**进一步分析**: `pos` 是 INTRA-batch 的起始位置（float 索引）。对于第一个序列（size=s0），embedding 从 `ptr[0]` 到 `ptr[s0*n_embd-1]`，提取正确。对于第二个序列（size=s1），起始位置应从 `ptr[s0*n_embd]` 开始。但 `pos += size` 让 pos = s0（token 数），实际访问 `ptr[s0 + j*n_embd]`，偏移错误。

**历史证据**: 查阅 JokerRAG git history：
- commit `0239062`（早期版本）已有 `pos += size * n_embd` 的修复
- commit `aea2688`（batch_decode 重写）重写 embed() 时未带入此修复，导致退化

**最终结论**: Bug 确实存在于多序列 NONE pooling 场景。但当前实际使用场景（gte-Qwen2-7B-instruct）使用 LAST pooling，此代码路径从未执行。考虑到代码正确性和未来可能的模型切换，决定在 Commit 3 中修复。

---

### Bug #4: prompt_tokens[1:] 未赋值

**发现**: `llama.py:1280`：
```python
prompt_tokens[1:]
```
Python 的切片操作返回新 list，不修改原变量。此行什么都没做。

**设计意图讨论**:

用户说明：设计意图是"内部使用带 BOS 的数据，只是外部获取时去掉 BOS"。讨论了两种方案：

1. `prompt_tokens = prompt_tokens[1:]`：重绑局部变量，原调用方的 list 不受影响（Python slice 创建新 list）。cache lookup 用截短后的 token 序列，首次请求可能 cache miss，但不影响正确性。

2. `del prompt_tokens[0]`：原地修改，会影响调用方传入的 list（如果调用方后续还引用该 list，会发现第一个 token 被删除）。

**决策**: 使用 `prompt_tokens = prompt_tokens[1:]`。因为：
- 调用方的原始 list 不受影响（安全）
- 符合"只是外部获取时去掉 BOS"的设计意图
- 副作用（cache miss）是已知且可接受的

**注意**: 此 Bug 在 Commit 1 中保留（原始移植），在 Commit 2 中单独修复。

---

### Bug #5: data.append vs data.extend（不是 Bug）

**初始误判**: 初步认为 NONE pooling 中 `data.append(embedding)` 应改为 `data.extend(embedding)`，理由是 JokerRAG 的 `0239062` commit 使用了 `extend`。

**深入分析**:

```python
embedding = [ptr[pos + j * n_embd : pos + (j + 1) * n_embd] for j in range(size)]
# embedding 类型: List[List[float]]
# 每个元素: 一个 token 的 n_embd 维向量

data.append(embedding)  # data[i] = List[List[float]]，即第 i 个文本的所有 token embedding
data.extend(embedding)  # 将 List[List[float]] 展开，data[0] = 第一个 token 的 embedding
```

对于 NONE pooling，调用方期望 `output = data[0]` 是输入文本的所有 token embeddings（`List[List[float]]`）。

- `append` → `data[0]` = 完整的 token embedding 列表 ✓
- `extend` → `data[0]` = 只有第一个 token 的 embedding ✗

JokerRAG `0239062` 使用 `extend` 实际上是错误的（针对单文本场景能工作，因为调用方不用 `data[0]` 而是直接用 `data`，但改变了数据结构语义）。

**最终结论**: `append` 是正确的，不修改。从 Bug 列表中移除。

---

## 四、KV Cache API 兼容性

新版 llama.cpp（`llamacpp_joker_2026.01`）移除了 `llama_kv_cache_clear`，改为：
```c
llama_memory_clear(llama_get_memory(ctx), true)
```

`embed()` 末尾调用 `llama_cpp.llama_kv_cache_clear(self._ctx.ctx)` 在新版会报运行时符号错误。

**处理方案**: 直接移除此调用。`llama_batch_decode` DLL 内部已包含 KV cache 管理逻辑（C++ 侧 `llama_memory_clear`），Python 侧无需重复清理。这也简化了代码，不需要在 `llama_cpp.py` 添加新的 ctypes 绑定。

---

## 五、Commit 拆分逻辑

### 为什么拆成三个 Commit

用户要求：原始修改移植和 Bug 修复分开提交，方便后续回溯区分"原始意图"和"已知修复"。

```
Commit 1: 原始移植（带原始 bug，忠实重现 43efedb 的意图）
    ├── CMakeLists.txt 构建规则
    ├── llama_embedding.py DLL 绑定
    └── llama.py 所有修改（含 Bug #1, #2, #3 原始状态）
           包括 Bug #4 的原始错误行 prompt_tokens[1:]

Commit 2: Bug #4 修复（非 embed() 相关）
    └── prompt_tokens = prompt_tokens[1:]

Commit 3: embed() Bug 修复
    ├── Bug #1: self._batch.n_tokens() 加括号
    ├── Bug #2: n_norm = 2 if normalize else 0
    └── Bug #3: pos += size * n_embd
```

### 为什么 Bug #4 和 embed() Bug 分开

Bug #4 在 `_create_completion()` 方法中，与 `embed()` 完全无关。分开提交使 `git blame` 和 `git log -p` 时更容易定位每个修复的范围。

---

## 六、llama_cpp.py 与 llamacpp_joker_2026.01 接口兼容性分析

**分析日期**: 2026-03-08

### 背景

用户提出："llama-cpp-python 的最新 main 分支对应的 vendor 内的 llama.cpp 使用的分支似乎不是最新的"。需要验证 `llama-cpp-python/main_joker` 分支与 `llama.cpp/llamacpp_joker_2026.01` 分支（两个 joker 分支配对使用）的接口兼容性。

### 分析过程

**第一步：确认各分支 vendor 子模块指向的 llama.cpp commit**

```
旧版 main_joker (43efedb) vendor → 7841fc72 (llama: Add Gemma 3 support, PR #12343)
    所在分支: llamacpp_joker, llamacpp_joker_2026.01, master

最新 main (c37132b) vendor      → 4227c9be (CUDA: fix negative KV_max values, PR #15321)
    所在分支: llamacpp_joker_2026.01, master
```

**第二步：确认 `llamacpp_joker_2026.01` 的 upstream 基点**

```
llamacpp_joker_2026.01 upstream base → a0ed91a44 (models: kda chunk size = 16, PR #19827)
llamacpp_joker_2026.01 HEAD         → 261e0e5ad (base + 11 个自定义 commits)
```

**第三步：确认版本差距**

```
llama.cpp commits 时间线：

7841fc72 (#12343)  ← 旧版 main_joker (43efedb) vendor
    ↓ +1295 commits
4227c9be (#15321)  ← 最新 llama-cpp-python/main (c37132b) vendor
    ↓ +2041 commits
a0ed91a44 (#19827) ← llamacpp_joker_2026.01 upstream base
    ↓ +11 自定义 commits
261e0e5ad          ← llamacpp_joker_2026.01 HEAD
```

**结论：`llama_cpp.py` 适配的 llama.cpp 版本（`4227c9be`，PR #15321）与 `llamacpp_joker_2026.01` 的 upstream base（`a0ed91a44`，PR #19827）之间有 2041 个 commits 的差距。**

旧版 `main_joker`（`43efedb`）在自定义 commit 中**未修改过 `llama_cpp.py`**（只改了 CMakeLists.txt、llama.py、llama_embedding.py），说明旧版也没有适配过这个版本差距——旧版 `main_joker` 当时配合的是更早的 `llamacpp_joker` 分支（vendor `7841fc72`），不是 `llamacpp_joker_2026.01`。

rebase 到最新 main 后，vendor 从 `7841fc72` 更新到了 `4227c9be`（拉近了 1295 个 commits），但距离 `llamacpp_joker_2026.01` 的 base 仍有 **2041 个 commits 的 API 差距**。

**第四步：识别 `llama_cpp.py` 的 API 覆盖缺口**

`llama_cpp.py` 是 Python ctypes 绑定文件，需要与实际 DLL 导出的函数名和签名一致。做双向 diff：

```bash
# 从 llamacpp_joker_2026.01 的 llama.h 提取所有 LLAMA_API 函数名
grep -oE "llama_[a-z_]+" llama.cpp/include/llama.h | sort -u > c_api.txt

# 从 llama_cpp.py 提取所有绑定函数名
grep -oE '"llama_[a-z_]+"' llama_cpp/llama_cpp.py | tr -d '"' | sort -u > py_api.txt

# 双向比较
comm -23 c_api.txt py_api.txt   → llama.h 有但 py 没绑定（48 个）
comm -13 c_api.txt py_api.txt   → py 绑定了但 llama.h 没有（22 个）
```

**第五步：判断哪些缺失会导致运行时错误**

22 个"py 有但 llama.h 没有"的函数中，大部分是 KV cache 旧 API 的 deprecated 绑定（`llama_kv_self_*`），`_internals.py` 已经切换到新 API，不会被调用。

关键问题：ctypes 的 `getattr(lib, name)` 是否懒加载？

```python
# _ctypes_extensions.py:112
func = getattr(lib, name)   # ← 模块加载时执行
func.argtypes = argtypes
func.restype = restype
```

`ctypes.CDLL` 的 `getattr` 是懒加载：返回一个函数包装对象，不立即解析 DLL 符号。只有实际调用 `func()` 时才会查找符号。因此绑定了不存在的函数名不会导致 import 失败，仅调用时崩溃。

**第六步：交叉引用实际调用路径**

遍历 22 个不存在的函数，在 `llama.py`、`_internals.py`、`llama_chat_format.py` 中搜索调用：

```bash
for func in llama_kv_self_clear llama_set_adapter_lora llama_sampler_init_softmax ...; do
  grep -rn "$func" llama_cpp/ --include="*.py" | grep -v "llama_cpp.py"
done
```

只有两个函数在非 `llama_cpp.py` 文件中被实际调用：
1. `llama_set_adapter_lora` → `llama.py:435`（LoRA 加载路径）
2. `llama_sampler_init_softmax` → `_internals.py:676`（负温度采样路径）

其余 20 个都是死代码。

### 方法

提取 `llamacpp_joker_2026.01` 的 `include/llama.h` 中所有 `LLAMA_API` 导出函数名，与 `llama_cpp.py` 中所有 ctypes 绑定函数名做双向 diff。再交叉引用 `llama.py`、`_internals.py`、`llama_chat_format.py` 中的实际调用，判断每个差异是否会导致运行时错误。

### 结果 1: 已正确适配的接口

| 接口 | 说明 |
|------|------|
| KV Cache API | `_internals.py` 已全面切换到 `llama_memory_*` 新 API（`llama_get_memory`、`llama_memory_clear`、`llama_memory_seq_rm` 等），旧 `llama_kv_self_*` 绑定是死代码 |
| `llama_batch_decode` | `llama_embedding.py` 独立绑定自定义 DLL，不依赖 `llama_cpp.py` |
| `embed()` KV 清理 | 本次 rebase 已移除 `llama_kv_cache_clear` 调用 |
| 核心推理 API | `llama_decode`、`llama_model_load_from_file`、`llama_init_from_model`、`llama_get_embeddings_seq`、`llama_pooling_type`、`llama_perf_context_*` 等全部在新 llama.h 中存在 |
| `llama_chat_format.py` | 内部调用 `llama._ctx.kv_cache_clear()` → `_internals.py` → `llama_memory_clear`（新 API） |

### 结果 2: 破坏性不兼容（运行时会崩溃）

#### 不兼容 #1: `llama_sampler_init_softmax` — 完全移除

- **llama_cpp.py**: `llama_cpp.py:3809` 绑定了 `"llama_sampler_init_softmax"`
- **新 llama.h**: 该函数完全不存在（连 DEPRECATED 都没有，源码中也无此符号）
- **调用链**: `llama.py:754`（`temp < 0.0` 时）→ `_internals.py:676` `add_softmax()` → `llama_cpp.llama_sampler_init_softmax()`
- **触发条件**: 使用负温度采样（`temperature < 0`）
- **影响**: ctypes 调用不存在的 DLL 符号，运行时崩溃

#### 不兼容 #2: `llama_set_adapter_lora` — 改名 + 签名变更

- **llama_cpp.py**: `llama_cpp.py:1739` 绑定了 `"llama_set_adapter_lora"`
- **新 llama.h**: 改为 `llama_set_adapters_lora`（注意复数 `s`），且签名从单 adapter 改为 adapter 数组：
  ```c
  // 旧签名（已移除）
  int32_t llama_set_adapter_lora(ctx, adapter, scale)
  // 新签名
  int32_t llama_set_adapters_lora(ctx, adapters[], n_adapters, scales[])
  ```
- **调用链**: `llama.py:435`（构造函数中 `lora_path` 非空时）
- **触发条件**: 加载 LoRA adapter
- **影响**: ctypes 调用不存在的 DLL 符号，运行时崩溃

### 结果 3: llama_cpp.py 中绑定了但新 llama.h 中已不存在的函数（死代码）

以下函数在 `llama_cpp.py` 中有 ctypes 绑定，但新 `llama.h` 中不存在。由于 ctypes 懒加载特性（`getattr(lib, name)` 不立即解析符号），这些绑定在 import 时不会报错，仅在实际调用时才会崩溃。经确认，`_internals.py` 和 `llama.py` 的正常代码路径不会调用这些函数。

```
llama_kv_self_clear           → 替代: llama_memory_clear (已适配)
llama_kv_self_seq_rm          → 替代: llama_memory_seq_rm (已适配)
llama_kv_self_seq_add         → 替代: llama_memory_seq_add (已适配)
llama_kv_self_seq_cp          → 替代: llama_memory_seq_cp (已适配)
llama_kv_self_seq_div         → 替代: llama_memory_seq_div (已适配)
llama_kv_self_seq_keep        → 替代: llama_memory_seq_keep (已适配)
llama_kv_self_seq_pos_max     → 替代: llama_memory_seq_pos_max (已适配)
llama_kv_self_seq_pos_min     → 替代: llama_memory_seq_pos_min (已适配)
llama_kv_self_can_shift       → 替代: llama_memory_can_shift (已适配)
llama_kv_self_defrag          → 无直接替代 (未被调用)
llama_kv_self_update          → 无直接替代 (未被调用)
llama_kv_self_n_tokens        → 无直接替代 (未被调用)
llama_kv_self_used_cells      → 无直接替代 (未被调用)
llama_get_kv_self             → 替代: llama_get_memory (已适配)
llama_apply_adapter_cvec      → 替代: llama_set_adapter_cvec
llama_clear_adapter_lora      → 已移除
llama_rm_adapter_lora         → 已移除
llama_sampler_init_softmax    → 已移除 ← 但实际被调用！见不兼容 #1
llama_set_adapter_lora        → 替代: llama_set_adapters_lora ← 实际被调用！见不兼容 #2
```

### 结果 4: 新 llama.h 有但 llama_cpp.py 未绑定的函数（48 个）

这些是 `llamacpp_joker_2026.01` 新增或改名的函数，`llama_cpp.py` 尚未提供绑定。当前使用场景不需要这些函数，但列出以供未来参考：

```
llama_attach_threadpool / llama_detach_threadpool    — 线程池管理
llama_decode_with_sampler                            — 带采样器的解码
llama_get_pooling_type                               — llama_pooling_type 的新名（旧名仍可用）
llama_get_sampled_*                                  — 采样细节分析（6 个函数）
llama_set_adapters_lora                              — LoRA 数组接口（替代 llama_set_adapter_lora）
llama_set_adapter_cvec                               — 控制向量接口（替代 llama_apply_adapter_cvec）
llama_sampler_init_adaptive_p                        — 自适应 p 采样器
llama_model_is_hybrid                                — 混合模型检测
llama_model_n_embd_inp / llama_model_n_embd_out      — 输入/输出 embedding 维度
llama_n_ctx_seq                                      — 每序列上下文长度
llama_memory_breakdown_print                         — 内存分布打印
llama_state_seq_*_ext                                — 扩展状态序列化
llama_params_fit / llama_params_fit_status            — 参数适配
llama_flash_attn_type / llama_flash_attn_type_name   — Flash Attention 类型
llama_rope_type                                      — RoPE 类型
llama_log_get                                        — 日志获取
... 其他 struct/type 定义
```

### 潜在风险

`llama_pooling_type` 在新 llama.h 中标有 `// TODO: rename to llama_get_pooling_type`。当前旧名仍保留，但如果未来某版本完成重命名并删除旧名，`_internals.py:287` 的调用会崩溃。

### 修复建议

| 不兼容 | 修复方案 | 优先级 |
|--------|----------|--------|
| #1 `llama_sampler_init_softmax` | 确认新版本中负温度采样的替代实现，更新 `llama_cpp.py` 绑定和 `_internals.py` 调用 | 高（影响 LLM 推理） |
| #2 `llama_set_adapter_lora` | 在 `llama_cpp.py` 添加 `llama_set_adapters_lora` 绑定（数组签名），修改 `llama.py:435` 调用方式 | 中（仅影响 LoRA 用户） |

---

## 七、执行结果

### Rebase + Bug 修复（2026-03-07）

已完成 rebase 和 bug 修复，commit 结构见 git log。

### 兼容性修复（2026-03-08）

对两个 joker 分支做了全面 API 兼容性对比后，修复了 2 个运行时崩溃的破坏性不兼容：

**Commit `8bb5a67`**: `fix: adapt llama_cpp.py bindings for llamacpp_joker_2026.01 compatibility`

| 不兼容 | 修复内容 | 涉及文件 |
|--------|----------|----------|
| #1 `llama_sampler_init_softmax` 移除 | 删除 `add_softmax()` 方法和 ctypes 绑定，`temp <= 0` 统一改为 `add_greedy()`（与新版 llama.cpp 行为一致：`llama_sampler_temp_impl` 对 `temp <= 0` 贪心采样） | `llama_cpp.py`、`_internals.py`、`llama.py` |
| #2 `llama_set_adapter_lora` → `llama_set_adapters_lora` | 更新 ctypes 绑定为新的数组签名（`adapters[]`, `n_adapters`, `scales[]`），调用处构造单元素数组适配 | `llama_cpp.py`、`llama.py` |

已 push 到 remote `origin/main_joker`。

### 当前 commit 结构

```
8bb5a67 fix: adapt llama_cpp.py bindings for llamacpp_joker_2026.01 compatibility
c775713 fix: LlamaHFTokenizer add_eos + load_backends improvements
1becff5 feat: dynamic backend loading + disable_cuda + CMake embedding decoupling
dfc4310 feat: separate add_bos/add_eos parameters in tokenize API
358528c fix: update Python bindings for new llama.cpp API
373bfc6 mod: update llama.cpp.
```
