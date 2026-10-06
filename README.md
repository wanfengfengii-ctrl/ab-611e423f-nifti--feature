# nifti-sampler

神经影像质控平台的体数据抽查服务：按扫描仪 RAS 世界坐标对三维 NIfTI-1
体数据做三线性插值采样。严格校验方向矩阵（sform/qform）、字节序与强度
缩放，避免核对到错误体素。纯 Python 标准库实现，无第三方依赖。

## API

### `POST /api/nifti/sample`

`multipart/form-data`，包含：

| 部分 | 说明 |
| --- | --- |
| 文件部分（任意字段名，须带 `filename`） | 单个 NIfTI-1 `.nii` 文件，≤ 16 MiB |
| `points` 字段 | JSON 数组，1–256 个元素：`[{"id": 0, "point": [x, y, z]}, ...]` |
| `derivatives` 字段（可选） | 目前仅接受字面量 `world_gradient`；省略时请求、响应与错误完全不变 |

`id` 为 `[0, 2^31)` 内唯一整数；`point` 为三个有限数值（RAS 世界坐标，
单位与文件仿射一致，通常为 mm）。

```bash
curl -F "file=@vol.nii" \
     -F 'points=[{"id": 1, "point": [12.0, 26.0, 42.0]}]' \
     http://localhost:8000/api/nifti/sample

# 需要世界方向梯度时加上 derivatives 字段
curl -F "file=@vol.nii" \
     -F 'points=[{"id": 1, "point": [12.0, 26.0, 42.0]}]' \
     -F "derivatives=world_gradient" \
     http://localhost:8000/api/nifti/sample
```

**成功响应（200）**，结果顺序与请求一致：

```json
{
  "transform": "sform",
  "results": [
    {"id": 1, "status": "ok", "voxel": [1.0, 2.0, 3.0],
     "intensity": 647.0, "transform": "sform",
     "gradient": [0.5, 3.33, 25.0]},
    {"id": 2, "status": "error",
     "error": {"code": "out_of_bounds", "message": "...",
               "voxel": [-5.0, -6.67, -7.5]}}
  ]
}
```

每个成功点给出：连续体素坐标（`voxel`）、插值强度（`intensity`，已应用
`scl_slope`/`scl_inter` 缩放）、所用变换来源（`transform`，
`sform` 或 `qform`）。越界点、非有限数据以**逐点错误**返回，不影响其余点。

`derivatives=world_gradient` 时，每个成功点额外给出 `gradient`：按
**R、A、S 顺序**排列的有限梯度三元组，表示该点局部三线性强度场相对
*世界坐标*（而非体素轴）的变化率。计算方式为：先对所选三线性单元做体素轴
有限差分（已计入 `scl_slope` 强度缩放），再经所选 sform/qform 仿射的逆
转置映射到世界方向，`g_world = M^{-T} g_voxel`，因此对角仿射下分量即
“缩放后的体素梯度 ÷ 体素尺寸”，旋转仿射下分量会混合轴别，不会把体素轴
梯度误报成 RAS 方向。单元选择规则：

- 点恰落在某个体素分界（体素坐标为整数的内部节点）时取**负体素侧**单元
  `[i-1, i]`（在末节点即 `[n-2, n-1]`）；
- 零侧边界（体素坐标 0）没有负侧单元，改取**正侧**单元 `[0, 1]`；
- 该轴仅一个体素（`n=1`）时，对应梯度分量为 0；
- 梯度所需 2×2×2 邻域内任一缩放后数据非有限时，**仅该点**返回可定位的
  逐点错误 `non_finite_gradient`（携带连续体素坐标与问题体素下标），其余
  点照常给出强度、连续体素坐标、变换来源与梯度。

### 错误

文件/请求级错误返回 4xx，JSON 形如
`{"error": {"code", "message", "field"}}`，`field` 定位到头部字段或表单字段：

| code | status | field | 含义 |
| --- | --- | --- | --- |
| `invalid_multipart` | 400/415 | — | 表单结构非法 |
| `missing_file` / `multiple_files` | 400 | `file` | 文件部分缺失/多于一个 |
| `missing_points` / `invalid_points` | 400 | `points` | 坐标字段缺失、数量越界、id 重复/非法、坐标非有限（message 含点号） |
| `invalid_derivatives` | 400 | `derivatives` | `derivatives` 取值不支持（仅支持 `world_gradient`） |
| `file_too_large` | 413 | `file` | 超过 16 MiB |
| `header_too_short` | 400 | `file` | 不足 348 字节头部 |
| `bad_sizeof_hdr` | 400 | `sizeof_hdr` | 两种字节序下都不是 348 |
| `unsupported_magic` | 400 | `magic` | 非 `n+1`（如 `.hdr/.img` 对的 `ni1`） |
| `invalid_dimensions` | 400 | `dim` | 非完整三维（`dim[0]!=3`、`dim[4..7]!=1`、维度非正） |
| `unsupported_datatype` | 400 | `datatype` | 仅接受 int16(4)/float32(16) |
| `bitpix_mismatch` | 400 | `bitpix` | 与 datatype 不一致 |
| `invalid_vox_offset` | 400 | `vox_offset` | 非有限整数或 < 352 |
| `payload_length_mismatch` | 400 | `file` | 载荷长度与头部不符（截断） |
| `trailing_bytes` | 400 | `file` | 文件含尾随字节 |
| `non_finite_scaling` | 400 | `scl_slope`/`scl_inter` | 缩放参数非有限 |
| `missing_affine` | 400 | `qform_code,sform_code` | 两种变换码均为 0 |
| `non_finite_affine` | 400 | `srow_*`/`quatern_*`/`qoffset_*` | 仿射参数非有限 |
| `invalid_pixdim` | 400 | `pixdim` | qform 所需体素尺寸非正/非有限 |
| `singular_affine` | 400 | `srow`/`qform` | 所选仿射不可逆 |

逐点错误（200 响应内）：`out_of_bounds`（逆变换后落在体素中心闭域
`[0, n-1]` 之外）、`non_finite_data`（插值邻域内缩放后数据非有限）、
`non_finite_gradient`（仅请求 `world_gradient` 时可能出现：梯度单元
2×2×2 邻域内存在非有限数据；采样本身成功、只有梯度无法计算时也用此码）。

### `GET /healthz`

就绪探针：服务启动时对采样流水线做自检，通过后才返回
`200 {"status": "ok", "ready": true}`，否则 503。

## 接受的 NIfTI 子集与采样规则

- 单文件 `.nii`（magic `n+1`），完整三维：`dim[0]=3`、`dim[4..7]=1`、维度为正。
- 数据类型 int16 或 float32，bitpix 一致；大端/小端均可（按 `sizeof_hdr` 判定）。
- `vox_offset` 为 ≥352 的有限整数；载荷长度须与维度精确一致，不得有尾随字节。
- `scl_slope`/`scl_inter` 必须有限；`scl_slope == 0` 时按 NIfTI 约定不缩放。
- 仿射选择：**有效 sform（`sform_code>0`）优先于 qform**；两者皆缺、或所选
  仿射不可逆（行列式为 0/非有限）则拒绝。qform 按 nifti1_io 规则由四元数、
  `pixdim[1..3]` 与 `qfac = sign(pixdim[0])` 重建。
- 世界坐标经所选仿射的逆变换映射到连续体素坐标，须落在体素中心闭域
  `[0, n-1]`（各轴，含边界；另有 1e-6 体素的浮点容差）。对缩放后的 8 个
  邻近体素做三线性插值；边界轴固定到唯一端点（权重 1），零权重邻居不参与。
- 世界梯度（`derivatives=world_gradient`）在同一三线性场上以体素轴有限差分
  计算，并经 `g_world = M^{-T} g_voxel` 映射到 RAS 世界方向；分界点取负侧
  单元、零侧取正侧、单体素轴分量为 0（详见 API 一节）。

## 运行

本地（需 Python ≥ 3.11）：

```bash
PORT=8000 python -m app.server
```

Docker（宿主机端口用 `NIFTI_HOST_PORT` 配置，默认 8000）：

```bash
docker compose build
NIFTI_HOST_PORT=9000 docker compose up -d app
```

## 验证（verify 一次性服务）

```bash
docker compose up --build --exit-code-from verify verify
```

`verify` 等待 `app` 健康检查通过后执行三个阶段，退出码按位汇总：

| 位 | 值 | 阶段 |
| --- | --- | --- |
| bit0 | 1 | 代码测试（`tests/` 单元 + 集成测试） |
| bit1 | 2 | 镜像构建校验（构建清单、运行时版本、模块导入、采样自检） |
| bit2 | 4 | API 冒烟（大/小端 × sform/qform × int16/float32 样本矩阵 + 结构错误与逐点错误用例 + 强度缩放与梯度边界，含单体素轴、两类仿射与非有限梯度邻域） |

退出码 0 表示全部通过。本地复现（stage 2 需要镜像构建清单
`image-manifest.json`，仅在 Dockerfile 构建时生成）：

```bash
python -m unittest discover -s tests -t .          # 仅代码测试
VERIFY_BASE_URL=http://127.0.0.1:8000 python -m verify.verify
```

## 目录结构

```
app/            服务实现（server: HTTP/multipart；nifti: 头部解析校验；sampling: 插值）
tests/          单元与集成测试（stdlib unittest）
verify/         一次性校验服务 + NIfTI 样本生成器 + multipart 客户端
Dockerfile      单阶段镜像（python:3.11-slim，无外部依赖）
docker-compose.yml  app（健康检查、可配置宿主机端口）+ verify（一次性）
```
