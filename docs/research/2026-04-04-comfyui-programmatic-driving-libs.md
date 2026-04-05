# Research: ComfyUI Programmatic Driving Libraries

**Date:** 2026-04-04  
**Sources:** 18 sources (GitHub API, PyPI, README reads, official docs)

---

## Executive Summary

The ComfyUI programmatic library space is fragmented: many projects exist, most are abandoned, a handful are healthy. The two clear survivors in the Python-client space are **comfy_api_simplified** (active, ~420 lines, solid code quality, MCP server included) and **Chaoses-Ib/ComfyScript** (661 stars, active Feb 2026, code-as-workflow paradigm ideal for LLM-driven generation). The originally-recommended `comfy-catapult` is **dead** (last commit July 2024). For the MCP/agentic layer, **joenorton/comfyui-mcp-server** is the best option: Python-native, 274 stars, tested, well-structured. The `artokun/comfyui-mcp` plugin has only 24 stars but is being actively pushed as of yesterday. For a pipeline focused on learning, a thin custom ~200-line client using `httpx` is worth building first — the ComfyUI API is simple enough that understanding it directly is valuable before committing to any library.

---

## Key Findings

### Project Landscape and Alive/Dead Assessment

The following table summarizes all identified projects with their current health:

| Project | Stars | Last Push | Status | Type |
|---|---|---|---|---|
| `Chaoses-Ib/ComfyScript` | 661 | 2026-02-17 | **ALIVE** | Python workflow DSL |
| `joenorton/comfyui-mcp-server` | 274 | 2026-02-17 | **ALIVE** | Python MCP server |
| `ConstantineB6/Comfy-Pilot` | 159 | 2026-02-16 | **ALIVE** | MCP + terminal |
| `deimos-deimos/comfy_api_simplified` | 109 | 2026-03-04 | **ALIVE** | Python HTTP client |
| `artokun/comfyui-mcp` | 24 | 2026-04-03 | **ALIVE** | Node.js MCP + Claude plugin |
| `Comfy-Org/comfy-cloud-mcp` | 3 | 2026-03-30 | **ALIVE** (cloud-only) | Official cloud MCP |
| `sugarkwork/Comfyui_api_client` | 29 | 2025-06-12 | marginal | Python HTTP client |
| `realazthat/comfy-catapult` | 27 | 2024-07-09 | **DEAD** | Python async client |
| `andreyryabtsev/comfyui-python-api` | 45 | 2023-12-19 | **DEAD** | Python HTTP client |
| `SamratBarai/ComfyAPI` | 0 | 2026-01-04 | Dead/minor | Python HTTP client |

Two projects previously referenced in the project's research doc are no longer viable: `comfy-catapult` has not been touched in 9 months and has only 2 forks; `andreyryabtsev/comfyui-python-api` was last updated in December 2023. There is also a `comfyorg/comfyscript` repo under the official Comfy-Org GitHub organization (0 stars, last push Nov 2025) that appears to be an early mirror of `Chaoses-Ib/ComfyScript` — the community still treats the Chaoses-Ib repo as canonical.

No official Python SDK from Comfy-Org exists. The official documentation references the raw HTTP API and `ComfyScript` as the programmatic paths.

### comfy_api_simplified — Best Simple HTTP Client

`deimos-deimos/comfy_api_simplified` (109 stars, active March 2026) is the cleanest general-purpose Python HTTP client that is currently maintained [1]. The codebase is ~420 lines across five files: `comfy_api_wrapper.py`, `comfy_workflow_wrapper.py`, `exceptions.py`, `mcp_server.py`, and `__init__.py`. Code quality is high: consistent type hints throughout, comprehensive docstrings, a custom `ComfyApiError` exception hierarchy, and a match-case pattern for custom node result extraction. The key method `queue_prompt_and_wait()` uses WebSocket-based completion detection rather than HTTP polling — this is superior to the polling approach since it avoids both over-polling and completion latency.

Known weaknesses: no configurable timeout on HTTP/WebSocket connections, minimal retry logic for transient failures, and the WebSocket loop could hang indefinitely on unexpected message sequences. The library also requires `nest_asyncio` as a workaround for environments (like Jupyter) that already have an event loop running — a known friction point. The built-in `mcp_server.py` is a bonus for the agentic use case, though it is thinner than the dedicated MCP servers.

The `ComfyWorkflowWrapper` class provides `set_node_param(node_title, param_name, value)` for modifying workflow parameters by node title — useful but requires unique node titles, which is a naming discipline constraint.

### ComfyScript — Best for LLM-Driven Workflow Generation

`Chaoses-Ib/ComfyScript` (661 stars, active Feb 2026) is architecturally different from the HTTP clients: it lets you write ComfyUI workflows as Python code [2]. Node functions (`CheckpointLoaderSimple`, `KSampler`, etc.) are dynamically generated from ComfyUI's `/object_info` endpoint and become callable Python functions inside a `with Workflow():` context. In virtual mode, calling a node doesn't execute it — instead, the entire workflow graph is collected and then submitted as JSON. In real mode, node calls execute immediately.

The key advantage for this project's agentic goal is the **transpiler**: existing workflow JSONs can be converted to ComfyScript automatically via `python -m comfy_script.transpile workflow.json`. This means Claude can receive a ComfyScript representation and modify it with full Python logic (loops, conditionals, variables) rather than manipulating opaque JSON. For an LLM generating varied texture workflows, this is significantly more readable and maintainable than workflow JSON editing.

The 42 open issues suggest active community use but also that the project has rough edges. The `run.py` file (which the fetch returned as a compatibility shim rather than core execution logic) shows the project uses some aggressive patching (`wrapt.ObjectProxy` on `__main__`) for compatibility with ComfyUI's custom node ecosystem — this is a known maintenance burden. Issues include real-mode stability (`How do I start Real Mode?`, March 2026), enum handling, and support for multiple ComfyUI instances.

Architecture is well-modularized: `client/`, `runtime/`, `transpile/`, `ui/` are clean sub-packages. Type stubs are generated dynamically from the running ComfyUI instance, enabling IDE autocomplete. The project is sponsored and has been adopted into the official `comfyorg` GitHub organization (though that mirror appears to lag behind).

### joenorton/comfyui-mcp-server — Best Python MCP Server

`joenorton/comfyui-mcp-server` (274 stars, active Feb 2026) is the strongest MCP server option [3]. It is Python-native (unlike artokun's Node.js implementation), has an actual test suite (`tests/test_basic.py`, `test_job_tools.py`, `test_edge_cases.py`, `test_bug_fixes.py`), and a proper architecture: `comfyui_client.py` (HTTP wrapper), `managers/` (asset registry, defaults, workflow, publish), `tools/` (generation, job, workflow, asset, configuration, publish). The CI includes a `.github/workflows/test.yml`.

The client implementation (`comfyui_client.py`) has good defensive coding — multiple fallback mechanisms for ComfyUI's variable response formats, comprehensive logging, and graceful timeout (returns `None` after max retries rather than raising). The asset identity system is UUID-based and survives hostname changes. The main weakness is ephemeral asset storage (24h TTL, no persistence across restarts).

It exposes ~15 MCP tools covering the full job lifecycle: `generate_image`, `regenerate`, `view_image`, `get_queue_status`, `get_job`, `cancel_job`, `list_assets`, `get_asset_metadata`, `list_models`, `list_workflows`, `run_workflow`, `get_defaults`, `set_defaults`, `publish_asset`.

### artokun/comfyui-mcp — Feature-Rich but Small Community

`artokun/comfyui-mcp` (24 stars, pushed yesterday) is the most feature-complete Claude Code integration [4]: 31 tools, 10 slash commands, 4 knowledge skills, 3 autonomous agents. It is Node.js-based (requires Node ≥22) and installs as a `claude plugin`. The breadth is impressive — HuggingFace model search, CivitAI model downloads, workflow-to-Mermaid visualization, custom node discovery. The project was very recently pushed (2026-04-03), suggesting active development but a small user base. Given it's Node.js and has 24 stars, there is a real risk it stalls.

### The Raw API — Simple Enough to Reimplement

The official ComfyUI Python example (`script_examples/basic_api_example.py`) demonstrates that the minimal round-trip is trivially simple: construct a workflow dict, POST to `/prompt`, get a `prompt_id`, poll `/history/{prompt_id}` until outputs appear, call `/view` to download. The API has no authentication, no pagination for the core workflow, and no complex state. The full surface needed for this pipeline is approximately:

- `POST /prompt` — submit
- `GET /history/{prompt_id}` — poll completion
- `GET /view?filename=&subfolder=&type=output` — download
- `POST /upload/image` — inject input images
- `GET /models/{folder}` — list available models
- `GET /system_stats` — health check
- `GET /ws?clientId=` — WebSocket for real-time progress (optional)

A clean implementation of this surface is ~200 lines. `comfy_api_simplified` is 420 lines including workflow editing helpers. The gap is primarily in WebSocket-based completion detection and the `ComfyWorkflowWrapper` parameter editing helpers.

---

## Comparison

| Dimension | comfy_api_simplified | ComfyScript | Custom implementation |
|---|---|---|---|
| **Alive** | Yes (Mar 2026) | Yes (Feb 2026) | N/A |
| **Stars** | 109 | 661 | N/A |
| **Code quality** | Good (420 lines, typed, docstrings) | Good (modular, typed, transpiler) | Your choice |
| **API surface** | HTTP + WebSocket | Python DSL → JSON | HTTP + WebSocket |
| **Workflow editing** | `set_node_param()` by title | Python code | Manual dict manipulation |
| **LLM-friendly** | Moderate (JSON editing) | High (Python code generation) | Moderate |
| **MCP server included** | Thin (5 tools) | No | Must build or adopt |
| **Learning value** | Moderate (hides details) | Low (high abstraction) | High (see everything) |
| **Dependencies** | `websockets`, `nest_asyncio` | Many (mirrors ComfyUI internals) | `httpx` only |
| **Risk** | Low | Medium (real-mode instability) | Zero |

---

## Recommendation

**For this project's dual goal (learn + build pipeline), the right approach is a two-phase strategy:**

**Phase 1 — Build a thin custom client** (~200 lines, `httpx`). This serves the learning goal directly: you implement the WebSocket completion loop, the upload/download cycle, and the workflow JSON manipulation yourself. Once this works, you understand exactly what the libraries do and can make an informed choice about whether to replace it.

**Phase 2 — Adopt ComfyScript for agentic workflow generation.** When Claude or a Python script needs to generate varied workflows (different texture gen configurations, different ControlNet setups), ComfyScript's code-as-workflow approach is significantly more maintainable than editing JSON dicts. The transpiler (workflow JSON → ComfyScript) makes onboarding existing workflows painless. Use virtual mode only initially to avoid real-mode stability issues.

**For MCP**, evaluate `joenorton/comfyui-mcp-server` as a drop-in: it is Python-native, tested, and actively maintained. Reference its `comfyui_client.py` for the robustness patterns (multiple fallbacks, logging strategy) when building the custom client.

**Do not adopt** comfy-catapult (dead) or andreyryabtsev/comfyui-python-api (dead). Do not adopt artokun/comfyui-mcp as a dependency yet — the small community is a risk, though it is worth monitoring.

---

## Open Questions

- ComfyScript real-mode stability: the open issue "How do I start Real Mode?" (March 2026) suggests it may not be production-ready. Needs hands-on testing before committing to it for pipeline use.
- joenorton MCP server: does it support custom workflow JSON submission or only its built-in templates? The `run_workflow` tool suggests yes, but not confirmed from the README alone.
- Official Comfy-Org client: the `comfyorg/comfyscript` mirror exists but has 0 stars and no visible maintenance. Whether Comfy-Org intends to officially maintain a Python SDK is unclear.

---

## Sources

[1] deimos-deimos. "comfy_api_simplified". https://github.com/deimos-deimos/comfy_api_simplified (Retrieved: 2026-04-04)  
[2] Chaoses-Ib. "ComfyScript". https://github.com/Chaoses-Ib/ComfyScript (Retrieved: 2026-04-04)  
[3] joenorton. "comfyui-mcp-server". https://github.com/joenorton/comfyui-mcp-server (Retrieved: 2026-04-04)  
[4] artokun. "comfyui-mcp". https://github.com/artokun/comfyui-mcp (Retrieved: 2026-04-04)  
[5] realazthat. "comfy-catapult". https://github.com/realazthat/comfy-catapult (Retrieved: 2026-04-04)  
[6] comfyorg. "comfyscript (official mirror)". https://github.com/comfyorg/comfyscript (Retrieved: 2026-04-04)  
[7] Comfy-Org. "comfy-cloud-mcp". https://github.com/Comfy-Org/comfy-cloud-mcp (Retrieved: 2026-04-04)  
[8] ConstantineB6. "Comfy-Pilot". https://github.com/ConstantineB6/Comfy-Pilot (Retrieved: 2026-04-04)  
[9] Comfy-Org. "ComfyUI basic_api_example.py". https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/basic_api_example.py (Retrieved: 2026-04-04)  
[10] andreyryabtsev. "comfyui-python-api". https://github.com/andreyryabtsev/comfyui-python-api (Retrieved: 2026-04-04)  
[11] sugarkwork. "Comfyui_api_client". https://github.com/sugarkwork/Comfyui_api_client (Retrieved: 2026-04-04)  
