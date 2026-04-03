# Research: Driving ComfyUI Programmatically from Claude Code

**Date:** 2026-04-03
**Sources:** 14 sources

---

## Executive Summary

Claude Code can drive ComfyUI through three distinct layers, each appropriate for different use cases. The lowest layer is ComfyUI's native HTTP/WebSocket API, which is stable, well-documented, and requires no additional dependencies — a Claude Code agent can submit a workflow JSON to `POST /prompt`, receive a `prompt_id`, poll `GET /history/{prompt_id}` until the job completes, and fetch images via `GET /view`. Above that, several Python client libraries (ComfyAPI, comfy-catapult, ComfyScript, comfy-api-simplified) abstract away the polling loop and provide typed workflow manipulation. The highest-level option is MCP: at least three community MCP servers exist for ComfyUI (the most capable being `artokun/comfyui-mcp`, installable as a Claude Code plugin), and an official Comfy Cloud MCP server is in limited early access. The practical recommendation is to use the bare HTTP API for a one-file agent script, ComfyAPI or comfy-catapult for a proper Python pipeline, and the `comfyui-mcp` plugin for full Claude Code interactive sessions where image generation should be callable as a tool.

---

## Key Findings

### The ComfyUI HTTP/WebSocket API Surface

ComfyUI exposes a REST API on `http://127.0.0.1:8188` by default [1]. The core workflow for programmatic use involves four endpoints. `POST /prompt` submits a workflow in "API format" JSON; the server validates it, enqueues it, and returns `{"prompt_id": "<uuid>", "number": <queue_position>}` or an error body with `node_errors` [1]. `GET /queue` returns the current running and pending items [1]. `GET /history/{prompt_id}` returns the completed job including all output file references; `GET /history` returns the full history with an optional `max_items` parameter [1]. `GET /view?filename=<name>&subfolder=<path>&type=output` downloads the actual image bytes [1][2].

For real-time progress without polling, the `GET /ws?clientId=<uuid>` WebSocket endpoint streams JSON messages: `executing` (which node is running), `status` (queue depth), and `progress` (step count with `value` and `max` fields) [2][3]. A job is considered complete when a `status` message arrives with `queue_remaining == 0`. The WebSocket approach is preferable when you need live progress bars; the HTTP polling approach is simpler for fire-and-forget automation [3].

Additional useful endpoints include `GET /object_info` (returns metadata for every registered node type, useful for dynamic workflow construction), `GET /models/{folder}` (lists available model files), `POST /upload/image` (injects input images), `POST /interrupt` (cancels the running job), and `POST /free` (unloads models from VRAM) [1][4].

The workflow must be in "API format" — a flat JSON object where keys are node IDs (numeric strings like `"3"`, `"6"`) and each value has a `class_type` string and an `inputs` dict. Links between nodes are expressed as `["source_node_id", output_index]` inside the `inputs` dict. This differs from the "UI format" that ComfyUI saves to disk by default; to obtain API format from the GUI, enable Dev Mode Options in settings and use "Save (API Format)" [5][6].

### Python Client Libraries

Several libraries wrap the raw HTTP/WebSocket API with varying feature depth.

**ComfyAPI** (`pip install comfyapi`) is the most ergonomic option for straightforward automation [7]. It provides `ComfyAPIManager` with `load_workflow()`, `edit_workflow(path, value)` for updating nested inputs by path, `submit_workflow()` returning a `prompt_id`, `check_queue(prompt_id)` for polling, `find_output(prompt_id)` for extracting result URLs, and `download_output(url, save_path)` for saving images locally. A complete generate-and-save cycle is around 10 lines. It also supports `batch_submit(num_seeds, seed_node_path)` for seed sweeps and `set_base64_image(node_id, image_path)` for injecting local images.

**comfy-catapult** (`pip install comfy_catapult`) is the most production-grade option [8]. It includes a `ComfyCatapult` async scheduler, a `ComfyAPIClient`, and Pydantic models for type-safe workflow validation (`APIWorkflow.model_validate_json()`). The API is fully async/await; you call `await catapult.Catapult(job_id=..., prepared_workflow=..., important=...)` which returns a status and future, and await the future for results. A companion repo (`comfy-catapult-fastapi`) demonstrates wrapping it in a FastAPI service for public-facing multi-user deployments.

**ComfyScript** (`pip install "comfy-script[default]"`) takes a different philosophy: instead of manipulating workflow JSON, you write workflows as Python code using callable node functions [9]. Nodes become Python functions (`CheckpointLoaderSimple`, `CLIPTextEncode`, `KSampler`, etc.) invoked inside a `with Workflow():` context. Scripts are human-readable, diffable, and can use Python loops, conditionals, and library calls. ComfyScript also includes a transpiler that converts existing workflow JSON to ComfyScript code automatically. This is the best choice when you want to generate varied or complex workflows from code rather than editing a fixed template.

**comfyui-python-api** (`pip install comfyui_utils`) is a lighter-weight library primarily designed for chatbot integrations [10]. It supports parameter parsing embedded in text strings, callback-based progress handling, and cached result retrieval. It currently only supports integer parameters, so it is more limited than the options above.

**comfy-api-simplified** (`pip install comfy-api-simplified`) provides minimal workflow editing and queueing, suitable for simple one-shot scripts [11].

### MCP Servers for ComfyUI

At least three separate MCP server implementations exist for ComfyUI, plus an official cloud option.

The most feature-complete is `artokun/comfyui-mcp`, which is explicitly designed as a Claude Code plugin [12]. It installs via `claude plugin install comfyui-mcp` or via `npx -y comfyui-mcp` in `.mcp.json`. It exposes 31 MCP tools across 13 categories covering workflow execution, image management, model discovery, VRAM monitoring, custom node installation, and diagnostics. It also provides 10 Claude Code slash commands (`/comfy:gen`, `/comfy:batch`, `/comfy:debug`, `/comfy:viz`, `/comfy:recipe`, `/comfy:install`, etc.), 4 knowledge skills for prompt engineering and troubleshooting, and 3 autonomous agents (comfy-explorer for node research, comfy-debugger for failure diagnosis, comfy-optimizer for VRAM profiling). A SQLite database tracks generation history with deduplication. It requires Node.js ≥22 and auto-detects a running ComfyUI instance.

`alecc08/comfyui-mcp` is a simpler alternative targeting local ComfyUI [13]. It exposes 7 tools: `comfyui_generate_image`, `comfyui_modify_image` (img2img), `comfyui_resize_image`, `comfyui_remove_background`, `comfyui_get_image`, `comfyui_list_workflows`, and `comfyui_get_request_history`. Configuration requires setting `COMFYUI_URL` and `COMFYUI_WORKFLOW_DIR`. It serves generated images via a local HTTP proxy on port 8190 rather than base64-encoding them. Installation requires `npm install && npm run build`.

The **official Comfy Cloud MCP Server** (documented at `docs.comfy.org/development/cloud/mcp-server`) connects to Comfy Cloud's hosted GPU infrastructure rather than a local instance [1]. It is in limited early access as of early 2026 and translates MCP tool calls to cloud workflow executions. This is the appropriate path if no local GPU is available and cloud costs are acceptable.

### LLM Agent Use Cases and Research

Beyond Claude Code integration, there is active research into using LLMs to generate ComfyUI workflows from natural language. The most significant recent work is **ComfyGPT** (arxiv 2503.17671, March 2025) [14]. ComfyGPT is a four-agent pipeline: a ReformatAgent converts existing workflows to simplified link diagrams; a FlowAgent (Qwen2.5-14B, fine-tuned with SFT on 12,571 examples, then GRPO-optimized) generates new workflow link diagrams from task descriptions; a RefineAgent validates node types against a 6,362-node database and corrects errors; and an ExecuteAgent converts the diagram back to JSON and submits it. On the FlowBench benchmark of 1,000 diverse tasks, ComfyGPT achieved 86.0% pass accuracy and 84.8% instruction alignment, versus ~12% for few-shot prompted GPT-4 and 14.5% for the prior ComfyAgent system.

Within ComfyUI itself, several custom node packs embed LLM capabilities: `ComfyUI-IF_LLM` connects to Anthropic, OpenAI, Gemini, Ollama, and others to generate or rewrite prompts inside a workflow [4]. `comfyui_LLM_party` is a full LLM agent framework as ComfyUI nodes, supporting MCP server creation from within ComfyUI workflows, enabling bidirectional integration [4].

### Practical Integration Pattern for Claude Code

The minimal round-trip for a Claude Code agent to generate an image and retrieve the result using only the standard library is:

1. Load a pre-exported `workflow_api.json` (or construct one from scratch as a Python dict).
2. Patch the relevant inputs: positive prompt text at the CLIPTextEncode node, seed at KSampler, dimensions at EmptyLatentImage, etc. These are addressed by node ID and input key path.
3. `POST /prompt` with body `{"prompt": workflow_dict, "client_id": client_uuid}`. Capture `prompt_id`.
4. Poll `GET /history/{prompt_id}` every 0.5–1 second until the key appears in the response. The response maps node IDs to their outputs; `SaveImage` nodes produce `images` arrays with `filename`, `subfolder`, `type`.
5. `GET /view?filename=<filename>&subfolder=<subfolder>&type=output` to download the image bytes.

Using ComfyAPI, steps 3–5 compress to `submit_workflow()` + `while not check_queue(pid): sleep(1)` + `find_output(pid)` + `download_output(url, path)` [7]. Using comfy-catapult, the entire thing is a single `await catapult.Catapult(...)` followed by awaiting the future [8].

For Claude Code MCP integration, the `artokun/comfyui-mcp` plugin is the most powerful option: after `claude plugin install comfyui-mcp`, Claude Code can use `/comfy:gen "a sunset over mountains"` as a slash command, or call MCP tools directly in agent code [12]. The plugin handles ComfyUI process management, model inventory, and generation history automatically.

---

## Comparison

| Option | Install | Use Case | Async | Pydantic | Workflow Construction |
|---|---|---|---|---|---|
| Raw HTTP API | None (stdlib) | Minimal scripts | Manual | No | Manual JSON dict |
| ComfyAPI | `pip install comfyapi` | Simple automation | Poll loop | No | Edit by path |
| comfy-catapult | `pip install comfy_catapult` | Production pipelines | Full async | Yes | Validated dicts |
| ComfyScript | `pip install comfy-script[default]` | Complex/dynamic workflows | Yes | No | Python code |
| artokun/comfyui-mcp | `claude plugin install comfyui-mcp` | Claude Code sessions | Handled internally | N/A | Natural language |
| alecc08/comfyui-mcp | `npm install` + config | Simple Claude integration | Handled internally | N/A | Pre-built workflows |
| Comfy Cloud MCP | Cloud API key | No local GPU | Handled | N/A | Cloud workflows |

---

## Open Questions

The official `docs.comfy.org` routes page was inaccessible during research (WebFetch permission denied), so the endpoint table in this report is sourced from the DeepWiki mirror and community documentation — it should be verified against the upstream source if used for production code. The exact tool list exposed by the official Comfy Cloud MCP server could not be confirmed since the official docs page was unavailable; the cloud server's feature set may differ significantly from local MCP servers. The `artokun/comfyui-mcp` plugin's 31 tools are documented in the README but were not verified against a running instance; some tools may require optional configuration (CivitAI API key, etc.). Whether ComfyScript's transpiler handles all custom nodes correctly is unknown — community nodes with unusual input types may produce incomplete translations.

---

## Sources

[1] Comfy-Org / hiddenswitch. "Prompt Server and REST API." DeepWiki mirror. https://deepwiki.com/hiddenswitch/ComfyUI/4.1-prompt-server-and-rest-api (Retrieved: 2026-04-03)

[2] YushanT7. "ComfyUI: WebSockets API: Part 2." Medium. https://medium.com/@yushantripleseven/comfyui-websockets-api-part-2-0ab988acfd97 (Retrieved: 2026-04-03)

[3] YushanT7. "ComfyUI: Using the API: Part 1." Medium. https://medium.com/@yushantripleseven/comfyui-using-the-api-261293aa055a (Retrieved: 2026-04-03)

[4] Comfy-Org. "API and Programmatic Usage." DeepWiki. https://deepwiki.com/Comfy-Org/ComfyUI/7-api-and-programmatic-usage (Retrieved: 2026-04-03)

[5] Shawn Wong. "How to Use ComfyUI API with Python: A Complete Guide." Medium. https://medium.com/@next.trail.tech/how-to-use-comfyui-api-with-python-a-complete-guide-f786da157d37 (Retrieved: 2026-04-03, via search snippet)

[6] 9elements. "Hosting a ComfyUI Workflow via API." 9elements Blog. https://9elements.com/blog/hosting-a-comfyui-workflow-via-api/ (Retrieved: 2026-04-03, via search snippet)

[7] SamratBarai. "ComfyAPI: Python client for ComfyUI via WebSockets." GitHub. https://github.com/SamratBarai/ComfyAPI (Retrieved: 2026-04-03)

[8] realazthat. "comfy-catapult: Programmatically schedule ComfyUI workflows." GitHub. https://github.com/realazthat/comfy-catapult (Retrieved: 2026-04-03)

[9] Chaoses-Ib. "ComfyScript: A Python frontend and library for ComfyUI." GitHub. https://github.com/Chaoses-Ib/ComfyScript (Retrieved: 2026-04-03)

[10] andreyryabtsev. "comfyui-python-api: Utilities library for working with the ComfyUI API." GitHub. https://github.com/andreyryabtsev/comfyui-python-api (Retrieved: 2026-04-03)

[11] realazthat / PyPI. "comfy-api-simplified." PyPI. https://pypi.org/project/comfy-api-simplified/ (Retrieved: 2026-04-03, via search snippet)

[12] artokun. "comfyui-mcp: ComfyUI MCP server + Claude Code plugin." GitHub. https://github.com/artokun/comfyui-mcp (Retrieved: 2026-04-03)

[13] alecc08. "comfyui-mcp: Simple MCP server for text-to-image and img2img." GitHub. https://github.com/alecc08/comfyui-mcp (Retrieved: 2026-04-03)

[14] Anonymous et al. "ComfyGPT: A Self-Optimizing Multi-Agent System for Comprehensive ComfyUI Workflow Generation." arXiv:2503.17671. https://arxiv.org/html/2503.17671v1 (Retrieved: 2026-04-03)
