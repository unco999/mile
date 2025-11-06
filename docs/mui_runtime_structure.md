# Mile UI Runtime Structure


# MuiRuntime Entry

- holds BufferArena, RenderPipelines, RuntimeState
- exposes egin_frame, push_cpu_event, egister_panel_events, write_panel_bytes
- stores global EventBus / MileDb handles and simple frame history


