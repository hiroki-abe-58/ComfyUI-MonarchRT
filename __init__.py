"""ComfyUI-MonarchRT: unofficial ComfyUI integration of MonarchRT (training-free, on the public Self-Forcing weights)."""

# ComfyUI imports this file as a package. pytest also imports it, as a bare
# module without a parent package, while collecting the repository root;
# the relative import is only meaningful in the first case.
if __package__:
    from .monarchrt_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
