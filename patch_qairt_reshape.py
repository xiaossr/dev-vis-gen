"""
Monkey-patch QAIRT SDK 2.45 to fix C++ memory corruption.

Root cause: IrStaticTensor objects are created as temporaries in Op.__init__
methods. When they go out of scope, their data (including numpy arrays) gets
freed. The C++ Op objects hold raw pointers to this freed data, causing
garbage values in shape inference.

Fix: Replace ir_graph.IrStaticTensor/IrTensor with factory functions that keep
references to all created tensors and their numpy array arguments in a global
list, preventing them from being freed.

Usage: import this module before calling qairt.convert()
"""
import logging

logger = logging.getLogger("patch_qairt_reshape")

# Global list to keep all C++ wrapper objects and numpy arrays alive
_keep_alive_refs = []


def apply_patch():
    """Patch QAIRT to prevent C++ memory corruption from freed Python objects."""
    try:
        from qti.aisw.converters.common import ir_graph

        # Wrap IrStaticTensor constructor to keep all args alive
        _original_IrStaticTensor = ir_graph.IrStaticTensor

        def _safe_IrStaticTensor(*args, **kwargs):
            _keep_alive_refs.append(args)
            _keep_alive_refs.append(kwargs)
            obj = _original_IrStaticTensor(*args, **kwargs)
            _keep_alive_refs.append(obj)
            return obj

        ir_graph.IrStaticTensor = _safe_IrStaticTensor

        # Wrap IrTensor constructor to keep all args alive
        if hasattr(ir_graph, "IrTensor"):
            _original_IrTensor = ir_graph.IrTensor

            def _safe_IrTensor(*args, **kwargs):
                _keep_alive_refs.append(args)
                _keep_alive_refs.append(kwargs)
                obj = _original_IrTensor(*args, **kwargs)
                _keep_alive_refs.append(obj)
                return obj

            ir_graph.IrTensor = _safe_IrTensor

        # Wrap IrTensorShape constructor to keep all args alive
        if hasattr(ir_graph, "IrTensorShape"):
            _original_IrTensorShape = ir_graph.IrTensorShape

            def _safe_IrTensorShape(*args, **kwargs):
                _keep_alive_refs.append(args)
                _keep_alive_refs.append(kwargs)
                obj = _original_IrTensorShape(*args, **kwargs)
                _keep_alive_refs.append(obj)
                return obj

            ir_graph.IrTensorShape = _safe_IrTensorShape

        # Also need to preserve PyIrStaticTensor / PyIrTensor if they exist
        if hasattr(ir_graph, 'PyIrStaticTensor'):
            _original_PyIrStaticTensor = ir_graph.PyIrStaticTensor

            def _safe_PyIrStaticTensor(*args, **kwargs):
                _keep_alive_refs.append(args)
                _keep_alive_refs.append(kwargs)
                obj = _original_PyIrStaticTensor(*args, **kwargs)
                _keep_alive_refs.append(obj)
                return obj

            ir_graph.PyIrStaticTensor = _safe_PyIrStaticTensor

        if hasattr(ir_graph, 'PyIrTensor'):
            _original_PyIrTensor = ir_graph.PyIrTensor

            def _safe_PyIrTensor(*args, **kwargs):
                _keep_alive_refs.append(args)
                _keep_alive_refs.append(kwargs)
                obj = _original_PyIrTensor(*args, **kwargs)
                _keep_alive_refs.append(obj)
                return obj

            ir_graph.PyIrTensor = _safe_PyIrTensor

        if hasattr(ir_graph, 'PyIrTensorShape'):
            _original_PyIrTensorShape = ir_graph.PyIrTensorShape

            def _safe_PyIrTensorShape(*args, **kwargs):
                _keep_alive_refs.append(args)
                _keep_alive_refs.append(kwargs)
                obj = _original_PyIrTensorShape(*args, **kwargs)
                _keep_alive_refs.append(obj)
                return obj

            ir_graph.PyIrTensorShape = _safe_PyIrTensorShape

        # Wrap IrAttributes constructor and add method
        _original_IrAttributes = ir_graph.IrAttributes

        def _safe_IrAttributes(*args, **kwargs):
            obj = _original_IrAttributes(*args, **kwargs)
            _keep_alive_refs.append(obj)
            return obj

        ir_graph.IrAttributes = _safe_IrAttributes

        logger.info(
            "Patched IrStaticTensor/IrTensor and IrAttributes to keep references alive (%d refs so far)",
            len(_keep_alive_refs),
        )
        return True

    except ImportError as e:
        logger.warning("Could not apply QAIRT patches: %s", e)
        return False


# Auto-apply on import
apply_patch()
