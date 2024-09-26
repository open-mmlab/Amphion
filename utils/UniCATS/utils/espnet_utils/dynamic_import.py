import importlib


def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet_transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ':' not in import_path:
        raise ValueError(
            'import_path should be one of {} or '
            'include ":", e.g. "espnet_transform.add_deltas:AddDeltas" : '
            '{}'.format(set(alias), import_path))
    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, objname)
