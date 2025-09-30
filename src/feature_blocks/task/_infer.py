# Cache to avoid re-instantiating models on each call within the same worker
_model_cache = {}


def infer(chunk, model_identifier):
    """
    Infer features from a chunk using the specified model.

    Args:
        chunk: Input data chunk
        model_identifier: Either a string (model name) or a callable (model instance)

    Returns:
        Extracted features
    """
    # Check if it's already a callable (for backwards compatibility)
    if callable(model_identifier):
        return model_identifier(chunk)

    # Otherwise treat it as a model name string
    if model_identifier not in _model_cache:
        # Lazy import to avoid circular dependency
        from feature_blocks.models import available_models

        assert (
            model_identifier in available_models
        ), f"'{model_identifier}' is not a valid model name. Valid names are: {', '.join(list(available_models.keys()))}"

        _model_cache[model_identifier] = available_models[model_identifier]()

    feature_extract_fn = _model_cache[model_identifier]
    return feature_extract_fn(chunk)
