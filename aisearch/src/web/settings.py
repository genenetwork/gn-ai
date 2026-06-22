def configure(lm, api_key, model):
    lm.kwargs["api_key"] = api_key
    lm.model = model
    return lm
