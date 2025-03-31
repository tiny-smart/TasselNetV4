from models.local_count_multihead.CntVit_3layers_scalepos_patchmat_withr_plain import local_count_mutihead_loose
def build_model(**kwargs):
    name = kwargs.pop('name')
    if isinstance(name,str):
        model = globals()[name](**kwargs)
    else:
        model = name(**kwargs)
    return model