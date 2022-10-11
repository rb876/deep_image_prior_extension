
def one_step_gd_update_wtups(tupparams, tupgrads, scalalpha):
    assert len(tupparams) == len(tupgrads)
    update = []
    for param, grad in zip(tupparams, tupgrads): 
        update.append(param - scalalpha * grad)
    return tuple(update)
