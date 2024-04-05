from fvcore.nn import FlopCountAnalysis

def _init_flops_counters():
    global FLOPS_PER_BATCH_COUNTER
    global FLOPS_PER_EPOCH_COUNTER
    FLOPS_PER_BATCH_COUNTER = 0
    FLOPS_PER_EPOCH_COUNTER = 0

def get_flops_per_epoch(model, dataset):
    global FLOPS_PER_EPOCH_COUNTER

    x, _ = dataset[0]
    flops = FlopCountAnalysis(model, x)
    total_flops = flops.total() * len(dataset)
    FLOPS_PER_EPOCH_COUNTER += total_flops
    return FLOPS_PER_EPOCH_COUNTER

def get_flops_per_batch(model, batch):
    global FLOPS_PER_BATCH_COUNTER
    x, _ = batch
    flops = FlopCountAnalysis(model, x)
    FLOPS_PER_BATCH_COUNTER += flops.total()
    return FLOPS_PER_BATCH_COUNTER