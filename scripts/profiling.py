%load aimport

import torch
import numpy as np
from torch import nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler

from sklearn.metrics import mean_absolute_error
from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
import src.models.parallel_scikit as ps

######################### Test profiling
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet20'),
        activities=[ProfilerActivity.CPU],
        with_stack=True,
        record_shapes=True) as prof:    
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages(
    group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=30))

########################## Profile models
i = -1

spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("TEST", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()
ptemp = model_wrapper.params()
ptemp["n_epochs"] = 1

i += 1
def trace_handler(p):
    print(p.key_averages(group_by_stack_n=5).table(
        row_limit=300)) 
    torch.profiler.tensorboard_trace_handler(f'log/obn_{i}')(p)
    
profiler = profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=3, warmup=10, active=1, repeat=1),
    on_trace_ready=trace_handler)
ptemp["profile"] = profiler

regr = model_wrapper.make(ptemp)
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)
ps.set_all_seeds(0)

with profiler:    
    regr.fit(X, Y)

yhat = model_wrapper.predict_val(regr, Xv)

mean_absolute_error(Yv, yhat)

############################# TEST
class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero()

        return out, hi_idx

model = MyModule(500, 10)
input = torch.rand(128, 500, dtype=torch.float32)
mask = torch.rand((500, 500, 500), dtype=torch.float32)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

with profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=trace_handler
) as p:
    for idx in range(20):
        model(input, mask)
        p.step()    
