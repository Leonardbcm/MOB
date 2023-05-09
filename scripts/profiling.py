%load aimport

import torch, pandas, numpy as np, os
from torch import nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler

from sklearn.metrics import mean_absolute_error
from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
import src.models.parallel_scikit as ps

from src.models.torch_models.torch_obn import parse_key_averages_output, filter_key_averages

########################## Profile models
spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("TEST", "Bruges", country="BE", spliter=spliter,
                           skip_connection=True,  use_order_books=False,
                           order_book_size=20, alpha=1/3, beta=1/3, gamma=1/3)
X, Y = model_wrapper.load_train_dataset()
ptemp = model_wrapper.params()
ptemp["n_epochs"] = 1

def trace_handler(p):
    out = p.key_averages(group_by_stack_n=5)
    print(out)
    df = filter_key_averages(out)
    df.to_csv(os.path.join(model_wrapper.logs_path,f"table_{model_wrapper.ID}.csv"))
    torch.profiler.tensorboard_trace_handler(model_wrapper.logs_path)(p)
    
profiler = profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=3, warmup=2, active=25, repeat=1),
    on_trace_ready=trace_handler)
ptemp["profile"] = profiler

regr = model_wrapper.make(ptemp)
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)
ps.set_all_seeds(0)

with profiler as prof:    
    regr.fit(X, Y)

yhat = model_wrapper.predict_val(regr, Xv)    

print(model_wrapper.logs_path)

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
