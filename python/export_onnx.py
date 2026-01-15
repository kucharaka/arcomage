import os
import torch
import torch.nn as nn
import arkomag_cpp

HIDDEN = 256  # должно совпадать с train_selfplay.py

class PolicyValue(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.pi = nn.Linear(HIDDEN, act_dim)
        self.v  = nn.Linear(HIDDEN, 1)

    def forward(self, obs):
        x = self.body(obs)
        return self.pi(x), self.v(x).squeeze(-1)

class PolicyOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, obs):
        logits, _ = self.net(obs)
        return logits

def main():
    os.makedirs("models", exist_ok=True)

    eng = arkomag_cpp.Engine()
    eng.reset(1)
    obs_dim = len(eng.get_observation(0))
    act_dim = int(eng.action_size())

    net = PolicyValue(obs_dim, act_dim).to("cpu").eval()

    pt_path = "models/policy.pt"
    onnx_path = "models/policy.onnx"

    state = torch.load(pt_path, map_location="cpu")
    net.load_state_dict(state)

    policy = PolicyOnly(net).to("cpu").eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        policy, dummy, onnx_path,
        input_names=["obs"], output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,   # <-- ключевое: отключаем новый exporter
    )

    print("Saved:", onnx_path)

if __name__ == "__main__":
    main()
