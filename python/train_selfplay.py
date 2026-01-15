# python/train_selfplay.py
# Self-play PPO-like training (one shared policy plays both sides)
# Uses C++ engine via pybind11: import arkomag_cpp
# Exports models/policy.onnx for C++ game (onnxruntime inference)

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import arkomag_cpp


# -------------------- Device (MPS on mac if available) --------------------
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("DEVICE =", DEVICE)

torch.set_default_dtype(torch.float32)
# Often helps stability / avoids CPU oversubscription
torch.set_num_threads(1)


# -------------------- Hyperparams --------------------
HIDDEN = 256
LR = 3e-4

CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

BATCH_DECISIONS = 8192     # how many decisions (turn actions) per update
EPOCHS = 4
MINIBATCH = 512
UPDATES = 1000            # increase for stronger bot (e.g., 3000+)

SELFPLAY_START = 40          # с этого апдейта P1 тоже управляется сетью
MIX_RANDOM_PROB = 0.20       # (опционально) даже после 40 иногда делаем P1 random для стабильности

EVAL_EVERY = 10
EVAL_GAMES = 100           # increase for more stable metric, decrease for speed
SAVE_EVERY = 200          # save .pt checkpoint
SEED0 = 1000              # training seed base


# -------------------- Model --------------------
class PolicyValue(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.pi = nn.Linear(HIDDEN, act_dim)  # logits
        self.v  = nn.Linear(HIDDEN, 1)

    def forward(self, obs):
        x = self.body(obs)
        return self.pi(x), self.v(x).squeeze(-1)


def masked_logits(logits: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, A]
    mask_bool: [B, A]  True=allowed, False=forbidden
    Use -1e9 instead of -inf/min-float for numerical stability (esp. on MPS).
    """
    neg = torch.full_like(logits, -1e9)
    return torch.where(mask_bool, logits, neg)


@torch.no_grad()
def eval_vs_random(net: nn.Module, games: int = 50, base_seed: int = 50000) -> float:
    """
    Evaluate current policy as P0 vs random P1.
    Returns winrate of P0.
    """
    eng = arkomag_cpp.Engine()
    wins = 0

    net.eval()
    for g in range(games):
        eng.reset(base_seed + g)

        while not eng.done():
            eng.ensure_turn_begun()
            p = int(eng.current_player())

            if p == 0:
                obs = np.asarray(eng.get_observation(0), dtype=np.float32)
                mask = np.asarray(eng.get_action_mask(0), dtype=np.bool_)

                obs_t = torch.from_numpy(obs).to(DEVICE).unsqueeze(0)
                mask_t = torch.from_numpy(mask).to(DEVICE).unsqueeze(0)

                logits, _ = net(obs_t)
                logits = masked_logits(logits, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample().item()  # stochastic, as you want
                eng.step(int(a))
            else:
                mask = np.asarray(eng.get_action_mask(1), dtype=np.bool_)
                legal = np.flatnonzero(mask)
                eng.step(int(np.random.choice(legal)))

        if int(eng.winner()) == 0:
            wins += 1

    net.train()
    return wins / games


def export_onnx(net: nn.Module, obs_dim: int, out_path: str):
    class PolicyOnly(nn.Module):
        def __init__(self, net_):
            super().__init__()
            self.net = net_
        def forward(self, obs):
            logits, _ = self.net(obs)
            return logits

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    policy = PolicyOnly(net).to("cpu").eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        policy, dummy, out_path,
        input_names=["obs"], output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )


def main():
    os.makedirs("models", exist_ok=True)

    # Init engine to determine obs/action dims
    eng = arkomag_cpp.Engine()
    eng.reset(1)
    obs_dim = len(eng.get_observation(0))
    act_dim = int(eng.action_size())

    net = PolicyValue(obs_dim, act_dim).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)
    best_wr = -1.0
    seed = SEED0
    t0 = time.time()

    for upd in range(1, UPDATES + 1):
        # ---- rollout buffers ----
        obs_buf  = []
        mask_buf = []
        act_buf  = []
        logp_buf = []
        val_buf  = []
        ret_buf  = []  # +1 winner moves, -1 loser moves (for all moves in episode)

        # ---- collect self-play batch ----
        while len(act_buf) < BATCH_DECISIONS:
            seed += 1
            eng.reset(seed)
            episode_steps = []  # list of (global_idx, player_who_acted)

            while (not eng.done()) and (len(act_buf) < BATCH_DECISIONS):
                eng.ensure_turn_begun()
                p = int(eng.current_player())

                # --- choose action depending on phase ---
                use_policy = True
                if p == 1 and upd < SELFPLAY_START:
                    use_policy = False
                elif p == 1 and MIX_RANDOM_PROB > 0.0:
                    # после self-play старта можно иногда оставлять random-ходы для устойчивости
                    if np.random.rand() < MIX_RANDOM_PROB:
                        use_policy = False

                if use_policy:
                    obs = np.asarray(eng.get_observation(p), dtype=np.float32)
                    mask = np.asarray(eng.get_action_mask(p), dtype=np.bool_)

                    obs_t = torch.from_numpy(obs).to(DEVICE).unsqueeze(0)
                    mask_t = torch.from_numpy(mask).to(DEVICE).unsqueeze(0)

                    with torch.no_grad():
                        logits, v = net(obs_t)
                        logits = masked_logits(logits, mask_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample()
                        logp = dist.log_prob(a)

                    idx = len(act_buf)
                    obs_buf.append(obs)
                    mask_buf.append(mask)
                    act_buf.append(int(a.item()))
                    logp_buf.append(float(logp.item()))
                    val_buf.append(float(v.item()))
                    episode_steps.append((idx, p))

                    eng.step(int(a.item()))
                else:
                    # random move (only for P1 in warmup or mixed)
                    mask = np.asarray(eng.get_action_mask(p), dtype=np.bool_)
                    legal = np.flatnonzero(mask)
                    a = int(np.random.choice(legal))
                    eng.step(a)
                    w = int(eng.winner())
            for (idx, p) in episode_steps:
                ret_buf.append(1.0 if p == w else -1.0)

        # ---- to tensors ----
        obs_t  = torch.from_numpy(np.asarray(obs_buf, dtype=np.float32)).to(DEVICE)
        mask_t = torch.from_numpy(np.asarray(mask_buf, dtype=np.bool_)).to(DEVICE)
        act_t  = torch.from_numpy(np.asarray(act_buf, dtype=np.int64)).to(DEVICE)

        oldlogp_t = torch.from_numpy(np.asarray(logp_buf, dtype=np.float32)).to(DEVICE)
        oldv_t    = torch.from_numpy(np.asarray(val_buf, dtype=np.float32)).to(DEVICE)
        ret_t     = torch.from_numpy(np.asarray(ret_buf, dtype=np.float32)).to(DEVICE)

        adv_t = (ret_t - oldv_t).detach()
        std = adv_t.std()
        if std > 1e-3:
            adv_t = (adv_t - adv_t.mean()) / (std + 1e-8)
        else:
            # если почти константа — хотя бы центрируем, но не "убиваем" градиент делением
            adv_t = adv_t - adv_t.mean()


    # ---- PPO update ----
        n = obs_t.shape[0]
        idxs = np.arange(n)

        loss_pi_acc = 0.0
        loss_v_acc = 0.0
        ent_acc = 0.0
        cnt = 0

        for _ in range(EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, n, MINIBATCH):
                mb = idxs[start:start+MINIBATCH]
                mb_obs = obs_t[mb]
                mb_mask = mask_t[mb]
                mb_act = act_t[mb]
                mb_oldlogp = oldlogp_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                logits, v = net(mb_obs)
                logits = masked_logits(logits, mb_mask)
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_oldlogp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = (v - mb_ret).pow(2).mean()

                loss = loss_pi + VF_COEF * loss_v - ENT_COEF * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

                loss_pi_acc += float(loss_pi.item())
                loss_v_acc  += float(loss_v.item())
                ent_acc     += float(entropy.item())
                cnt += 1

        # ---- logging ----
        if upd % 10 == 0:
            phase = "SELF" if upd >= SELFPLAY_START else "WARMUP(vs random)"

            dt = time.time() - t0
            t0 = time.time()
            print(
                f"[train] upd={upd:5d} "
                f"phase={phase:16s} "
                f"batch={n:5d} "
                f"loss_pi={loss_pi_acc/max(cnt,1):.4f} "
                f"loss_v={loss_v_acc/max(cnt,1):.4f} "
                f"ent={ent_acc/max(cnt,1):.4f} "
                f"sec={dt:.2f}"
            )


        if upd % EVAL_EVERY == 0:
            wr = eval_vs_random(net, games=EVAL_GAMES, base_seed=70000 + upd * 100)
            print(f"[eval ] upd={upd:5d} winrate_vs_random={wr:.3f} (games={EVAL_GAMES})")

        if upd % SAVE_EVERY == 0:
            torch.save(net.state_dict(), "models/policy.pt")
            print("[save ] models/policy.pt")

    # ---- export ONNX ----
    export_onnx(net, obs_dim, "models/policy.onnx")
    print("[done ] models/policy.onnx exported")


if __name__ == "__main__":
    main()
