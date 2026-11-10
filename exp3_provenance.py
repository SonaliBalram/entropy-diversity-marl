import hashlib, os, platform, subprocess, sys, time

FILES = [
    "exp3_ladder.py", "exp3_outcome.py", "token_diversity.py",
    "environment.py", "maddpg_system.py", "replay_buffer.py",
    "models/actor.py", "models/critic.py", "exp3/train_exp3.py",
]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()[:16]

def main():
    arms = sys.argv[1:] or ["baseline", "beta_3"]
    W = 72
    print("=" * W)
    print("EXPERIMENT 3 -- HELD-OUT EVALUATION LOG")
    print("=" * W)
    print(f"  generated        : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  working directory: {os.getcwd()}")
    print(f"  platform         : {platform.platform()}  {platform.machine()}")
    print(f"  python           : {sys.version.split()[0]}")
    for mod in ["torch", "numpy", "scipy", "gymnasium", "pettingzoo", "mpe2"]:
        try:
            m = __import__(mod)
            print(f"  {mod:17s}: {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"  {mod:17s}: NOT IMPORTABLE ({e.__class__.__name__})")

    print()
    print("-" * W)
    print("CODE FINGERPRINTS  (sha256, first 16 hex)")
    print("-" * W)
    print("  Any change to these files changes the numbers below.")
    for f in FILES:
        if os.path.exists(f):
            print(f"  {sha(f)}  {f}")
        else:
            print(f"  {'MISSING':16s}  {f}")

    print()
    print("-" * W)
    print("EVALUATION PROTOCOL")
    print("-" * W)
    print("  held-out layouts : 200, seeds 900,000,000 .. 900,000,199")
    print("  disjoint from    : training, which uses SEED*1,000,000 + episode")
    print("  policy           : DETERMINISTIC (actor mean, no sampling)")
    print("  measure          : D = H(f_bar) - mean_i H(f_i), bits")
    print("                     tokens = nearest landmark, from obs[4:10]")
    print("                     bound  = log2 min(N,|A|) = 1.5850 bits")
    print("  empirical zero   : permutation null, 25 shuffles per layout")
    print("  statistic        : matched-return Delta D, reported under")
    print("                     branch = ascending (primary, pre-registered)")
    print("                     branch = any        (declared secondary)")

    print()
    print("-" * W)
    print("CHECKPOINT INVENTORY")
    print("-" * W)
    for arm in arms:
        root = os.path.join("checkpoints", arm)
        if not os.path.isdir(root):
            print(f"  {arm}: NOT PRESENT")
            continue
        for seed in [42, 128, 456, 789]:
            d = os.path.join(root, f"seed_{seed}")
            if not os.path.isdir(d):
                continue
            ck = sorted(f for f in os.listdir(d) if f.startswith("ckpt_ep"))
            best = os.path.join(d, "best.pt")
            mt = time.strftime("%Y-%m-%d %H:%M",
                               time.localtime(os.path.getmtime(best))) \
                if os.path.exists(best) else "no best.pt"
            print(f"  {arm:10s} seed {seed:>3}: {len(ck):>3} rungs, "
                  f"best.pt written {mt}")
    print()

if __name__ == "__main__":
    main()
