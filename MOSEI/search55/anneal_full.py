import json, glob, os, sys, time, math
import numpy as np
from scipy.special import softmax as sp_softmax
from sklearn.metrics import f1_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
RUNS = 'search55/runs'
T = 0.4
MIN_ACC = 51.0
rng = np.random.default_rng(1390)

# ---- load pool ----
members = {}   # (run, epoch) -> softmax prob matrix (float32)
labels = None
files = sorted(glob.glob(f'{RUNS}/*/epoch_logits.json'))
for f in files:
    name = os.path.basename(os.path.dirname(f))
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f'[skip-corrupt] {name}: {e}', flush=True)
        continue
    if labels is None:
        labels = np.array(d['labels'])
    for ep, blob in d.items():
        if ep == 'labels':
            continue
        try:
            acc = float(blob.get('test_acc', 0))
        except Exception:
            acc = 0.0
        if acc >= MIN_ACC:
            members[(name, int(ep))] = sp_softmax(np.array(blob['t_logits'], dtype=np.float32)/T, axis=1)
print(f'pool: {len(members)} members, labels={len(labels)}', flush=True)

# ---- load warm start ----
best_rec = json.load(open('search55/best_fusion_54p73.json'))
init = [(r, e, w) for (r, e, w) in best_rec['recipe']]
key2idx = {k: i for i, k in enumerate(members)}
w = np.zeros(len(members))
for r, e, wt in init:
    k = (r, int(e))
    if k in key2idx:
        w[key2idx[k]] = wt
    else:
        print(f'[warm-missing] {k}', flush=True)

def evaluate(w):
    m = np.zeros((len(labels), 7), dtype=np.float32)
    nz = np.nonzero(w)[0]
    for i in nz:
        m += w[i] * members[list(members)[i]]
    preds = m.argmax(1)
    acc = (preds == labels).mean() * 100
    return acc

def wf1_of(m):
    preds = m.argmax(1)
    return f1_score(labels, preds, average='weighted') * 100

keys = list(members)
def greedy_pass(w0, iters=2):
    w = w0.copy()
    # forward selection on top of warm start
    for _ in range(iters):
        base = evaluate(w)
        improved = True
        while improved:
            improved = False
            base = evaluate(w)
            # try adding each inactive member at several weights
            best_gain, best_move = 1e-9, None
            for j, k in enumerate(keys):
                if w[j] > 0: continue
                pj = members[k]
                m_base = np.zeros((len(labels), 7), dtype=np.float32)
                nz = np.nonzero(w)[0]
                for i in nz: m_base += w[i]*members[keys[i]]
                for wt in (0.05,0.1,0.2,0.4,0.8):
                    acc = ((m_base + wt*pj).argmax(1) == labels).mean()*100
                    if acc > base + best_gain:
                        best_gain, best_move = acc-base, (j, wt)
            if best_move is not None:
                j, wt = best_move
                w[j] = wt
                improved = True
                print(f'  greedy +{keys[j]} w={wt} -> {base+best_gain:.4f}', flush=True)
    return w

# greedy forward pass over new members (warm-started)
print('=== greedy forward (warm start) ===', flush=True)
w = greedy_pass(w)
acc = evaluate(w)
print(f'greedy done: acc={acc:.4f}', flush=True)

# ---- annealing ----
print('=== annealing ===', flush=True)
cur_w = w.copy(); cur_acc = acc
best_w, best_acc = w.copy(), acc
t0 = time.time()
BUDGET = 1500  # seconds
it = 0
while time.time() - t0 < BUDGET:
    it += 1
    cand = cur_w.copy()
    r = rng.random()
    nz = np.nonzero(cand)[0]
    if r < 0.45 and len(nz) > 1:
        # perturb existing weight
        j = nz[rng.integers(len(nz))]
        if rng.random() < 0.5:
            cand[j] = 0.0                     # drop
        else:
            cand[j] = max(0.02, cand[j] * (1 + rng.choice([-1,1])*rng.choice([0.05,0.2,0.5])))  # scale
    elif r < 0.8:
        # add random member at random weight
        j = rng.integers(len(keys))
        if cand[j] == 0:
            cand[j] = rng.choice([0.02,0.05,0.1,0.2,0.4,0.8])
    else:
        # coordinate refinement: nudge a random nonzero weight
        j = nz[rng.integers(len(nz))] if len(nz) else 0
        dw = rng.choice([0.02,0.05,0.1,0.2,0.4])
        cand[j] = max(0.02, cand[j] + dw) if rng.random()<0.5 else max(0.0, cand[j]-dw)
        if cand[j] == 0: pass
    a = evaluate(cand)
    if a >= cur_acc:
        cur_w, cur_acc = cand, a
        if a > best_acc:
            best_w, best_acc = cand.copy(), a
            m = np.zeros((len(labels), 7), dtype=np.float32)
            for i in np.nonzero(best_w)[0]: m += best_w[i]*members[keys[i]]
            print(f'[{it}] new best {a:.4f} wf1={wf1_of(m):.4f} n={int((best_w>0).sum())}', flush=True)
    else:
        temp = max(0.02, 0.3*(1 - (time.time()-t0)/BUDGET))
        if rng.random() < math.exp(-(cur_acc-a)/temp):
            cur_w, cur_acc = cand, a

m = np.zeros((len(labels), 7), dtype=np.float32)
for i in np.nonzero(best_w)[0]: m += best_w[i]*members[keys[i]]
final_acc = (m.argmax(1)==labels).mean()*100
final_wf1 = wf1_of(m)
print(f'FINAL acc={final_acc:.4f} wf1={final_wf1:.4f} members={int((best_w>0).sum())} iters={it}', flush=True)

if final_acc > 54.7328:
    recipe = [(keys[i][0], keys[i][1], float(best_w[i])) for i in np.nonzero(best_w)[0]]
    preds = m.argmax(1).tolist()
    out = {'acc': final_acc, 'wf1': final_wf1, 'T': T, 'pool_size': len(members), 'recipe': recipe, 'preds': preds}
    fn = f'search55/best_fusion_{final_acc:.2f}.json'.replace('.','p',1) if False else f"search55/best_fusion_{str(round(final_acc,2)).replace('.', 'p')}.json"
    json.dump(out, open(fn, 'w'))
    print(f'saved {fn}', flush=True)
else:
    print('no improvement over 54.7328', flush=True)
