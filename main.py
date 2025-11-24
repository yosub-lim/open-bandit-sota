import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import inspect

from sklearn.linear_model import LogisticRegression

from obp.dataset import OpenBanditDataset
from obp.policy import (
    Random,
    EpsilonGreedy,
    BernoulliTS,
    LinUCB,
    LinTS,
)
from obp.ope import (
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    DoublyRobust,
    SwitchDoublyRobust,
    SubGaussianDoublyRobust,
    DoublyRobustTuning,
    RegressionModel,
)

plt.style.use("ggplot")


# =========================================
# 0) Discounted LinUCB (직접 구현)
# =========================================
class DiscountedLinUCB:
    def __init__(self, dim, n_actions, len_list=1, alpha=1.0, gamma=0.99):
        self.dim = dim
        self.n_actions = n_actions
        self.len_list = len_list
        self.alpha = alpha
        self.gamma = gamma

        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]

        self.policy_name = f"dLinUCB(gamma={gamma})"
        self.policy_type = "contextual"

    def update_params(self, action, reward, context):
        x = context.reshape(-1)
        self.A[action] = self.gamma * self.A[action] + np.outer(x, x)
        self.b[action] = self.gamma * self.b[action] + reward * x

    def select_action(self, context):
        x = context.reshape(-1)
        scores = []

        # 작은 정규화(ridge) 항 – 수치 안정성용
        eps = 1e-6

        for a in range(self.n_actions):
            A = self.A[a]
            A_reg = A + eps * np.eye(self.dim)

            # 역행렬이 안 되면 pseudo-inverse로 fallback
            try:
                A_inv = np.linalg.inv(A_reg)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A_reg)

            theta = A_inv @ self.b[a]
            # 혹시 부동소수점 때문에 음수 나오면 0으로 클리핑
            var = x @ A_inv @ x
            var = max(var, 0.0)
            ucb = self.alpha * np.sqrt(var)

            scores.append(theta @ x + ucb)

        a = int(np.argmax(scores))
        return np.array([a], dtype=int)


# =========================================
# 1) 데이터 로딩 (OBD)
# =========================================
def load_obd():
    dataset = OpenBanditDataset(
        behavior_policy="random",
        campaign="all",
        data_path="/Users/jeongjun-u/Downloads/open_bandit_dataset",
    )
    fb = dataset.obtain_batch_bandit_feedback()
    print(f"[INFO] rounds={fb['n_rounds']}  actions={fb['n_actions']}")
    fb["position"] = np.zeros_like(fb["action"], dtype=int)
    return dataset, fb


# =========================================
# 2) offline 학습
# =========================================
def train_policy(policy, fb):
    X, A, R = fb["context"], fb["action"], fb["reward"]
    name = getattr(policy, "policy_name", type(policy).__name__)
    print(f"[TRAIN] {name}")

    # update_params 시그니처를 보고 context 사용 여부 판단
    upd_sig = inspect.signature(policy.update_params)
    # bound method라 self 빠지고 남은 파라미터 개수 기준
    # context-free: (action, reward) -> 2개
    # contextual: (action, reward, context) -> 3개
    accepts_context = (len(upd_sig.parameters) == 3)

    for x, a, r in tqdm(zip(X, A, R), total=len(R), leave=False):
        if accepts_context:
            policy.update_params(
                action=int(a),
                reward=float(r),
                context=x.reshape(1, -1),
            )
        else:
            policy.update_params(
                action=int(a),
                reward=float(r),
            )
    return policy


# =========================================
# 3) action_dist 계산
# =========================================
def compute_action_dist(policy, X, n_actions):
    n = X.shape[0]
    dist = np.zeros((n, n_actions, 1))
    name = getattr(policy, "policy_name", type(policy).__name__)
    print(f"[ACTION_DIST] {name}")

    # select_action 시그니처로 context 필요 여부 판단
    sel_sig = inspect.signature(policy.select_action)
    # context-free: () -> 0개
    # contextual: (context) -> 1개
    takes_context = (len(sel_sig.parameters) == 1)

    for i in tqdm(range(n), leave=False):
        if takes_context:
            a = int(policy.select_action(X[i].reshape(1, -1))[0])
        else:
            a = int(policy.select_action()[0])
        dist[i, a, 0] = 1.0

    return dist


# =========================================
# 4) 보상 모델 (RegressionModel) — DR/MRDR용
# =========================================
def build_reward_model(fb, n_actions):
    print("[REWARD MODEL] LogisticRegression per-action")
    model = RegressionModel(
        n_actions=n_actions,
        len_list=1,
        base_model=LogisticRegression(max_iter=500),
    )

    est = model.fit_predict(
        context=fb["context"],
        action=fb["action"],
        reward=fb["reward"],
        position=fb["position"],
        pscore=fb.get("pscore", None),
        n_folds=1,
        random_state=12345,
    )

    print("  est_rewards shape:", est.shape)
    return est


# =========================================
# 5) OPE (IPW, DR, Switch-DR, MRDR)
# =========================================
def run_ope(fb, action_dist, reward_model):
    ope = OffPolicyEvaluation(
        fb,
        ope_estimators=[
            InverseProbabilityWeighting(),
            DoublyRobust(),
            SwitchDoublyRobust(),
            SubGaussianDoublyRobust(),
            DoublyRobustTuning(
                lambdas=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                tuning_method="slope",
                estimator_name="mrdr",
            ),
        ],
    )

    values = ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=reward_model,
    )

    return values


# =========================================
# 6) 플롯: 정책별 OPE 결과 비교
# =========================================
def plot_ope_bar(all_values, behavior_mean):
    plt.rcParams.update({
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    estimators = list(next(iter(all_values.values())).keys())
    policies = list(all_values.keys())

    show_keys = [k for k in estimators if k.lower() in ["dr", "switch-dr", "mrdr"]]

    for key in show_keys:
        plt.figure(figsize=(7, 4))
        vals = [all_values[p][key] for p in policies]
        plt.bar(policies, vals)
        plt.axhline(behavior_mean, linestyle="--", color="black", label="Behavior (Random) mean")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Estimated policy value")
        plt.title(f"OPE Estimates ({key})")
        plt.legend()
        plt.tight_layout()
        fname = f"ope_{key.replace(' ', '_')}.png"
        plt.savefig(fname)
        plt.close()
        print(f"[PLOT] Saved {fname}")


# =========================================
# 7) 메인 실험 실행
# =========================================
def main():
    total_start = time()

    dataset, fb = load_obd()
    X = fb["context"]
    n_actions = fb["n_actions"]

    behavior_mean = fb["reward"].mean()
    print(f"\n[Behavior (Random) avg reward] {behavior_mean:.6f}\n")

    dim = X.shape[1]

    policies = {
        "Random": Random(n_actions=n_actions, len_list=1),
        "EpsGreedy": EpsilonGreedy(n_actions=n_actions, len_list=1, epsilon=0.1),
        "BernoulliTS": BernoulliTS(n_actions=n_actions, len_list=1),
        "LinUCB": LinUCB(dim=dim, n_actions=n_actions, len_list=1),
        "LinTS": LinTS(dim=dim, n_actions=n_actions, len_list=1),
        "dLinUCB": DiscountedLinUCB(dim=dim, n_actions=n_actions, len_list=1, alpha=1.0, gamma=0.99),
    }

    trained = {}
    action_dists = {}
    all_values = {}

    reward_model = build_reward_model(fb, n_actions=n_actions)

    for name, pol in policies.items():
        t0 = time()
        pol = train_policy(pol, fb)
        trained[name] = pol
        print(f"  -> train done in {time() - t0:.2f} sec")

        dist = compute_action_dist(pol, X, n_actions)
        action_dists[name] = dist

        vals = run_ope(fb, dist, reward_model)
        all_values[name] = vals

        print(f"[OPE] {name}")
        for est_name, v in vals.items():
            print(f"   {est_name:<20} {v:.6f}")
        print()

    plot_ope_bar(all_values, behavior_mean)

    print(f"\n[TOTAL] {time() - total_start:.2f} sec")
    print("Done. Check ope_*.png files.")


if __name__ == "__main__":
    main()