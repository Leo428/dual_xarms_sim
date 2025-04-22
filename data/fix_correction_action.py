import numpy as np

from dual_xarms_sim.utils.transformation import construct_adjoint_matrix

file_name = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318/sim_dual_xarms_double_insert_human_intervention_20250318_174808.npz"
data   = np.load(file_name, allow_pickle=True)
obses  = data["obses"] # list of dicts, each has state["<side>/tcp_pose"]
actions= data["actions"] # list of 14‑dim arrays (corrupted)
infos  = data["infos"] # list of dicts (to detect interventions)

fixed_actions = []
for obs, act, info in zip(obses, actions, infos):
    # 1) grab the base‑frame poses you saved
    left_pose  = obs["state"]["left/tcp_pose"]   # shape (7,)
    right_pose = obs["state"]["right/tcp_pose"]  # shape (7,)

    # 2) build the two 6×6 adjoints
    A_l = construct_adjoint_matrix(left_pose)
    A_r = construct_adjoint_matrix(right_pose)

    # 3) if this was a human‑intervene step, you already recorded a wrist‑frame action
    if "og_intervene_action" in info:
        fixed_actions.append(act.copy())
        continue

    # 4) otherwise, undo the bug: map the saved absolute → wrist
    rel = act.copy()
    rel[:6]    = np.linalg.inv(A_l) @ act[:6]
    rel[7:13]  = np.linalg.inv(A_r) @ act[7:13]
    # gripper dims (6 and 13) stayed the same under the original wrapper
    fixed_actions.append(rel)

fixed_file_name = "/home/huzheyuan/dual_xarms/dual_xarms_sim/data/sim_double_insert_zheyuan_correction_0318_fixed/fixed_sim_dual_xarms_double_insert_human_intervention_20250318_174808.npz"
np.savez(
    fixed_file_name,
    obses=obses,
    actions=fixed_actions,
    rews=data["rews"],
    dones=data["dones"],
    truncateds=data["truncateds"],
    infos=infos,
    allow_pickle=True,
)
