import os
import sys

# TODO: this is a trick to avoid infinite warnings
# but should be removed in the future
sys.stderr = open(os.devnull, "w")

# ruff: noqa: E402
import time

from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from tensordict import TensorDict
from tqdm.auto import tqdm

from routefinder.baselines.solve import solve
from routefinder.envs.mtdvrp import MTVRPEnv

# Size to solving time as in paper (seconds)
size_to_time = {
    5: 10,
    6: 10,
    7: 10,
    10: 10,
    15: 10,
    20: 10,
    50: 10,
    100: 20,
}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_procs", type=int, default=16)
    parser.add_argument("--solver", type=str, default="pyvrp")
    args = parser.parse_args()
    num_procs = args.num_procs
    solver = args.solver

    print("Writing output to", f"output_{solver}.txt")
    # sys.stdout = open(f"output_{solver}.txt", "w")

    print("Solver:", solver, "Num procs:", num_procs)

    env = MTVRPEnv(check_solution=False)
    data_files = []
    for root, dirs, files in os.walk("dataset_mini"):
        for file in files:
            if file == "50.npz" or file == "100.npz" or file == "15.npz" or file == "20.npz" or file == "10.npz" or file == "5.npz" or file == "6.npz" or file == "7.npz":
                if file.split('.')[0] + '_sol_' + solver + '.npz' in files:
                    print("Already solved", file)
                    continue
                data_files.append(os.path.join(root, file))
    # sort data files by length of name
    data_files = sorted(data_files, key=lambda x: len(x))
    # sort by "val" or "test"
    data_files = sorted(data_files, key=lambda x: x.split("/")[-2])
    # sort by size
    data_files = sorted(data_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    # print('here1')
    # Go through the files, and run the solver
    for file in tqdm(data_files, desc="Generating dataset solutions with " + solver):
        td_test = load_npz_to_tensordict(file)
        num_problems, size = td_test["demand_linehaul"].shape
        max_runtime = size_to_time[size]
        print(42 * "=" + "\nProcessing", file, "...")
        print(f"Estimated time : {size_to_time[size] * num_problems / num_procs:.3f} s")

        start = time.time()
        # print('here2')
        # Main solver
        td_test = env.reset(td_test)
        actions_solver, costs_solver = solve(
            td_test, max_runtime=max_runtime, num_procs=num_procs, solver=solver
        )
        # print('here3')
        rewards_solver = env.get_reward(td_test.clone(), actions_solver)

        total_time = time.time() - start

        print(f"Time: {total_time:.3f} s")
        print(f"Average cost: {-rewards_solver.mean():.3f}")

        out = TensorDict(
            {
                "actions": actions_solver,
                "costs": costs_solver,
                "time": total_time,
            },
            batch_size=[],
        )
        # print('here')
        save_tensordict_to_npz(out, file.replace(".npz", f"_sol_{solver}.npz"))
