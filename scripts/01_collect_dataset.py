import os
import numpy as np
from src.sim.simulator.simulator import Simulator
from src.sim.backend_dynamics.kbm import KBM
from src.sim.terrain.world import MathExpressionWorld


def run_simple_collection(num_steps: int = 200, dt: float = 0.05, seed: int = 42):
    """Simple verification script using our World object architecture."""
    np.random.seed(seed)

    world = MathExpressionWorld(expression="0.3 * sin(0.5 * x) + 0.3 * cos(0.5 * y)")
    backend = KBM(wheelbase=0.33, drag_coeff=0.1)
    sim = Simulator(backend=backend, world=world, dt=dt)


    initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # v = 1.0 m/s
    state = sim.reset(initial_state)

    states = [state]
    actions = []
    print(f"Starting rollout for {num_steps} steps...")
    for step in range(num_steps):
        accel = np.random.uniform(-0.5, 0.5)
        steering = np.random.uniform(-0.3, 0.3)
        action = np.array([accel, steering], dtype=np.float64)

        next_state = sim.step(action)

        states.append(next_state)
        actions.append(action)


    states_arr = np.array(states)    
    actions_arr = np.array(actions) 

    # 6. Save dataset
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "test_trajectory.npz")

    # np.savez_compressed(
    #     save_path,
    #     states=states_arr,
    #     actions=actions_arr,
    #     dt=dt
    # )

    # Verification readouts
    print("\n--- Verification Summary ---")
    print(f"Saved dataset to: {save_path}")
    print(f"States shape:  {states_arr.shape}  [x, y, yaw, v, roll, pitch, w]")
    print(f"Actions shape: {actions_arr.shape} [accel, steering]")
    print(f"Final Position (x, y): ({states_arr[-1, 0]:.2f}, {states_arr[-1, 1]:.2f})")
    print(f"Final Pitch/Roll:      pitch={states_arr[-1, 5]:.3f} rad, roll={states_arr[-1, 4]:.3f} rad")


if __name__ == "__main__":
    run_simple_collection()