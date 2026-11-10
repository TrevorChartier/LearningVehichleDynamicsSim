"""TODO Still: Factor out configurations into a config file (root/configs).
Have a script entry point under root/scripts/ parse config and run this pipeline.
Print out some progress as it's running.
Automatically detect when names are the same and print an error to force new names. This should also make a new directory for it.
    As in don't let datasets get overwritten.
Sampled parameters should get rounded to lower precision rather than long decimals.
"""
import json
from datetime import datetime, timezone
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

from policies.smooth_noise_policy import SmoothNoisePolicy
from sim.backend_dynamics.kbm import KBM
from sim.simulator.simulator import Simulator
from sim.terrain.world import BaseWorld
from sim.vehicles.vehicle import BaseVehicle, KBMVehicle
from sim.terrain.world import ParameterizedSineWorld 

class DatasetGenerator:
    def __init__(self, output_dir: str, dt: float = 0.05):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dt = dt

    def collect_rollout(self, vehicle: BaseVehicle, world: BaseWorld, steps: int) -> dict:
        backend = KBM(vehicle=vehicle)
        sim = Simulator(backend=backend, world=world, dt=self.dt)
        policy = SmoothNoisePolicy(dt=self.dt)
        
        state = sim.reset()
        
        states = np.zeros((steps, 7))
        actions = np.zeros((steps, 2))
        next_states = np.zeros((steps, 7))
        
        for i in range(steps):
            action = policy.get_action()
            next_state = sim.step(action)
            
            states[i] = state
            actions[i] = action
            next_states[i] = next_state
            state = next_state
            
        v_current = states[:, 3]
        yaw_current = states[:, 2]
        v_next = next_states[:, 3]
        yaw_next = next_states[:, 2]
        
        delta_yaw = np.arctan2(np.sin(yaw_next - yaw_current), np.cos(yaw_next - yaw_current))
        
        return {
            "x": states[:, 0],
            "y": states[:, 1],
            "yaw": states[:, 2],
            "v": v_current,
            "accel": actions[:, 0],
            "steering": actions[:, 1],
            "delta_v": v_next - v_current,
            "delta_yaw": delta_yaw,
            "ground_truth_pitch": states[:, 5],
            "ground_truth_drag": np.full(steps, vehicle.drag_coeff),
            "ground_truth_wheelbase": np.full(steps, vehicle.wheelbase)
        }
        
    def generate_dataset(self, configurations: list, rollouts_per_config: int, steps_per_rollout: int):
        keys = ["timestep", "rollout_id", "vehicle_dist_id", "vehicle_id", 
                "world_dist_id", "world_id", "x", "y", "yaw", "v", 
                "accel", "steering", "delta_v", "delta_yaw", "ground_truth_pitch", 
                "ground_truth_drag", "ground_truth_wheelbase"]
        all_data = {k: [] for k in keys}
        
        current_rollout_id = 0
        total_expected_rollouts = len(configurations) * rollouts_per_config
        
        # Wrap the generation process in a tqdm progress bar
        with tqdm(total=total_expected_rollouts, desc="Generating Rollouts", unit="rollout") as pbar:
            for config in configurations:
                vehicle = config["vehicle"]
                world = config["world"]
                
                for _ in range(rollouts_per_config):
                    # 1. Get pure physics data
                    rollout_data = self.collect_rollout(vehicle=vehicle, world=world, steps=steps_per_rollout)
                    
                    # 2. Stamp metadata at the dataset level
                    rollout_data["timestep"] = np.arange(steps_per_rollout)
                    rollout_data["rollout_id"] = np.full(steps_per_rollout, current_rollout_id)
                    
                    rollout_data["vehicle_dist_id"] = np.full(steps_per_rollout, config["vehicle_dist_id"])
                    rollout_data["vehicle_id"] = np.full(steps_per_rollout, config["vehicle_id"])
                    
                    rollout_data["world_dist_id"] = np.full(steps_per_rollout, config["world_dist_id"])
                    rollout_data["world_id"] = np.full(steps_per_rollout, config["world_id"])
                    
                    # 3. Append to master lists
                    for k in keys:
                        all_data[k].append(rollout_data[k])
                        
                    current_rollout_id += 1
                    pbar.update(1)  # Advance the progress bar by 1 rollout
                    
        print("\nConcatenating and saving dataset... ", end="", flush=True)
        for k in keys:
            all_data[k] = np.concatenate(all_data[k])
            
        np.savez_compressed(self.output_dir / "monolithic_dataset.npz", **all_data)
        print(f"Done! Saved to {self.output_dir / 'monolithic_dataset.npz'}")
        
        return len(all_data['v']), current_rollout_id, keys

if __name__ == "__main__":
    DT = 0.05
    STEPS_PER_ROLLOUT = 1500
    DATA_DIR_NAME = "data/raw/testing_new_dataset_pipeline"
    ROLLOUTS_PER_CONFIG = 20

    vehicle_dists = {
        0: {"wheelbase_range": [0.30, 0.35], "drag_range": [0.08, 0.12], "num_samples": 2},
        1: {"wheelbase_range": [0.40, 0.50], "drag_range": [0.25, 0.35], "num_samples": 3},
    }
    world_dists = {
        0: {"amp_range": [0.1, 0.3], "fx_range": [0.4, 0.6], "fy_range": [0.4, 0.6], "num_samples": 2},
        1: {"amp_range": [0.4, 0.6], "fx_range": [0.1, 0.3], "fy_range": [0.1, 0.3], "num_samples": 2},
    }

    sampled_vehicles = {}
    v_id_counter = 0
    for dist_id, params in vehicle_dists.items():
        for _ in range(params["num_samples"]):
            wb = np.random.uniform(*params["wheelbase_range"])
            drag = np.random.uniform(*params["drag_range"])
            sampled_vehicles[v_id_counter] = {
                "dist_id": dist_id,
                "instance": KBMVehicle(wheelbase=wb, drag_coeff=drag)
            }
            v_id_counter += 1

    sampled_worlds = {}
    w_id_counter = 0
    for dist_id, params in world_dists.items():
        for _ in range(params["num_samples"]):
            amp = np.random.uniform(*params["amp_range"])
            fx = np.random.uniform(*params["fx_range"])
            fy = np.random.uniform(*params["fy_range"])
            sampled_worlds[w_id_counter] = {
                "dist_id": dist_id,
                "instance": ParameterizedSineWorld(amplitude=amp, freq_x=fx, freq_y=fy)
            }
            w_id_counter += 1
    
    # 5. Create combinations to run
    combinations = []
    for v_id, v_info in sampled_vehicles.items():
        for w_id, w_info in sampled_worlds.items():
            combinations.append({
                "vehicle_dist_id": v_info["dist_id"],
                "vehicle_id": v_id,
                "vehicle": v_info["instance"],
                "world_dist_id": w_info["dist_id"],
                "world_id": w_id,
                "world": w_info["instance"]
            })
            
    generator = DatasetGenerator(DATA_DIR_NAME, dt=DT)
    total_samples, total_rollouts, feature_keys = generator.generate_dataset(
        configurations=combinations, 
        rollouts_per_config=ROLLOUTS_PER_CONFIG, 
        steps_per_rollout=STEPS_PER_ROLLOUT
    )
    
    # 6. Generate Dataset Card with explicit mappings for distributions AND specific instances
    card = {
        "metadata": {
            "dataset_name": "PoC Dataset 2",
            "description": "Small PoC Dataset Over Sampled Distributions of Terrains and Vehicles",
            "author": "Trevor Chartier",
            "creation_timestamp": datetime.now().astimezone().strftime("%m_%d_%Y__%I_%M_%p_%Z"),
            "version": "1.1"
        },
        "simulation_params": {
            "backend_dynamics": "KBM",
            "dt_seconds": DT,
            "steps_per_rollout": STEPS_PER_ROLLOUT,
            "rollout_duration_seconds": STEPS_PER_ROLLOUT * DT,
            "total_rollouts": int(total_rollouts),
            "total_samples": int(total_samples)
        },
        "collection_policy": {
            "type": "SmoothNoisePolicy",
            "description": "Temporally correlated noise using sum of sine waves to prevent instantaneous gradient averaging and excite lateral dynamics.",
            "params": {
                "max_accel": 3.0,
                "max_steer": 0.5,
                "dt": DT
            }
        },
        "mappings": {
            "vehicle_distributions": vehicle_dists,
            "world_distributions": world_dists,
            "sampled_vehicles_exact": {
                v_id: {
                    "from_dist_id": v_info["dist_id"],
                    "wheelbase": v_info["instance"].wheelbase, 
                    "drag_coeff": v_info["instance"].drag_coeff
                } for v_id, v_info in sampled_vehicles.items()
            },
            "sampled_worlds_exact": {
                w_id: {
                    "from_dist_id": w_info["dist_id"],
                    "type": "ParameterizedSineWorld", 
                    "amplitude": w_info["instance"].amp, 
                    "freq_x": w_info["instance"].fx, 
                    "freq_y": w_info["instance"].fy
                } for w_id, w_info in sampled_worlds.items()
            }
        },
        "schema": {
            "metadata_columns": ["timestep", "rollout_id", "vehicle_dist_id", "vehicle_id", "world_dist_id", "world_id"],
            "global_states": ["x", "y", "yaw"],
            "observables": ["v", "accel", "steering"],
            "targets": ["delta_v", "delta_yaw"],
            "ground_truths": ["ground_truth_pitch", "ground_truth_drag", "ground_truth_wheelbase"],
            "raw_features_list": feature_keys
        }
    }
    
    with open(os.path.join(DATA_DIR_NAME, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=4)