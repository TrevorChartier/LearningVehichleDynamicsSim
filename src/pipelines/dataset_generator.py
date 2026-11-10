"""
Sampled parameters should get rounded to lower precision rather than long decimals.
"""
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from policies.smooth_noise_policy import SmoothNoisePolicy
from sim.backend_dynamics.kbm import KBM
from sim.simulator.simulator import Simulator
from sim.terrain.world import BaseWorld, ParameterizedSineWorld
from sim.vehicles.vehicle import BaseVehicle, KBMVehicle

class DatasetGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.gen_params = config["generation_params"]
        self.metadata = config["metadata"]
        
        self.dt = self.gen_params["dt"]
        
        # Format the directory name (e.g. "poc_dataset_v1_1")
        safe_name = self.metadata["dataset_name"].replace(" ", "_").lower()
        safe_version = self.metadata["version"].replace(".", "_")
        folder_name = f"{safe_name}_v{safe_version}"
        
        self.output_dir = Path(self.gen_params["data_dir_name"]) / folder_name
        
        # Enforce overwrite protection
        if self.output_dir.exists():
            raise FileExistsError(
                f"Dataset directory already exists at: {self.output_dir}\n"
                f"Please update the 'version' or 'dataset_name' in your config file to prevent overwriting."
            )
            
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def _sample_distributions(self):
        """Orchestrates the sampling of concrete simulator objects from the config distributions."""
        sampled_vehicles = {}
        v_id_counter = 0
        for dist_id, params in self.config["vehicle_dists"].items():
            for _ in range(params["num_samples"]):
                wb = np.round(np.random.uniform(*params["wheelbase_range"]), 3)
                drag = np.round(np.random.uniform(*params["drag_range"]), 3)
                sampled_vehicles[v_id_counter] = {
                    "dist_id": int(dist_id),
                    "instance": KBMVehicle(wheelbase=wb, drag_coeff=drag)
                }
                v_id_counter += 1

        sampled_worlds = {}
        w_id_counter = 0
        for dist_id, params in self.config["world_dists"].items():
            for _ in range(params["num_samples"]):
                amp = np.round(np.random.uniform(*params["amp_range"]), 3)
                fx = np.round(np.random.uniform(*params["fx_range"]), 3)
                fy = np.round(np.random.uniform(*params["fy_range"]), 3)
                sampled_worlds[w_id_counter] = {
                    "dist_id": int(dist_id),
                    "instance": ParameterizedSineWorld(amplitude=amp, freq_x=fx, freq_y=fy)
                }
                w_id_counter += 1
                
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
                
        return combinations, sampled_vehicles, sampled_worlds

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
        
    def run_pipeline(self):
        """Main execution method that triggers sampling, collection, and saving."""
        print("Sampling distributions...")
        combinations, sampled_vehicles, sampled_worlds = self._sample_distributions()
        
        steps_per_rollout = self.gen_params["steps_per_rollout"]
        rollouts_per_config = self.gen_params["rollouts_per_config"]
        
        keys = ["timestep", "rollout_id", "vehicle_dist_id", "vehicle_id", 
                "world_dist_id", "world_id", "x", "y", "yaw", "v", 
                "accel", "steering", "delta_v", "delta_yaw", "ground_truth_pitch", 
                "ground_truth_drag", "ground_truth_wheelbase"]
        all_data = {k: [] for k in keys}
        
        current_rollout_id = 0
        total_expected_rollouts = len(combinations) * rollouts_per_config
        
        with tqdm(total=total_expected_rollouts, desc="Generating Rollouts", unit="rollout") as pbar:
            for config_combo in combinations:
                vehicle = config_combo["vehicle"]
                world = config_combo["world"]
                
                for _ in range(rollouts_per_config):
                    rollout_data = self.collect_rollout(vehicle=vehicle, world=world, steps=steps_per_rollout)
                    
                    rollout_data["timestep"] = np.arange(steps_per_rollout)
                    rollout_data["rollout_id"] = np.full(steps_per_rollout, current_rollout_id)
                    
                    rollout_data["vehicle_dist_id"] = np.full(steps_per_rollout, config_combo["vehicle_dist_id"])
                    rollout_data["vehicle_id"] = np.full(steps_per_rollout, config_combo["vehicle_id"])
                    
                    rollout_data["world_dist_id"] = np.full(steps_per_rollout, config_combo["world_dist_id"])
                    rollout_data["world_id"] = np.full(steps_per_rollout, config_combo["world_id"])
                    
                    for k in keys:
                        all_data[k].append(rollout_data[k])
                        
                    current_rollout_id += 1
                    pbar.update(1)
                    
        print("\nConcatenating and saving dataset... ", end="", flush=True)
        for k in keys:
            all_data[k] = np.concatenate(all_data[k])
            
        np.savez_compressed(self.output_dir / "dataset.npz", **all_data)
        print(f"Done! Saved to {self.output_dir / 'dataset.npz'}")
        
        total_samples = len(all_data['v'])
        
        self._save_dataset_card(
            sampled_vehicles=sampled_vehicles, 
            sampled_worlds=sampled_worlds, 
            total_samples=total_samples, 
            total_rollouts=current_rollout_id, 
            feature_keys=keys
        )
        print("Dataset card saved successfully.")

    def _save_dataset_card(self, sampled_vehicles: dict, sampled_worlds: dict, 
                           total_samples: int, total_rollouts: int, feature_keys: list):
        """Internal method to generate and save the dataset provenance."""
        card = {
            "metadata": self.config["metadata"],
            "simulation_params": {
                "backend_dynamics": "KBM",
                "dt_seconds": self.dt,
                "steps_per_rollout": self.gen_params["steps_per_rollout"],
                "rollout_duration_seconds": self.gen_params["steps_per_rollout"] * self.dt,
                "total_rollouts": int(total_rollouts),
                "total_samples": int(total_samples)
            },
            "collection_policy": {
                "type": "SmoothNoisePolicy",
                "description": "Temporally correlated noise using sum of sine waves to prevent instantaneous gradient averaging and excite lateral dynamics.",
                "params": {
                    "max_accel": 3.0,
                    "max_steer": 0.5,
                    "dt": self.dt
                }
            },
            "mappings": {
                "vehicle_distributions": self.config["vehicle_dists"],
                "world_distributions": self.config["world_dists"],
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
        
        # Add creation timestamp directly to metadata block dynamically
        card["metadata"]["creation_timestamp"] = datetime.now().astimezone().strftime("%m_%d_%Y__%I_%M_%p_%Z")
        
        with open(self.output_dir / "dataset_card.json", "w") as f:
            json.dump(card, f, indent=4)