import json
from datetime import datetime, timezone
import os
from pathlib import Path
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
        # Initialize dictionary keys based on physics + metadata
        keys = ["timestep", "rollout_id", "vehicle_id", "world_id", "x", "y", "yaw", "v", 
                "accel", "steering", "delta_v", "delta_yaw", "ground_truth_pitch", 
                "ground_truth_drag", "ground_truth_wheelbase"]
        all_data = {k: [] for k in keys}
        
        current_rollout_id = 0
        
        for config in configurations:
            vehicle = config["vehicle"]
            world = config["world"]
            
            for _ in range(rollouts_per_config):
                # 1. Get pure physics data
                rollout_data = self.collect_rollout(vehicle=vehicle, world=world, steps=steps_per_rollout)
                
                # 2. Stamp metadata at the dataset level
                rollout_data["timestep"] = np.arange(steps_per_rollout)
                rollout_data["rollout_id"] = np.full(steps_per_rollout, current_rollout_id)
                rollout_data["vehicle_id"] = np.full(steps_per_rollout, config["vehicle_id"])
                rollout_data["world_id"] = np.full(steps_per_rollout, config["world_id"])
                
                # 3. Append to master lists
                for k in keys:
                    all_data[k].append(rollout_data[k])
                    
                current_rollout_id += 1
                    
        for k in keys:
            all_data[k] = np.concatenate(all_data[k])
            
        np.savez_compressed(self.output_dir / "monolithic_dataset.npz", **all_data)
        return len(all_data['v']), current_rollout_id, keys

if __name__ == "__main__":
    DT = 0.05
    STEPS_PER_ROLLOUT = 1500
    DATA_DIR_NAME = "data/raw/testing_new_dataset_pipeline"
    ROLLOUTS_PER_CONFIG = 20
    
    # 1. Define Vehicles mapping
    vehicles = {
        0: KBMVehicle(wheelbase=0.33, drag_coeff=0.1),
        1: KBMVehicle(wheelbase=0.45, drag_coeff=0.3),
        2: KBMVehicle(wheelbase=0.25, drag_coeff=0.5)
    }
    
    # 2. Define Worlds mapping
    worlds = {
        0: ParameterizedSineWorld(amplitude=0.2, freq_x=0.5, freq_y=0.5),
        1: ParameterizedSineWorld(amplitude=0.5, freq_x=0.2, freq_y=0.2),
        2: ParameterizedSineWorld(amplitude=0.4, freq_x=0.8, freq_y=0.8)
    }
    
    # 3. Create combinations to run
    combinations = []
    for v_id, vehicle in vehicles.items():
        for w_id, world in worlds.items():
            combinations.append({
                "vehicle_id": v_id,
                "vehicle": vehicle,
                "world_id": w_id,
                "world": world
            })
            
    generator = DatasetGenerator(DATA_DIR_NAME, dt=DT)
    total_samples, total_rollouts, feature_keys = generator.generate_dataset(
        configurations=combinations, 
        rollouts_per_config=ROLLOUTS_PER_CONFIG, 
        steps_per_rollout=STEPS_PER_ROLLOUT
    )
    
    # 4. Generate Dataset Card with explicit mappings
    card = {
        "metadata": {
            "dataset_name": "PoC Dataset",
            "description": "Small PoC Dataset Over Various Terrains and Vehicle Types",
            "author": "Trevor Chartier",
            "creation_timestamp": datetime.now().astimezone().strftime("%m_%d_%Y__%I_%M_%p_%Z"),
            "version": "1.0"
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
                "max_accel": 3.0,  # Match these to your actual SmoothNoisePolicy defaults
                "max_steer": 0.5,
                "dt": DT
            }
        },
        "mappings": {
            "vehicles": {
                v_id: {"wheelbase": v.wheelbase, "drag_coeff": v.drag_coeff}
                for v_id, v in vehicles.items()
            },
            "worlds": {
                w_id: {
                    "type": "ParameterizedSineWorld", 
                    "amplitude": w.amp, 
                    "freq_x": w.fx, 
                    "freq_y": w.fy
                } for w_id, w in worlds.items()
            }
        },
        "schema": {
            "metadata_columns": ["timestep", "rollout_id", "vehicle_id", "world_id"],
            "global_states": ["x", "y", "yaw"],
            "observables": ["v", "accel", "steering"],
            "targets": ["delta_v", "delta_yaw"],
            "ground_truths": ["ground_truth_pitch", "ground_truth_drag", "ground_truth_wheelbase"],
            "raw_features_list": feature_keys
        }
    }
    
    with open(os.path.join(DATA_DIR_NAME, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=4)