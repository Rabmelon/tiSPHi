import argparse
import taichi as ti
from eng.simulation import Simulation, SimConfiger
from eng.ui_sim import ui_sim

if __name__ == "__main__":
    # * using parser
    parser = argparse.ArgumentParser(description='tiSPHi')
    parser.add_argument('--scene_file', default='', help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file

    # * using variable
    # scene_path = r"./data/scenes/test0.json"
    # scene_path = r"./data/scenes/test1_db_water.json"
    # scene_path = r"./data/scenes/test2_cc_sand.json"
    # scene_path = r"./data/scenes/test3_ht.json"

    cfg = SimConfiger(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

    # ti.init(arch=ti.cpu, debug=True, cpu_max_num_threads=1)
    ti.init(arch=ti.gpu, device_memory_fraction=cfg.get_cfg("GPUmemoryPercent"), default_fp=ti.f64, kernel_profiler=True)

    print("\n========== SIMULATION ==========")
    case = Simulation(config=cfg)
    ui_sim(config=cfg, case=case)

    print("\n========== END ==========")