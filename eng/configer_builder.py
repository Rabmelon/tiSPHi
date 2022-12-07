import json

class SimConfiger:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        # print(self.config)
        print("\n========== CONFIGURE LOADED ==========")

    def get_cfg(self, name, enforce_exist=False):
        if enforce_exist:
            assert name in self.config["Configuration"]
        return self.config["Configuration"][name]

    def get_ui(self, name, enforce_exist=False):
        if enforce_exist:
            assert name in self.config["UIcontrol"]
        return self.config["UIcontrol"][name]

    def get_materials(self):
        if "Materials" in self.config:
            return self.config["Materials"]
        else:
            return []

    def get_fluid_bodies(self):
        if "FluidBodies" in self.config:
            return self.config["FluidBodies"]
        else:
            return []

    def get_fluid_blocks(self):
        if "FluidBlocks" in self.config:
            return self.config["FluidBlocks"]
        else:
            return []

    def get_rigid_bodies(self):
        if "RigidBodies" in self.config:
            return self.config["RigidBodies"]
        else:
            return []

    def get_rigid_blocks(self):
        if "RigidBlocks" in self.config:
            return self.config["RigidBlocks"]
        else:
            return []
