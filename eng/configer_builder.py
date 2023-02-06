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

    def get_materials(self):
        if "Materials" in self.config:
            return self.config["Materials"]
        else:
            return []

    def get_blocks(self):
        if "Blocks" in self.config:
            return self.config["Blocks"]
        else:
            return []

    def get_bodies(self):
        if "Bodies" in self.config:
            return self.config["Bodies"]
        else:
            return []

    def get_motions(self):
        if "Motions" in self.config:
            return self.config["Motions"]
        else:
            return []

