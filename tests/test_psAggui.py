import taichi as ti

ti.init(arch=ti.cpu)

@ti.data_oriented
class TmpParticleSystem:
	def __init__(self) -> None:
		print("temp particle system serve.")



if __name__ == "__main__":
	print("hallo test particle system and ggui show!")

	case = TmpParticleSystem()
