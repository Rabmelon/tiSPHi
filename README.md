# tiSPHi

<div align="center">
  <img width="200px" src="./img/tiSPHi_logo_squre.png">
</div>

An accurate, stable, fast, extendable fluid-solid coupling SPH solver

## News

**07 Dec. 2022** - SORRY BUT ONLY WATER DAMBREAK CAN RUN NOW!

## Demos

<div align="center">
  <img width="600px" src="./img/sim_2022_12_05_db_WC_p.png">
</div>

<div align="center">
  <img width="600px" src="./img/sim_2022_11_30_db_WC_vel_3d.png">
</div>

Fig. Water dambreak at 1.6s, 2D colored by pressure and 3D colored by velocity

## Runtime

### Run

1. Creat or change a `.\data\scenes\***.json`, through which you can design a simulation case.
2. run `python run_simulation.py`.

### Hot key

* `SPACE`: control the pause/run of simulation.
* `ESC`: stop and exit the simulation.
* `P`: make a screenshot and save to the folder "screenshots".
* `V`: restore the initiate view.
* Press `Q`, `E` to move `down` and `up` the camera, press `W`, `A`, `S`, `D` to move `forward`, `left`, `back`, `right` the camera.
* Move the slider in `Control panel` to control the moving speed of camera (lcoked when equals to zero) or change the drawing radius of particles.

## Ambition

![Architecture](./img/code_architecture.png)
