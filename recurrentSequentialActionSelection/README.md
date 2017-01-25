## Sequential Action Selection for Budgeted Localization in Robots

* agent.lua: Class to create a simulated agent with its sensors

* maze.lua: Class to create a simulated maze

* model.lua: Class that describes th model

* main.lua: main executable


The command to run the main:

* OMP_NUM_THREADS=1 th main.lua [--options] -- direct execution

* OMP_NUM_THREADS=1 th -i main.lua [--options] -- interaction at the end of the execution


[--options]:

* Maze parameters

  - --maze_name: In case maze already created, give the name of the file (Empty string by default)

  - --size_x_maze: Size of length of the maze in pixels (50 by default).

  - --size_y_maze: Size of width of the maze (50 by default).

  - --size_min_wall: Minimum size of walls (20 by default).

  - --size_max_wall: Maximum size of walls (40 by default).

  - --nb_walls: Number of walls (50 by default).

  - --nb_colors: number of colors of walls (5 by default).


* Agent parameters

  - --turn_angle: The defined turning angle of the agent (pi/4 by default).

  - --sensor_angle_min: The minimum range of the sensors (0 by default).

  - --sensor_angle_max: The maximum range of the sensors  (pi by default).

  - --resolution: Resolution of sensors  (10 by default).


* Model Parameters

  - --budget: Number of actions allowed in the model  (1 by default).

  - --lrc: Learning Rate for representation learning (0.01 by default).

  - --lra: Learning Rate for policy gradient  (0.01 by default).

  - --N: Size of latent state  (50 by default).

  - --type_transformation: Type of cell for the transformations, can be RNN or GRU  (RNN by default).

  - --nb_iteration: Number of learning iterations (1000 by default).

  - --type_policy: Type of applied policy, can be forced or free (free by default).

  - --train_size: Size of train data (1250 by default).

  - --test_size: Size of test data (625 by default).

  - --valid_size: Size of valid data (625 by default).

  - --init_dist: Parameter for uniform initialization of the model (0.1 by default).
