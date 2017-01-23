**The ReadMe for the Sequential Action Selection Paper**

* agent.lua: Class to create a simulated agent with its sensors

* maze.lua: Class to create a simulated maze

* model.lua: Class that describes th model

* main.lua: main executable


The command to run the main:

* OMP_NUM_THREADS=1 th main.lua [--options] -- direct execution

* OMP_NUM_THREADS=1 th -i main.lua [--options] -- interaction at the end of the execution


[--options]:

* Maze parameters

- --maze_name: In case maze already created, give the name of the file

- --size_x_maze: Size of length of the maze

- --size_y_maze: Size of width of the maze

- --size_min_wall: Minimum size of walls

- --size_max_wall: Maximum size of walls

- --nb_walls: Number of walls

- --nb_colors: number of colors of walls


Agent parameters

--turn_angle: The defined turning angle of the agent

--sensor_angle_min: The minimum range of the sensors

--sensor_angle_max: The maximum range of the sensors

--resolution: Resolution of sensors


Model Parameters

--budget: Number of actions allowed in the model

--lrc: Learning Rate of Classifier

--lra: Learning Rate of Action selection

--N: Size of latent state

--type_transformation: Type of cell for the transformations, can be RNN or GRU

--nb_iteration: Number of learning iterations

--type_policy: Type of applied policy, can be forced or free

--train_size: Size of train data

--test_size: Size of test data

--valid_size: Size of valid data

--init_dist: Parameter for uniform initialization of the model.
