require 'nn'
require 'dpnn'
require 'nngraph'
require 'image'
require 'maze'
require 'optim'
require 'agent'
require 'model'

torch.seed()
cmd = torch.CmdLine()
cmd:text()

cmd:option("--run_anno", 1, "file #")
--Maze parameters
cmd:option('--maze_name', '', 'Name of the maze in case already created')
cmd:option('--size_x_maze', 50, 'X length for the maze')
cmd:option('--size_y_maze', 50, 'Y length for the maze')
cmd:option('--size_min_wall', 20, 'Size min of walls')
cmd:option('--size_max_wall', 40, 'Size max of walls')
cmd:option('--nb_walls', 50, '# of walls')
cmd:option('--nb_colors', 5, '# of colors')

--agent parameters
cmd:option('--turn_angle', math.pi/4, 'Turning angle for the action')
cmd:option('--sensor_angle_min', 0, 'Min of range of sensor')
cmd:option('--sensor_angle_max', math.pi/2, 'max of range of sensor')
cmd:option('--resolution', 10, 'resolution of the sensor')

--Model Parameters
cmd:option('--budget', 1, '# of actions allowed in a sequence')
cmd:option('--lrc', 0.01, 'learning rate classif')
cmd:option('--lra', 0.01, 'learning rate action')
cmd:option('--N', 50, 'size state')
cmd:option('--type_transformation', "RNN", "type of latent latent transformation")
cmd:option('--nb_iteration', 1000, '# of iteration')
cmd:option('--type_policy', 'free', 'type of ation selection: free or forced')
cmd:option('--train_size', 1250, 'Size of train set')
cmd:option('--test_size', 625, 'Size of test set')
cmd:option('--valid_size', 625, 'Size of validation set')
cmd:option('--init_dist', 0.1, 'intialization of NNs')



-- Define classes
local nb_classe = 16

-- Define policy in case of forced policy
local policy = {3,1,3,1,3}

opt=cmd:parse(arg or {})

-- Use given maze
if opt.maze_name~='' then
	maze = torch.load(opt.maze_name)
-- or create a new one
else
	maze = maze(opt.size_x_maze, opt.size_y_maze, opt.size_min_wall, opt.size_max_wall, opt.nb_walls, opt.nb_colors)
	torch.save("labyrinth"..opt.size_x_maze.."x"..opt.size_y_maze.."_"..opt.nb_walls..".maze", maze)
	opt.maze_name = "labyrinth"..opt.size_x_maze.."x"..opt.size_y_maze.."_"..opt.nb_walls..".maze"
end

-- Init agent and model
agent = agent(maze, opt.turn_angle, opt.sensor_angle_min, opt.sensor_angle_max, opt.resolution)
model = model('MLP', opt.N, opt.budget, opt.type_policy, opt.type_transformation, agent)

-- Create Maze
maze:createMaze()

-- Init classifier
local classifier = nn.Sequential():add(nn.Linear(opt.N,opt.N*2)):add(nn.Tanh()):add(nn.Linear(opt.N*2,nb_classe)):add(nn.Tanh())
local criterion = nn.MSECriterion()

-- Initialize the model
model:reset(opt.init_dist)
classifier:reset(opt.init_dist)


train_set={}
test_set={}
valid_set={}

-- Sampling train set
for i = 1, opt.train_size do
	local x = torch.random(maze.size_x)
	local y = torch.random(maze.size_y)
	while maze.maze[x][y]~=0 do
		x = torch.random(maze.size_x)
		y = torch.random(maze.size_y)
	end
	local ind = torch.random(1,4)
	local angle = nil
	if ind == 1 then
		angle = 0
	elseif ind == 2 then
		angle = math.pi/2
	elseif ind == 3 then
		angle = math.pi
	else
		angle = 3*math.pi/2
	end
 	local feat = {x, y, angle}
	train_set[i] = feat
end

-- Sampling test set
for i = 1, opt.test_size do
	local x = torch.random(maze.size_x)
	local y = torch.random(maze.size_y)
	while maze.maze[x][y]~=0 do
		x = torch.random(maze.size_x)
		y = torch.random(maze.size_y)
	end
	local ind = torch.random(1,4)
	local angle = nil
	if ind == 1 then
		angle = 0
	elseif ind == 2 then
		angle = math.pi/2
	elseif ind == 3 then
		angle = math.pi
	else
		angle = 3*math.pi/2
	end
	local feat = {x, y, angle}
	test_set[i] = feat
end

-- Sampling valid set
for i = 1, opt.valid_size do
	local x = torch.random(maze.size_x)
	local y = torch.random(maze.size_y)
	while maze.maze[x][y]~=0 do
		x = torch.random(maze.size_x)
		y = torch.random(maze.size_y)
	end
	local ind = torch.random(1,4)
	local angle = nil
	if ind == 1 then
		angle = 0
	elseif ind == 2 then
		angle = math.pi/2
	elseif ind == 3 then
		angle = math.pi
	else
		angle = 3*math.pi/2
	end
	local feat = {x, y, angle}
	valid_set[i] = feat
end

-- Create Log file
local perfFileModel = assert(io.open("perf_model_"..opt.type_policy.."_"..opt.maze_name.."_"..math.sqrt(nb_classe).."_"..opt.train_size.."_"..opt.test_size.."_"..opt.resolution.."_"..opt.budget.."_"..opt.lrc.."_"..opt.lra.."_"..opt.type_transformation.."_"..opt.nb_iteration.."_"..opt.run_anno.."_test.csv","w"))

local splitter="\t"
perfFileModel:write("Iteration",splitter,"Test Perf", splitter, "Train Perf", splitter, "Valid Perf", splitter, "Budget",splitter, "Action Chosen", splitter, "Type policy",splitter, "Proba acion 1",splitter,"Proba action 2",splitter,"Proba action 3", splitter, "Resolution", splitter, "lrc", splitter, "lra", splitter, "nbIterations", "\n")

modelsToSave = {}

-- initial state
local init_state = torch.Tensor(opt.N):fill(0)
for i = 1, opt.nb_iteration do
	print("Iteration", i)
	local ok_train = 0
	local ok_train_unic = 0
	local average_loss = 0

	-- Training Loop
	for j = 1, opt.train_size do
		model:zeroGradParameters()
		classifier:zeroGradParameters()

		agent:placeAgent(train_set[j][1],train_set[j][2], train_set[j][3])
		
		-- Forward
		local retour = model:forward(init_state,policy)
		local y_predit = classifier:forward(retour.output_state[opt.budget])
		local y = maze:getClass(math.sqrt(nb_classe),train_set[j][1],train_set[j][2])
		local loss = criterion:forward(y_predit, y)
		average_loss = (average_loss+loss)/j
		local delta = criterion:backward(y_predit, y)

		-- Backward
		delta = classifier:backward(retour.output_state[opt.budget], delta)
		model:backward_state_modules(retour, delta)
		model:backward_action_modules(retour, loss)
		classifier:updateParameters(opt.lrc)
		model:updateParameters(opt.lrc,opt.lra)
			
	end
	if i%10 == 0 then
		
		-- Performance on train set
		for j = 1, opt.train_size do

			agent:placeAgent(train_set[j][1],train_set[j][2], train_set[j][3])

			local retour = model:forward(init_state,policy)
			local y_predit = classifier:forward(retour.output_state[opt.budget])
			local y = maze:getClass(math.sqrt(nb_classe),train_set[j][1],train_set[j][2])
			local _, am3 = torch.max(y,1)
			local _, am4 = torch.max(y_predit,1)
			if am3[1]==am4[1] then
				ok_train = ok_train+1
			end
		end

		ok_train = ok_train/opt.train_size
		print("train", ok_train)


		-- Performance on test set
		local ok_test = 0
		local ok_test_unic = 0
		local list_of_chosen_policies = {}
		for j = 1, opt.test_size do

			agent:placeAgent(test_set[j][1],test_set[j][2], test_set[j][3])

			retour_te = model:forward(init_state,policy)
			local y_predit = classifier:forward(retour_te.output_state[opt.budget])
			local y = maze:getClass(math.sqrt(nb_classe),test_set[j][1],test_set[j][2])
			local _, am3 = torch.max(y,1)
			local _, am4 = torch.max(y_predit,1)

			local pol = table.concat(retour_te.history_actions,'-')
			if list_of_chosen_policies[pol] == nil then
				list_of_chosen_policies[pol] = 0
			else
				list_of_chosen_policies[pol] = list_of_chosen_policies[pol]+1
			end

			if am3[1]==am4[1] then
				ok_test = ok_test+1
			end
		end

		ok_test = ok_test/opt.test_size
		print("test", ok_test)

		local max = 0
		for k,v in pairs(list_of_chosen_policies) do
			if v>max then
				max = v
				best_policy = k
			end
		end
		print("Best Policy", best_policy)


		-- Performance on validation set
		local ok_valid = 0
		local ok_valid_unic = 0
		for j = 1, opt.valid_size do

			agent:placeAgent(valid_set[j][1],valid_set[j][2], valid_set[j][3])

			local retour = model:forward(init_state,policy)
			local y_predit = classifier:forward(retour.output_state[opt.budget])
			local y = maze:getClass(math.sqrt(nb_classe),valid_set[j][1],valid_set[j][2])
			local _, am3 = torch.max(y,1)
			local _, am4 = torch.max(y_predit,1)
			if am3[1]==am4[1] then
				ok_valid = ok_valid+1
			end
		end

		ok_valid = ok_valid/opt.valid_size
		-- Save the model at this iteration
		perfFileModel:write(i,splitter,ok_test, splitter, ok_train, splitter, ok_valid, splitter, opt.budget, splitter, best_policy, splitter, opt.type_policy,splitter,retour_te.proba_actions[opt.budget][1],splitter, retour_te.proba_actions[opt.budget][2],splitter,retour_te.proba_actions[opt.budget][3], splitter, opt.resolution, splitter, opt.lrc, splitter, opt.lra, splitter, opt.nb_iteration, "\n")
		model_to_save = {
			m = model,
			pol = best_policy,
			iteration = i
		}
		table.insert(modelsToSave, model_to_save)
		torch.save("models_"..opt.budget..".mod", modelsToSave)
	end
end
--Close log file
perfFileModel:close()



