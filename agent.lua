local agent = torch.class("agent");

function agent:__init(maze, turn_angle, sensor_angle_min, sensor_angle_max, sensor_resolution)
	self.state = {}
	--self.nb_action = nb_action
	self.maze = maze
	self.turn_angle = turn_angle
	self.sensor_angle_min = sensor_angle_min
	self.sensor_angle_max = sensor_angle_max
	self.sensor_resolution = sensor_resolution
	self.memory={}
end

-- Place the agent in the maze
function agent:placeAgent(x,y,angle)
	self.state[1] = x
	self.state[2] = y
	self.state[3] = angle
end

-- Execute the action on the agent for movement
function agent:action(action)
	if action == 1 then
		self.state[3] = self.state[3] - self.turn_angle
	elseif action == 2 then
		self.state[3] = self.state[3] + self.turn_angle
	else
		print("unknown action")
		os.exit()
	end
end

-- Get color in front of the agent at the given angle
function agent:getColor(angle)
	local flag = true
	local step_x = math.cos(angle)
	local step_y = math.sin(angle)
	local sx = self.state[1]
	local sy = self.state[2]
	while flag do
		local ssx = math.floor(sx)
		local ssy = math.floor(sy)
		if ssx<1 then ssx = 1 end
		if ssy<1 then ssy = 1 end
		if (self.maze.maze[ssx][ssy]>0) then
			return self.maze.maze[ssx][ssy]
		end
		sx = sx+step_x
		sy = sy+step_y
		if (sx<1 or sy<1 or sx>self.maze.size_x or sy>self.maze.size_y) then return 0 end
	end
end

-- Get the obstacle distance in fornt of the agent at the given angle
function agent:getDistance(angle)
	local flag = true
	local step_x = math.cos(angle)
	local step_y = math.sin(angle)
	local sx = self.state[1]
	local sy = self.state[2]
	local dist = 0
	while flag do
		local ssx = math.floor(sx)
		local ssy = math.floor(sy)
		if ssx<1 then ssx = 1 end
		if ssy<1 then ssy = 1 end
		if (self.maze.maze[ssx][ssy]>0) then
			return dist
		end
		dist = dist+1
		sx = sx+step_x
		sy = sy+step_y
		if (sx<1 or sy<1 or sx>self.maze.size_x or sy>self.maze.size_y) then return 0 end
	end
end

-- Get the image in the range of size_min_angle to size_max_angle
function agent:getImage()
	local elem = table.concat(self.state, '-')
	if self.memory[elem] ~= nil then
		return self.memory[elem]
	end

	local start_angle = (self.sensor_angle_min/2)+self.state[3]
	local step = (self.sensor_angle_max-self.sensor_angle_min)/self.sensor_resolution
	local image = torch.Tensor(self.sensor_resolution*3):fill(0)
	for pos = 1, self.sensor_resolution do
		local o = self:getColor(start_angle)
		if o == 0 then
			image[(pos-1)*3+1]=0
			image[(pos-1)*3+2]=0
			image[(pos-1)*3+3]=0
		else
			image[(pos-1)*3+1]=self.maze.palette[o][1]
			image[(pos-1)*3+2]=self.maze.palette[o][2]
			image[(pos-1)*3+3]=self.maze.palette[o][3]
		end
		start_angle=start_angle+step
	end
	self.memory[elem] = image
	return image
end

-- Get the distances of the laser in the range of size_min_angle to size_max_angle
function agent:getLaser()
	local start_angle = (self.sensor_angle_min/2)+self.state[3]
	local step = (self.sensor_angle_max-self.sensor_angle_min)/self.sensor_resolution
	local laser = torch.Tensor(self.sensor_resolution):fill(0)
	for pos = 1, self.sensor_resolution do
		local l = self:getDistance(start_angle)
		laser[pos] = l
		start_angle = start_angle+1
	end
	return laser
end