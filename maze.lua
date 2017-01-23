local maze = torch.class("maze");

function maze:__init(size_x, size_y, size_min_wall, size_max_wall, nb_walls, nb_colors)
	self.size_x = size_x
	self.size_y = size_y
	self.size_min_wall = size_min_wall
	self.size_max_wall = size_max_wall
	self.nb_walls = nb_walls
	self.nb_colors = nb_colors
	self.maze = torch.Tensor(self.size_x, self.size_y):fill(0)
	self.palette={}
end

-- Draw a line from point (x1,y1) to (x2,y2) with the given color
function maze:createWall(x1, x2, y1, y2, color)
	local pos_x = x1
	local pos_y = y1
	local length = math.sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))
	while (((pos_x-x2)*(x1-x2))+((pos_y-y2)*(y1-y2))>=0) do
	    local x=math.floor(pos_x)
   		local y=math.floor(pos_y)
    	if ((x>0) and (x<=self.size_x) and (y>0) and (y<=self.size_y)) then
        	self.maze[x][y]=color
	    end
    	pos_x=pos_x+(x2-x1)/length
    	pos_y=pos_y+(y2-y1)/length
	end
end

-- Create a maze the initialization parameters
function maze:createMaze()
	local n = 0
	while n<self.nb_walls do
		x1=math.floor(torch.uniform(self.size_x))
		y1=math.floor(torch.uniform(self.size_y))
		x2=math.floor(torch.uniform(self.size_x))
		y2=math.floor(torch.uniform(self.size_y))
		local dist = math.sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))
		if ((dist>=self.size_min_wall) and (dist<=self.size_max_wall)) then
			local color = math.random(self.nb_colors)
			self:createWall(x1,x2,y1,y2,color)
			n=n+1
		end
	end
	self:generateColors()
end

-- Generate the RGB palette of each color
function maze:generateColors()
	for i =1, self.nb_colors do
		self.palette[i]=torch.rand(3)
	end
end

-- Return the correspondant class of the point (x,y)
function maze:getClass(nb_classes, x, y)
	local class = torch.Tensor(nb_classes*nb_classes):fill(0)
	local px=math.floor((x-1)/self.size_x*nb_classes)
	local py=math.floor((y-1)/self.size_y*nb_classes)
	local idx = px*nb_classes+py+1
	class[idx] = 1
	return class
end


