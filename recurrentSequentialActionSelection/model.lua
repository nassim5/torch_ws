local model, parent = torch.class("model","nn.Module")

function model:__init(type_net, state_rep, budget, type_policy, type_transformation, agent)
	self.type_net = type_net
	self.type_policy = type_policy
	self.N = state_rep
	self.budget = budget
	self.agent=agent

	-- Create a module by action
	local state_module = {}
	-- RNN modules ( f = tanh(Zs+Zo) )
	if type_transformation == "RNN" then
		for a = 1, 3 do
			local in1 = nn.Identity()()
			local in2 = nn.Identity()()
			local m1 = nn.Sequential():add(nn.Linear(self.N, self.N))(in1)
			local m2 = nn.Sequential():add(nn.Linear(self.agent.sensor_resolution*3, self.N))(in2)
			local tadd = nn.CAddTable()({m1,m2})
			local tan = nn.Tanh()(tadd)
			state_module[a] = nn.gModule({in1,in2},{tan})
		end

	-- GRU modules
	elseif type_transformation == "GRU" then
		for a=1,3 do
			local prev_s = nn.Identity()()
			local obs = nn.Identity()()

			--update module
			local m1 = nn.Linear(self.N,self.N)(prev_s)
			local m2 = nn.Linear(self.agent.sensor_resolution*3,self.N)(obs)
			local mod1 = nn.CAddTable()({m1,m2})
			local update_mod = nn.Sigmoid()(mod1)

			--reset module
			local m3 = nn.Linear(self.N,self.N)(prev_s)
			local m4 = nn.Linear(self.agent.sensor_resolution*3,self.N)(obs)
			local mod2 = nn.CAddTable()({m3,m4})
			local reset_mod = nn.Sigmoid()(mod2)

			--hidden module
			local mul1=nn.CMulTable()({reset_mod, prev_s})
			local mod3=nn.Linear(self.N, self.N)(mul1)
			local mod4=nn.Linear(self.agent.sensor_resolution*3,self.N)(obs)
			local sum1=nn.CAddTable()({mod3,mod4})
			local hidden_mod = nn.Tanh()(sum1)

			--sortie, representation de st+1
			local mod5=nn.CMulTable()({update_mod, hidden_mod})
			local mod6=nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_mod))
			local mod7=nn.CMulTable()({mod6,prev_s})
			local out_mod=nn.CAddTable()({mod7,mod5})

			state_module[a]=nn.gModule({prev_s, obs},{out_mod})
		end
	else
		print("unknown transform")
		os.exit()
	end
	-- Clone the modules according to budget to create the recursion
	self.table_of_state_modules = {}
	for b = 1, self.budget do
		self.table_of_state_modules[b] = {}
		for a =1, 3 do
			self.table_of_state_modules[b][a] = state_module[a]:clone()
		end
	end

	-- SoftMax Module
	self.sm = nn.SoftMax()

	-- Action module
	module_action = nn.Sequential():add(nn.Linear(self.N,3))

	-- Cloning action modules
	self.action_modules = {}
	for a =1, self.budget do
		self.action_modules[a] = module_action:clone()
	end
end

-- Initialze modules
function model:reset(stdv)
	for j=1,self.budget do
		for i =1,3 do
			self.table_of_state_modules[j][i]:reset(stdv)
		end
		self.action_modules[j]:reset(stdv)
	end
end

-- Gradient reinitialization
function model:zeroGradParameters()
	for j=1,self.budget do
		for i =1,3 do
			self.table_of_state_modules[j][i]:zeroGradParameters()
		end
		self.action_modules[j]:zeroGradParameters()
	end
end

-- Model forward: Input is the initial state and the policy in case of forced policy
function model:forward(initial_state, policy)

	-- Lists of outputs
	local history_actions = {}
	local proba_actions = {}
	local action_scores = {}
	local input_state = {}
	local output_state = {}
	local observation ={}


	active_state=initial_state
	for b =1, self.budget do
		input_state[b] = active_state
		-- Computing action scores
		local out_action = self.action_modules[b]:forward(input_state[b])
		action_scores[b] = out_action
		-- Computing action probabilities
		proba_actions[b] = self.sm:forward(out_action)

		-- Choose action according to the policy type
		if self.type_policy == 'free' then
			action_chosen = torch.multinomial(proba_actions[b],1)[1]
		elseif self.type_policy=="forced" then
			action_chosen = tonumber(policy[b])
		else
			print("unknown")
			os.exit()
		end

		self.sm:zeroGradParameters()

		history_actions[b]=action_chosen
		-- Forward the observation
		if action_chosen == 1 or action_chosen == 2 then -- case action left or right (Vector of zeroes)
			self.agent:action(action_chosen)
			observation[b] = torch.Tensor(self.agent.sensor_resolution*3):fill(0)
			output_state[b] = self.table_of_state_modules[b][action_chosen]:forward({input_state[b],observation[b]})
		elseif action_chosen == 3 then -- case of an image
			observation[b] = self.agent:getImage()
			output_state[b] = self.table_of_state_modules[b][3]:forward({input_state[b],observation[b]})
		end
		active_state=output_state[b]
	end
	local retour = {history_actions=history_actions,proba_actions=proba_actions, action_scores=action_scores, input_state=input_state, output_state=output_state, observation=observation}
	return retour
end

-- Backward on state modules for the representation learning
function model:backward_state_modules(retour, delta)
	for b = self.budget, 1, -1 do
		local action_chosen = retour.history_actions[b]
		delta = self.table_of_state_modules[b][action_chosen]:backward({retour.input_state[b], retour.observation[b]}, delta)[1]
	end
end

-- Backward on action modules for the action selection
function model:backward_action_modules(retour, loss)
	local c2 = nn.LogSoftMax()
	for b = 1, self.budget do
		local del = torch.Tensor(3):fill(0)
		local out_action = retour.action_scores[b]
		local out_action2 = c2:forward(out_action)
		del[retour.history_actions[b]] = loss
		del = c2:backward(out_action, del)
		deltaAct = self.action_modules[b]:backward(retour.input_state[b], del)
		if b>1 then
			self.table_of_state_modules[b-1][retour.history_actions[b-1]]:backward({retour.observation[b-1],retour.input_state[b-1]}, deltaAct)
		end
	end
end

-- Update the model
function model:updateParameters(learningRateClassif, learningRateAction)
	for b =1, self.budget do
		for i=1, 3 do
			self.table_of_state_modules[b][i]:updateParameters(learningRateClassif)
		end
		self.action_modules[b]:updateParameters(learningRateAction)
	end
end
