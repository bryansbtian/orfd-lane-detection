-- The native input source whitelist also covers input.lua's private event()
-- references, which an exported-function wrapper alone cannot intercept.
local M = {}
local active = false
local authority = 'r3Autonomy'
local originalEvent
local patched = {}
local savedAllowed = {}
local channels = {'steering', 'throttle', 'brake', 'parkingbrake', 'clutch'}
local watched = {}
local command = {}
local events = {}
local interventions = 0
local faults = 0
local sequence = 0
local elapsed = 0
local file
local config
local latest
local startDamage = 0
local buffer = {}
local bufferOverflow = false

local function reject(kind, channel, value, filter)
  if value == 0 then
    return
  end
  interventions = interventions + 1
  events[#events+1] = {timestamp_s=elapsed, kind=kind, channel=channel,
    value=value, filter=tostring(filter), blocked=true}
end

local function patch(tableValue, name, fn)
  if type(tableValue[name]) ~= 'function' then
    return
  end
  patched[#patched+1] = {target=tableValue, name=name, original=tableValue[name], replacement=fn}
  tableValue[name] = fn
end

local function guardedMethod(target, name)
  local key = name
  patch(target, key, function(...) reject('Native Vehicle Action', key, 1, 'action') end)
end

function M.apply(values)
  for _, key in ipairs(channels) do
    if values[key] ~= nil then
      command[key] = values[key]
      originalEvent(key, values[key], 1, nil, nil, nil, authority)
    end
  end
end

function M.start(args)
  if active then
    error('R3 verifier already active')
  end
  config = args
  buffer = {}
  bufferOverflow = false
  events = {}
  latest = nil
  file = {
    write = function(self, text) buffer[#buffer+1] = text end,
    flush = function() end,
    close = function() end
  }
  file:write('sequence,timestamp_s,dt_s,x,y,z,fx,fy,fz,ux,uy,uz,vx,vy,vz,low_f,high_f,low_r,high_r,low_u,high_u,steering,throttle,brake,parkingbrake,manual_interventions,guard_faults,damage\n')
  originalEvent = input.event
  for _, key in ipairs(channels) do
    watched[key] = true
    savedAllowed[key] = input.allowedInputSources[key]
    input.allowedInputSources[key] = {[authority]=true}
    command[key] = 0
  end
  patch(input, 'event', function(key, value, filter, ...)
    if watched[key] then
      reject('Input Event', key, value, filter)
      return
    end
    return originalEvent(key, value, filter, ...)
  end)
  local toggle = input.toggleEvent
  patch(input, 'toggleEvent', function(key)
    if watched[key] then
      reject('Toggle Event', key, 1, 'toggle')
      return
    end
    return toggle(key)
  end)
  patch(input, 'kbdSteer', function(isRight, value, filter)
    reject('Keyboard Steering', 'steering', value, filter)
  end)
  patch(input, 'padAccelerateBrake', function(value, filter)
    reject('Gamepad Combined Pedals', 'pedals', value, filter)
  end)
  for _, name in ipairs({'shiftUp','shiftDown','shiftUpOnDown','shiftUpOnUp',
      'shiftDownOnDown','shiftDownOnUp','shiftToGearIndex','setGearboxMode',
      'cycleGearboxModes','smartParkingBrake'}) do
    guardedMethod(controller.mainController, name)
  end
  for _, name in ipairs({'setIgnitionLevel','toggleIgnitionLevel','setStarter'}) do
    guardedMethod(electrics, name)
  end
  guardedMethod(powertrain, 'toggleDefaultDiffs')
  for _, unit in pairs(controller.getControllersByType('4wd')) do
    for _, name in ipairs({'toggleDiffs','toggle4WD','toggleRange'}) do
      guardedMethod(unit, name)
    end
  end
  sequence, elapsed, interventions, faults = 0, 0, 0, 0
  startDamage = beamstate.damage or 0
  active = true
  M.apply({steering=0,throttle=0,brake=0,parkingbrake=1,clutch=0})
  enablePhysicsStepHook()
  return {physics_dt=obj:getPhysicsDt(), nodes=tableSize(v.data.nodes),
    patched_functions=#patched, initial_damage=startDamage}
end

local function integrity()
  for _, key in ipairs(channels) do
    local allowed = input.allowedInputSources[key]
    local state = input.state[key]
    if not allowed or not allowed[authority] or not state or
       state.source ~= authority or state.val ~= command[key] then
      return false
    end
    for source, permitted in pairs(allowed) do
      if source ~= authority and permitted then
        return false
      end
    end
  end
  for _, entry in ipairs(patched) do
    if entry.target[entry.name] ~= entry.replacement then
      return false
    end
  end
  return true
end

function M.onPhysicsStep(dt)
  if not active then
    return
  end
  if #buffer > config.max_buffered_samples then
    bufferOverflow = true
    faults = faults + 1
    M.apply({steering=0,throttle=0,brake=1,parkingbrake=1})
    return
  end
  elapsed = elapsed + dt
  sequence = sequence + 1
  if not integrity() then
    faults = faults + 1
  end
  local p = obj:getPosition()
  local f = obj:getDirectionVector():normalized()
  local right = f:cross(obj:getDirectionVectorUp()):normalized()
  local up = right:cross(f):normalized()
  local vel = obj:getVelocity()
  local b = config.body_bounds
  local margin = config.envelope_margin_m
  local lf, hf, lr, hr, lu, hu = b[1]-margin,b[2]+margin,b[3]-margin,b[4]+margin,b[5]-margin,b[6]+margin
  -- JBeam collision triangles lie within the envelope of their nodes. Include
  -- every current node as well as the calibrated visible-body OOBB, so wheel
  -- articulation is not treated as a rigid-width approximation.
  for _, node in pairs(v.data.nodes) do
    local n = obj:getNodePosition(node.cid)
    local nf, nr, nu = n:dot(f), n:dot(right), n:dot(up)
    lf, hf = math.min(lf,nf-margin),math.max(hf,nf+margin)
    lr, hr = math.min(lr,nr-margin),math.max(hr,nr+margin)
    lu, hu = math.min(lu,nu-margin),math.max(hu,nu+margin)
  end
  local damage = beamstate.damage or 0
  file:write(string.format(
    '%d,%.9f,%.12f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d,%.6f\n',
    sequence,elapsed,dt,p.x,p.y,p.z,f.x,f.y,f.z,up.x,up.y,up.z,vel.x,vel.y,vel.z,
    lf,hf,lr,hr,lu,hu,command.steering,command.throttle,command.brake,command.parkingbrake,
    interventions,faults,damage))
  latest = {sequence=sequence,timestamp_s=elapsed,position={p.x,p.y,p.z},
    direction={f.x,f.y,f.z},up={up.x,up.y,up.z},speed_mph=vel:length()/0.44704,
    bounds={lf,hf,lr,hr,lu,hu},manual_interventions=interventions,guard_faults=faults,
    damage=damage,initial_damage=startDamage}
end

function M.updateGFX(dt)
  if file then
    file:flush()
  end
end

function M.status()
  if file then
    file:flush()
  end
  return {active=active,latest=latest,interventions=interventions,guard_faults=faults,
    sequence=sequence,events=events,physics_dt=obj:getPhysicsDt(),buffer_overflow=bufferOverflow}
end

function M.drain()
  local result = table.concat(buffer)
  buffer = {}
  return result
end

function M.packet()
  local result = M.status()
  result.csv = M.drain()
  return result
end

function M.onReset()
  if active then
    reject('Vehicle Reset', 'reset', 1, 'reset')
    faults = faults + 1
  end
end

function M.stop()
  local result = M.status()
  active = false
  if file then
    file:flush()
    file:close()
    file=nil
  end
  for i = #patched, 1, -1 do
    local entry = patched[i]
    entry.target[entry.name] = entry.original
  end
  for _, key in ipairs(channels) do input.allowedInputSources[key] = savedAllowed[key] end
  patched = {}
  return result
end

M.onExtensionUnloaded = M.stop
return M
