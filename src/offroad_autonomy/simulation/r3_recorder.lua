-- Evaluation observations stay inside the simulator until drained, so slow RGB
-- inference cannot hide an excursion between successive control commands.
local M = {}
local samples = {}
local elapsed = 0
local sequence = 0
local vehicleId
local overflow = false
local active = false
local limit

function M.start(id, maxBufferedSamples)
  vehicleId = id
  limit = maxBufferedSamples
  samples = {}
  elapsed = 0
  sequence = 0
  overflow = false
  active = true
  M.command = {steering = 0, throttle = 0, brake = 0}
end

function M.onUpdate(dtReal, dtSim)
  if not active or dtSim <= 0 then
    return
  end
  elapsed = elapsed + dtSim
  sequence = sequence + 1
  if #samples >= limit then
    overflow = true
    return
  end
  local vehicle = scenetree.findObject(vehicleId)
  if not vehicle then
    active = false
    return
  end
  local position = vehicle:getPosition()
  local velocity = vehicle:getVelocity()
  local direction = vehicle:getDirectionVector()
  local box = vehicle:getSpawnWorldOOBB()
  local corners = {}
  for i = 0, 7 do
    local p = box:getPoint(i)
    corners['point_' .. i] = {p.x, p.y, p.z}
  end
  samples[#samples + 1] = {
    timestamp_s = elapsed,
    dt_s = dtSim,
    sequence = sequence,
    position = {position.x, position.y, position.z},
    direction = {direction.x, direction.y, direction.z},
    speed_mph = velocity:length() / 0.44704,
    bbox = corners,
    steering = M.command.steering,
    throttle = M.command.throttle,
    brake = M.command.brake,
    manual_intervention = 'unknown'
  }
end

function M.drain()
  local result = {samples = samples, overflow = overflow, sequence = sequence, active = active}
  samples = {}
  return result
end

return M
