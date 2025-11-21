print("[thread_demo] registering worker thread demo")

local worker_code = [=[
print("[thread_demo worker] started")
local tick = 0
while true do
    tick = tick + 1
    print(string.format("[thread_demo worker] tick %d", tick))
    if mile_sleep then
        mile_sleep(1.0)
    end
end
]=]

if mile_thread and mile_thread.spawn then
    mile_thread.spawn(worker_code)
else
    print("[thread_demo] mile_thread API unavailable")
end
