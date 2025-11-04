use std::time::{Duration, Instant};


pub struct TickUitl {
    interval: Duration, // 设定的间隔时间
    last_tick: Instant, // 上次 tick 的时间
}

impl TickUitl {
    // 创建一个新的 Tick，设置间隔时间
    pub fn new(interval_seconds: u64) -> Self {
        TickUitl {
            interval: Duration::from_secs(interval_seconds),
            last_tick: Instant::now(),
        }
    }

    // 检查当前时间是否超过设定的间隔，若是，返回 true，并重置计时器
    pub fn tick(&mut self) -> bool {
        if self.last_tick.elapsed() >= self.interval {
            self.last_tick = Instant::now(); // 重置计时器
            return true;
        }
        false
    }
}