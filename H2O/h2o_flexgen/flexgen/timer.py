"""Global timer for profiling."""
from collections import namedtuple
import time
from typing import Callable, Any

# 一个内部计时器，用于跟踪某个过程的执行时间
class _Timer:
    """An internal timer."""

    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.start_time = None

        # start-stop timestamp pairs
        self.start_times = []
        self.stop_times = []
        self.costs = []

    def start(self, sync_func: Callable = None): # 接收一个实例方法
        """Start the timer."""
        assert not self.started, f"timer {self.name} has already been started." # True时，即已经开始计时，则报错抛出异常
        if sync_func: # 如果传入了同步函数 sync_func，则调用该函数
            sync_func()

        self.start_time = time.perf_counter() # 开始计时
        self.start_times.append(self.start_time)
        self.started = True

    def stop(self, sync_func: Callable = None):
        """Stop the timer."""
        assert self.started, f"timer {self.name} is not started." # False时，即未开始计时，则报错抛出异常
        if sync_func:
            sync_func()

        stop_time = time.perf_counter()
        self.costs.append(stop_time - self.start_time) # 计算时间开销
        self.stop_times.append(stop_time)
        self.started = False 

    def reset(self): # 重置为初始状态
        """Reset timer."""
        self.started = False
        self.start_time = None
        self.start_times = []
        self.stop_times = []
        self.costs = []
    # 返回累计的执行时间
    def elapsed(self, mode: str = "average"):
        """Calculate the elapsed time."""
        if not self.costs: 
            return 0.0
        if mode == "average": # 平均时间
            return sum(self.costs) / len(self.costs)
        elif mode == "sum": # 总时间
            return sum(self.costs)
        else:
            raise RuntimeError("Supported mode is: average | sum")

# 管理多个 _Timer 实例的类
class Timers:
    """A group of timers."""
    
    def __init__(self):
        self.timers = {}
    # 返回一个计时器，若没有计时器则创建之后返回
    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]
    # 判断是否存在对应名称计时器
    def __contains__(self, name: str):
        return name in self.timers


timers = Timers()

Event = namedtuple("Event", ("tstamp", "name", "info"))


class Tracer:
    """An activity tracer."""

    def __init__(self):
        self.events = []

    def log(self, name: str, info: Any, sync_func: Callable = None):
        if sync_func:
            sync_func()

        self.events.append(Event(time.perf_counter(), name, info))


tracer = Tracer()
