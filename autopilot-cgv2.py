#!/usr/bin/env python3
import collections
import datetime
import numpy as np
import os
import pathlib
import time

from typing import Any, Dict, List, Set


def get_ctr_map(components):
    ctr_map = {}
    for name in components:
        ctr_map[name] = f"{name}/"  # cgroup v2: just the container name
    return ctr_map


def stat_path(ctr_map, name, stat):
    group = ctr_map[name]
    return pathlib.Path(f"/sys/fs/cgroup/system.slice/{group}{stat}")  # cgroup v2 unified path


def set_cpu_limit(ctr_map, name, limit, period=0.1):
    period_us = round(period * 1e6)
    assert 1000 <= period_us <= 1000000
    cpu_max_path = stat_path(ctr_map, name, "cpu.max")
    if limit is None:
        cpu_max_path.write_text("max %d" % period_us)
        print(f"{datetime.datetime.now()} Written cpu.max=max {period_us} to name={name},(qos,uid)={ctr_map[name]}")
    else:
        quota_us = round(limit * period_us)
        assert quota_us >= 1000
        cpu_max_path.write_text(f"{quota_us} {period_us}")
        print(f"{datetime.datetime.now()} Written cpu.max={quota_us} {period_us} to name={name},(qos,uid)={ctr_map[name]}")
    return

def get_running_containers(root_dir: str):
    # traverse root directory, and list directories as dirs and files as files
    ctrs: List[str] = []
    for root, dirs, files in os.walk(f"{root_dir}"):
        path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for d in dirs:
            if d.startswith('docker-') and d.endswith('.scope'):
                ctrs.append(d)

    print(f'Found {len(ctrs)} running containers: {[c.lstrip('docker-')[:5] for c in ctrs]}')
    return ctrs

class SHOWAR:
    def __init__(self, root_dir: str = f"/sys/fs/cgroup/system.slice") -> None:
        self.running_containers = []
        self.stats_history = {}
        self.ctr_map = {}
        self.root_dir = root_dir
        self.sample_rate_sec = 1  # 20ms
        self.scale_freq_sec = 1  # 20ms
        self.last_scale_t = 0
        self.window_len = 10  # 50ms
        self.thresh_perc = 0.15

        # State
        self.spread = {}
        self.last_t = 0
        self.files = {}

        self.update_state()
        # Init limits
        for name in self.running_containers:
            self.spread[name] = None
            set_cpu_limit(self.ctr_map, name, None)

    def update_state(self):
        self.running_containers = get_running_containers(self.root_dir)
        self.ctr_map = get_ctr_map(self.running_containers)
        for name in self.running_containers:
            if name not in self.spread:
                self.spread[name] = None

    def sleep_sample_period(self):
        t = time.perf_counter()
        # tt = (0.097 - t) * 1000 % 100 / 3000  # ~30ms
        tt = self.sample_rate_sec
        print(f'At {t:.4f} sleeping for {tt:.4f} sec ...')
        t += tt
        time.sleep(tt)
        print(f'At {t:.4f} woke up')
        self.last_t = t

    def wait_cgroup_exist(self):
        files_ready = False
        to_check = ["cpu.stat", "cpu.max"]
        for name in self.running_containers:
            while not files_ready:
                checked = []
                for f in to_check:
                    stat_obj = stat_path(self.ctr_map, name, f)
                    if stat_obj.is_file():
                        checked.append(f)
                if len(checked) == len(to_check):
                    files_ready = True
                    print(f"Cgroup for {name[:5]} ready! t={self.last_t}")
                else:
                    print(
                        f"Cgroup for {name[:5]} not ready, sleeping ... t={self.last_t}"
                    )
                self.sleep_sample_period()

    def open_cgroup_files(self):
        cgroup_files = [
            "cpu.stat",
            "cpu.max",
        ]
        for name in self.running_containers:
            for cf in cgroup_files:
                if (name, cf) not in self.files:
                    self.files[name, cf] = stat_path(self.ctr_map, name, cf).open()

    def get_stats(self):
        stats = collections.defaultdict(dict)
        for name in self.running_containers:
            # Parse cpu.stat for usage_usec, nr_periods, nr_throttled, throttled_usec
            self.files[name, "cpu.stat"].seek(0)
            usage_usec = None
            nr_periods = None
            nr_throttled = None
            throttled_usec = None
            for line in self.files[name, "cpu.stat"].read().splitlines():
                k, v = line.split()
                if k == "usage_usec":
                    usage_usec = int(v)
                elif k == "nr_periods":
                    nr_periods = int(v)
                elif k == "nr_throttled":
                    nr_throttled = int(v)
                elif k == "throttled_usec":
                    throttled_usec = int(v)
            stats[name]["cpu_usage"] = usage_usec / 1e6 if usage_usec is not None else 0  # seconds
            stats[name]["cpu_stat.nr_periods"] = nr_periods if nr_periods is not None else 0
            stats[name]["cpu_stat.nr_throttled"] = nr_throttled if nr_throttled is not None else 0
            stats[name]["cpu_stat.throttled_time"] = throttled_usec / 1e6 if throttled_usec is not None else 0  # seconds
            # Parse cpu.max for quota and period
            self.files[name, "cpu.max"].seek(0)
            cpu_max = self.files[name, "cpu.max"].read().strip().split()
            if cpu_max[0] == "max":
                stats[name]["cpu_cfs_quota_us"] = -1
            else:
                stats[name]["cpu_cfs_quota_us"] = int(cpu_max[0])
            stats[name]["cpu_cfs_period_us"] = int(cpu_max[1])
        print(stats)
        return stats

    def run(self):
        monotonic_base = time.time() - time.perf_counter()
        self.stats_history = collections.defaultdict(list)
        while True:
            # Need to parallelize this?
            self.sleep_sample_period()
            self.update_state()
            if len(self.running_containers) == 0:
                print(f"No running containers!")
                self.sleep_sample_period()
                continue

            self.open_cgroup_files()
            stats = self.get_stats()

            # Type + derive values
            for name in self.running_containers:
                try:
                    stats[name]["cpu_usage"] = int(stats[name]["cpu_usage"]) / 1e9
                    stats[name]["dt_cpu_usage"] = (
                        stats[name]["cpu_usage"]
                        - self.stats_history[name][-1][1]["cpu_usage"]
                        if name in self.stats_history
                        else 0
                    )
                    stats[name]["cpu_stat.nr_periods"] = int(
                        stats[name]["cpu_stat.nr_periods"]
                    )
                    stats[name]["cpu_stat.nr_throttled"] = int(
                        stats[name]["cpu_stat.nr_throttled"]
                    )
                    stats[name]["cpu_stat.throttled_time"] = (
                        int(stats[name]["cpu_stat.throttled_time"]) / 1e9
                    )
                    stats[name]["cpu_stat.throttled_time"] = (
                        int(stats[name]["cpu_stat.throttled_time"]) / 1e9
                    )
                    stats[name]["cpu_cfs_quota_us"] = int(
                        stats[name]["cpu_cfs_quota_us"]
                    )
                    stats[name]["cpu_cfs_period_us"] = int(
                        stats[name]["cpu_cfs_period_us"]
                    )
                except Exception as e:
                    print(f"At t={self.last_t} {name} error {e}")

            for name in stats:
                self.stats_history[name].append(
                    (self.last_t + monotonic_base, stats[name])
                )
                self.stats_history[name] = self.stats_history[name][-self.window_len :]

            # SCALE UP/DOWN
            for name in stats:
                cpu_usages = self.get_cpu_usages(name)
                mean = np.mean(cpu_usages)
                std = np.std(cpu_usages)
                spread = mean + (3 * std)
                target_core = spread / self.sample_rate_sec
                print(f'mean={mean:.4f}, std={std:.4f}, Target core for {name[:5]}={target_core:.4f}')
                if not self.spread[name]:
                    self.spread[name] = spread
                else:
                    diff = np.abs(spread - self.spread[name])
                    threshold = self.thresh_perc * self.spread[name]
                    last_scale_diff = np.abs(self.last_scale_t - self.last_t) 
                    print(f"At t={self.last_t:.4f}, {name[:5]}, curr. spread={self.spread[name]:.4f}, obs. spread={spread:.4f}, len={len(self.stats_history[name])}, thresh={threshold:.4f}, last_scale_diff={last_scale_diff:.4f}")
                    # hist = self.stats_history[name][-5:]
                    # dts = []
                    # for e in hist:
                    #     dts.append(e[1]["dt_cpu_usage"])
                    print(f'At t={self.last_t}, dts={cpu_usages}, mu={mean:.4f}, std={std:.4f}')
                    if (diff > threshold) and \
                        (last_scale_diff > self.scale_freq_sec):
                        # limit -> quota_us conversion requires quota >= 1000
                        target_core = max((spread / self.sample_rate_sec), 0.01)
                        set_cpu_limit(self.ctr_map, name, target_core)
                        self.spread[name] = spread
                        self.last_scale_t = self.last_t

    def get_cpu_usages(self, name: str) -> List[float]:
        hist = self.stats_history[name]
        cpu_usages = []
        for ts, stat in hist:
            cpu_usages.append(stat["dt_cpu_usage"])

        return cpu_usages


def main():
    showar = SHOWAR()
    showar.run()


if __name__ == "__main__":
    main()



