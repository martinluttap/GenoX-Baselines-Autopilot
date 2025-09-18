#!/usr/bin/env python3
import collections
import datetime
import numpy as np
import os
import pathlib
import time

from typing import Any, Dict, List, Set, Tuple


def get_ctr_map(components):
    ctr_map = {}
    for path in components:
        ctr_map[path] = path  # Use full cgroup path
    return ctr_map


def stat_path(ctr_map, name, stat):
    group = ctr_map[name]
    return pathlib.Path(f"{group}/{stat}")


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
    # Recursively find docker/cri-containerd scope directories under /sys/fs/cgroup
    ctrs: List[str] = []
    for root, dirs, files in os.walk("/sys/fs/cgroup"):
        for d in dirs:
            if (d.startswith("docker-") or d.startswith("cri-containerd-")) and d.endswith(".scope"):
                ctrs.append(os.path.join(root, d))
    print(f"Found {len(ctrs)} running containers: {[os.path.basename(c)[:20] for c in ctrs]}")
    return ctrs

class AutoPilot:
    def __init__(self, root_dir: str = f"/sys/fs/cgroup/system.slice") -> None:
        self.running_containers = []
        self.stats_history = {}
        self.ctr_map = {}
        self.root_dir = root_dir
        self.sample_rate_sec = 0.1  # Sample CPU usage every this sec
        self.scale_freq_sec = 0.1  # Trigger scaling decision every this sec
        self.agg_freq_sec = 1 # Aggregate every this secs
        self.last_scale_t = 0
        self.last_agg_t = 0
        self.agg_len = int(self.agg_freq_sec / self.scale_freq_sec) # Num. elements aggregated
        self.agg_lifetime_sec = 300 # Keep agg_sample for this long  
        self.rec_last_n_samples = 2 # Number of agg samples for weighted average
        self.window_len = 1000  # 
        self.thresh_perc = 0.1
        self.dt_wall = 0.0

        # State
        self.spread = {}
        self.last_t = 0
        self.raw_samples = []
        self.hist_agg_samples = []
        self.files = {}

        self.update_state()
        # Init limits
        for name in self.running_containers:
            self.spread[name] = None
            set_cpu_limit(self.ctr_map, name, None)

    def decaying_weight(self, age: float, t_half: float = 10.0) -> float:
        """Exponential decay function."""
        return 2**(-age / t_half)

    def hists_average(self, hists: List[Tuple]) -> float:
        numerator = []
        denominator = []
        for idx, hist in enumerate(hists):
            values, boundaries = hist
            avg = np.sum(boundaries[1:]*values) / np.sum(values)
            weight = self.decaying_weight(len(hists) - idx)
            numerator.append(weight * avg)
            denominator.append(weight)

        return np.sum(numerator) / np.sum(denominator)

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
        self.dt_wall = t - self.last_t
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
                try:
                    # Parse cpu.stat for usage_usec, nr_periods, nr_throttled, throttled_usec
                    if (name, "cpu.stat") not in self.files:
                        stat_path_obj = stat_path(self.ctr_map, name, "cpu.stat")
                        if not stat_path_obj.is_file():
                            print(f"Warning: cpu.stat not found for {name}, skipping.")
                            continue
                        self.files[name, "cpu.stat"] = stat_path_obj.open()
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
                    if (name, "cpu.max") not in self.files:
                        max_path_obj = stat_path(self.ctr_map, name, "cpu.max")
                        if not max_path_obj.is_file():
                            print(f"Warning: cpu.max not found for {name}, skipping.")
                            continue
                        self.files[name, "cpu.max"] = max_path_obj.open()
                    self.files[name, "cpu.max"].seek(0)
                    cpu_max = self.files[name, "cpu.max"].read().strip().split()
                    if cpu_max[0] == "max":
                        stats[name]["cpu_cfs_quota_us"] = -1
                    else:
                        stats[name]["cpu_cfs_quota_us"] = int(cpu_max[0])
                    stats[name]["cpu_cfs_period_us"] = int(cpu_max[1])
                except (OSError, IOError) as e:
                    print(f"Error reading cgroup files for {name}: {e}. Skipping this container.")
                    continue
                except Exception as e:
                    print(f"Unexpected error for {name}: {e}. Skipping this container.")
                    continue
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
                    stats[name]["cpu_usage"] = int(stats[name]["cpu_usage"])
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
                
                cpu_util = self.stats_history[name][-1][1]["dt_cpu_usage"]/self.dt_wall*100
                numcores = os.cpu_count()
                cores_used = cpu_util / 100 / numcores

                # Make sure the system samples at 0 < N < 1s granularity, and stores the values for up to 200 last readings. 
                self.raw_samples.append(cores_used)
                # On every T interval, aggregate latest agg_len samples into a single vector
                if len(self.raw_samples) >= self.agg_len:
                    self.hist_agg_samples.append((self.last_t, self.raw_samples))
                    self.raw_samples = []
                    self.last_agg_t = self.last_t
                else:
                    print(f't={self.last_t}, samples={len(self.raw_samples)}/{self.agg_len}, skipping ...')
                    continue 
                
                # Calculate recommendation 
                samples = np.array([e[1] for e in self.hist_agg_samples[-self.rec_last_n_samples:]])
                if len(samples) >= self.rec_last_n_samples:
                    hists = [np.histogram(s) for s in samples]
                    s_avg_t = self.hists_average(hists)

                    # limit -> quota_us conversion requires quota >= 1000
                    target_core = max(s_avg_t * numcores, 0.01)
                    set_cpu_limit(self.ctr_map, name, target_core)
                    self.last_scale_t = self.last_t

                # Check and evict agg_sample older than lifetime
                if self.hist_agg_samples:
                    oldest_agg_t, _ = self.hist_agg_samples[0]
                    if (self.last_t - oldest_agg_t) >= self.agg_lifetime_sec:
                        self.hist_agg_samples.pop(0)  


    def get_cpu_usages(self, name: str) -> List[float]:
        hist = self.stats_history[name]
        cpu_usages = []
        for ts, stat in hist:
            cpu_usages.append(stat["dt_cpu_usage"])

        return cpu_usages

def main():
    ap = AutoPilot()
    ap.run()


if __name__ == "__main__":
    main()



