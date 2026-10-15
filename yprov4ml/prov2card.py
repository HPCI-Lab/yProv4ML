import os, json, csv, sys
from datetime import datetime
import argparse
import psutil
from pathlib import Path

NF = "~"

def unwrap(val):
    if isinstance(val, str): 
        try: 
            val = eval(val)
        except: pass
    if isinstance(val, dict) and "$" in val:
        return val["$"]
    return val

def g(d, *keys, default=NF):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, None)
        if d is None:
            return default
    return unwrap(d) if d is not None else default

def parse_ts(ts_str):
    if not ts_str or ts_str == NF:
        return NF
    ts_str = ts_str.split("^^")[0]
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.isoformat() + " (ISO 8601)"
    except Exception:
        return ts_str

def parse_numeric_ts(ts_val):
    if ts_val is None or ts_val == "" or ts_val == NF:
        return NF
    ts_float = float(ts_val)
    dt=datetime.fromtimestamp(ts_float/1000)
    return dt.isoformat() + " (ISO 8601)"

def calculate_duration(start_str, end_str):
    if NF in [start_str, end_str] or not (start_str and end_str):
        return NF
    
    raw_start = start_str.split(" (")[0]
    raw_end = end_str.split(" (")[0]
    
    start_dt = datetime.fromisoformat(raw_start)
    end_dt = datetime.fromisoformat(raw_end)
    
    duration = end_dt - start_dt
    return duration
        
def calculate_duration(start_str, end_str):
    # Handle "Not Found" or empty cases
    if NF in [start_str, end_str] or not (start_str and end_str):
        return NF
    
    try:
        # 1. Strip the " (ISO 8601)" suffix to get the raw ISO string
        raw_start = start_str.split(" (")[0]
        raw_end = end_str.split(" (")[0]
        
        # 2. Parse back into datetime objects
        start_dt = datetime.fromisoformat(raw_start)
        end_dt = datetime.fromisoformat(raw_end)
        
        # 3. Calculate difference (returns a timedelta object)
        duration = end_dt - start_dt
        
        return duration
        
    except Exception as e:
        return f"Error calculating duration: {e}"

def duration_str(start, end):
    try:
        s = datetime.fromisoformat(start.split("^^")[0])
        e = datetime.fromisoformat(end.split("^^")[0])
        secs = (e - s).total_seconds()
        return f"{secs:.2f}s"
    except Exception:
        return NF

def fmt_bytes(b):
    try:
        b = float(b)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:.2f} {unit}"
            b /= 1024
        return f"{b:.2f} PB"
    except Exception:
        return NF

def read_csv_column(path, col="value"):
    with open(path) as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader: 
            if col in row:
                try: 
                    vals.append(float(row[col]))
                except ValueError: 
                    vals.append(0.0)
    if vals:
        return {"min": min(vals), "max": max(vals), "avg": sum(vals) / len(vals), "count": len(vals)}

def get_main_activity(yprov):
    for k, v in yprov.get("activity", {}).items():
        if "/" not in k and v and "yprov:experiment_name" in v:
            return k, v
    return NF, {}

def normalize_time(t):
    if isinstance(t, dict): 
        t = t["$"]
    if isinstance(t, str): 
        return datetime.fromisoformat(t)
    else: 
        return datetime.fromtimestamp(t / 1000.0)

def get_sub_activities(yprov, metrics_dir=None):
    acts = []
    start_times = []
    for k, v in yprov.get("activity", {}).items():
        if "/" not in k and "yprov:experiment_name" not in (v or {}):
            acts.append(k)
            if "prov:startedAtTime" in v: 
                start_times.append(v["prov:startedAtTime"])
            else: 
                file = Path(metrics_dir) / [f for f in os.listdir(metrics_dir) if k in f][0]
                time = read_csv_column(file, col="timestep")["min"]
                start_times.append(time)

    zipped = list(zip(start_times, acts))
    zipped.sort(key=lambda pair: normalize_time(pair[0]))
    acts = [val for time, val in zipped]
    return acts

def get_metadata(act): 
    used = ['dcterms:identifier', 'prov:generatedAtTime', 'prov:type', 'prov:role']
    metadata = ""
    for k in act.keys(): 
        if k not in used: 
            metadata += f"{k}={str(unwrap(act[k]))};"
    return metadata


def get_entities_by_role(yprov, role):
    seen = set()
    results = []
    for k, v in yprov.get("entity", {}).items():
        if isinstance(v, list) and len(v) == 1: 
            v = v[0]
        if v.get("prov:role") == role and "Metric" not in str(v.get("prov:type", "")):
            if v.get("prov:label", None): 
                p = v.get("prov:label").split("//")
                ident = p[1 if len(p) == 3 else 0].split(".")[0]
            else: 
                ident = v.get("dcterms:identifier", k)
            if ident not in seen:
                seen.add(ident)
                results.append((ident, v))
    return results


def get_metrics_summary(metrics_dir, activity_filter=None):
    summary = {}
    if not os.path.isdir(metrics_dir):
        return summary
    for fname in os.listdir(metrics_dir):
        if not fname.endswith(".csv"):
            continue
        if activity_filter and activity_filter.lower() not in fname.lower():
            continue
        metric_name = fname.split("_apple_gpu_")[0] if "_apple_gpu_" in fname else fname.replace(".csv", "")
        stats = read_csv_column(os.path.join(metrics_dir, fname))
        if stats:
            summary[metric_name] = stats
    return summary


def aggregate_metrics(metrics_dir):
    agg = {"cpu": {}, "memory": {}, "gpu": {}, "disk": {}}
    if not os.path.isdir(metrics_dir):
        return agg
    for fname in os.listdir(metrics_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(metrics_dir, fname)
        stats = read_csv_column(fpath)
        if not stats:
            continue
        if "cpu_usage" in fname:
            agg["cpu"].setdefault("avg_utilization_pct", []).append(stats["avg"])
        elif "memory_usage" in fname:
            agg["memory"].setdefault("avg_utilization_pct", []).append(stats["avg"])
            agg["memory"].setdefault("tot_utilization", []).append(stats["max"] * psutil.virtual_memory().total)
        elif "disk_usage" in fname:
            agg["disk"].setdefault("avg_utilization_pct", []).append(stats["avg"])
            agg["disk"].setdefault("tot_utilization", []).append(stats["max"] * psutil.disk_usage('/').total)
        elif "gpu_memory_usage" in fname:
            agg["gpu"].setdefault("avg_memory_mb", []).append(stats["avg"])
        elif "gpu_temperature" in fname:
            agg["gpu"].setdefault("avg_temperature_c", []).append(stats["avg"])

    for section, vals in agg.items():
        for metric, lst in vals.items():
            agg[section][metric] = round(sum(lst) / len(lst), 2)
    return agg


def build_card(wf_name, fc_lines, metrics_dir, artifacts_dir):

    _type = "yprov"
    if isinstance(fc_lines, list): 
        _type = "flowcept"

    if _type == "yprov": 
        main_act_key, main_act = get_main_activity(fc_lines)
        sub_acts = get_sub_activities(fc_lines, metrics_dir)

        started_raw = main_act.get("prov:startedAtTime", NF)
        ended_raw = main_act.get("prov:endedAtTime", NF)
        started = parse_ts(unwrap(started_raw))
        ended = parse_ts(unwrap(ended_raw))
        dur = duration_str(started_raw, ended_raw)
        status = "Completed"# if ended != NF else "Unknown"
        platform_info = {}
        platform_info["node"] = main_act.get("yprov:global_rank", NF)
        platform_info["system"] = main_act.get("yprov:operating_system", NF)
        platform_info["release"] = main_act.get("yprov:release", NF)
        platform_info["machine"] = main_act.get("yprov:machine", NF)


        python_ver = main_act.get("yprov:python_version", NF)

        input_entities = get_entities_by_role(fc_lines, "input")
        output_entities = get_entities_by_role(fc_lines, "output")

        repo = {}
        for i_name, i_val in input_entities: 
            if "source_code" in i_name: 
                url_part, full_sha = i_val["prov:value"].rsplit('/', 1)
                repo = {
                    "remote": url_part,
                    "branch": "NF",   
                    "short_sha": full_sha
                }
                break

        agent = list(fc_lines.get("agent", {}).keys())
        agent = agent[0] if len(agent) > 0 else NF

        agg = aggregate_metrics(metrics_dir)

        req_path = os.path.join(artifacts_dir, "requirements.txt")
        env_snapshot = req_path if os.path.isfile(req_path) else NF

    lines = []
    def h(text):
        lines.append(f"# {text}\n")
    def h2(text):
        lines.append(f"\n## {text}\n")
    def h3(text):
        lines.append(f"\n### {text}\n")
    def field(label, value, indent=0):
        prefix = "  " * indent
        lines.append(f"{prefix}- **{label}**: {value}")
    def section_break():
        lines.append("\n---\n")

    h(f"Workflow Card: {wf_name}")
    section_break()

    h2("1. Workflow")
    field("name", main_act.get("yprov:experiment_name", wf_name))
    field("description", f"ML workflow run identified as '{wf_name}', consisting of {len(sub_acts)} sub-activit{'ies' if len(sub_acts) != 1 else 'y'}.")

    h2("2. Summary")
    field("execution_id", main_act.get("yprov:experiment_name", NF))
    v = main_act.get("yprov:run_id", NF)
    if isinstance(v, dict): 
        v = v["$"]
    field("version", unwrap(v))
    field("started_at", started)
    field("ended_at", ended)
    time_zone = main_act.get("yprov:time_zone", NF)
    field("timezone", time_zone)
    field("duration", dur)
    field("status", status)
    loc = main_act.get("yprov:host_name", NF)
    field("location", loc)
    field("user", agent)

    h3("2.1 Entrypoint")

    field("repository", repo.get("remote", NF))
    field("branch", repo.get("branch", NF))
    field("short_sha", repo.get("short_sha", NF))

    h2("3. Infrastructure")
    cpu_info = {}
    cpu_info["count"] = int(metrics_dir.split("_")[-1].replace("GR", "")) + 1

    os_str = f"{platform_info.get('system', NF)} {platform_info.get('processor', '')} {platform_info.get('release', '')}".strip()
    field("host_os", os_str if os_str != NF else NF)
    cpu_brand = platform_info.get('machine', NF)
    cpu_count = cpu_info.get("count", NF)
    field("compute_hardware", f"{cpu_brand}, {cpu_count} cores" if cpu_brand != NF else NF)
    # proc_exec = GR
    field("runtime_environment", NF)
    field("resource_manager", NF)
    primary_sw = []
    if python_ver != NF:
        primary_sw.append(f"Python {python_ver.split('|')[0].strip()}")
    field("primary_software", "; ".join(primary_sw) if primary_sw else NF)
    field("environment_snapshot", open(env_snapshot).read().replace("\n", "; ")[:200] + "..." if env_snapshot != NF else env_snapshot)

    h2("4. Workflow Overview")

    h3("4.1 Run Summary")
    all_acts_in_wf = [k for k in fc_lines.get("activity", {}).keys() if "/" not in k]
    field("total_activities", len(sub_acts))

    finished_count = len(all_acts_in_wf)-1 #sum(1 for k, v in fc_lines.get("activity", {}).items() if "/" not in k and v and "prov:endedAtTime" in v)
    field("status_counts", f"finished: {finished_count}, unknown: {0}")#len(all_acts_in_wf)-1 - finished_count}")

    cmd = fc_lines.get("entity", NF)
    if cmd != NF: 
        if "execution_command" in cmd.keys(): 
            cmd = cmd["execution_command"]["prov:value"]
        else: 
            cmd = NF
    field("exec command", cmd)
    if cmd != NF: 
        cmd = " ".join(cmd.split(" ")[2:]).strip()
        cmd = cmd if cmd != "" else NF
    field("arguments", cmd)

    lines.append("\n**Significant Inputs:**")
    if input_entities:
        for ident, ent in input_entities:
            if ident == "": continue
            size = unwrap(ent.get("yprov:file_size", ""))
            typ = ent.get("prov:type", "file").replace("provml:", "")
            source = ent.get('dcterms:identifier', NF)
            lines.append(f"  - `{ident}` — format: {typ}, size: {size} bytes, source: {source if source != "" else NF}")
    else:
        lines.append(f"  - {NF}")

    lines.append("\n**Significant Outputs:**")
    if output_entities:
        for ident, ent in output_entities:
            if ident == "": continue
            typ = ent.get("prov:type", "file").replace("provml:", "")
            size = unwrap(ent.get("yprov:file_size", ""))
            source = ent.get('dcterms:identifier', NF)
            lines.append(f"  - `{ident}` — type: {typ}, size: {size} bytes, location: {source if source != "" else NF}")
    else:
        lines.append(f"  - {NF}")

    h3("4.2 Workflow Structure")

    for i, act in enumerate(sub_acts, 1):
        lines.append(f"  {i}. {act}")
    if not sub_acts:
        lines.append(f"  - {NF}")


    h3("4.3 Resource Usage")
    import pandas as pd
    cpu_agg = agg.get("cpu", {})
    field("cpu", f"avg utilization: {cpu_agg.get('avg_utilization_pct', NF)}%" if cpu_agg else NF)
    mem_agg = agg.get("memory", {})
    mem_total = fmt_bytes(mem_agg.get("tot_utilization")) if mem_agg else NF
    field("memory", f"total: {mem_total}, avg utilization: {mem_agg.get('avg_utilization_pct', NF)}%" if mem_agg else f"total: {mem_total}")
    gpu_agg = agg.get("gpu", {})
    gpu_agg["avg_temperature_c"] = gpu_agg["avg_temperature_c"] if gpu_agg["avg_temperature_c"] != 0.0 else NF
    field("gpu", f"avg memory: {gpu_agg.get('avg_memory_mb', NF)} MB, avg temp: {gpu_agg.get('avg_temperature_c', NF)} °C" if gpu_agg else NF)
    disk_agg = agg.get("disk", {})
    disk_total = fmt_bytes(disk_agg.get("tot_utilization")) if disk_agg else NF
    field("disk", f"total: {disk_total}, avg utilization: {disk_agg.get('avg_utilization_pct', NF)}%" if disk_agg else f"total: {disk_total}")
    field("network", NF)

    h3("4.4 Observations")
    observations = []
    for observation in observations: 
        lines.append(f"  - {observation}")

    h2("5. Activities")
    for act_name in sub_acts:
        start_times = [pd.read_csv(os.path.join(metrics_dir, f))["timestep"].min() for f in os.listdir(metrics_dir) if act_name in f]
        act_started_raw = min(start_times) if len(start_times) > 0 else None
        act_started = parse_numeric_ts(act_started_raw)

        if any([len(pd.read_csv(os.path.join(metrics_dir, f))["timestep"]) > 1 for f in os.listdir(metrics_dir) if act_name in f]): 
            act_ended_raw = max([pd.read_csv(os.path.join(metrics_dir, f))["timestep"].max() for f in os.listdir(metrics_dir) if act_name in f])
            act_ended = parse_numeric_ts(act_ended_raw)
        else: 
            act_ended = parse_ts(unwrap(ended_raw))

        act_metrics = get_metrics_summary(metrics_dir, activity_filter=act_name)
        act_outputs = [
            (k, v) for k, v in fc_lines.get("entity", {}).items()
            if v.get("yprov:context") == act_name and v.get("prov:role") == "output"
        ]
        act_inputs_via_used = [
            v.get("prov:entity") for v in fc_lines.get("used", {}).values()
        ]


        lines.append(f"\n#### Activity: `{act_name}`\n")
        field("name", act_name)
        field("task_count", 1)
        field("started_at", act_started)
        field("ended_at", act_ended)
        field("duration", calculate_duration(act_started, act_ended))
        act_succ = "Success" #if "prov:endedAtTime" in fc_lines.get("activity", {})[act_name].keys() or "prov:startedAtTime" in fc_lines.get("activity", {})[act_name].keys() else "unknown"
        field("status", act_succ)

        field("hosts", "")
        # lines.append("  - **hosts**:")
        host_id = unwrap(main_act.get("yprov:global_rank", NF))
        lines.append(f"    - host: `{host_id}`, tasks: 1")
        if act_metrics:
            for metric_name, stats in list(act_metrics.items()): # [:3]
                lines.append(f"      - {metric_name}: avg={stats['avg']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

        field("inputs", "")
        relevant_inputs = [e for e in act_inputs_via_used if act_name in str(e) or main_act_key in str(e)]
        if relevant_inputs:
            for inp in relevant_inputs: #[:3]
                lines.append(f"    - `{inp}`")
        else:
            lines.append(f"    - {NF}")

        field("outputs", "")
        if act_outputs:
            for k, v in act_outputs: #[:5]
                lines.append(f"    - `{v.get('prov:label', k)}` — {v.get('prov:type', 'file')}, path: {v.get('dcterms:identifier', NF)}")
        else:
            lines.append(f"    - {NF}")

    h2("6. Significant Workflow Artifacts")

    h3("Input Artifacts")
    seen_inputs = set()
    any_input = False
    for ident, ent in input_entities:
        if ident == "": continue
        
        if ident in seen_inputs:
            continue
        seen_inputs.add(ident)
        any_input = True
        lines.append(f"\n**Artifact: `{ident}`**")
        field("identifier", ident)
        typ = ent.get("prov:type", "file").replace("provml:", "")
        size_raw = ent.get("yprov:file_size", NF)
        size = size_raw.split("^^")[0] if "^^" in str(size_raw) else str(size_raw)
        tm = unwrap(ent.get("prov:generatedAtTime", NF).replace("prov:", ""))
        metadata = get_metadata(ent)
        field("description", f"Input file of type {typ}, used by the workflow at time {tm}.")
        field("reference", ent.get("dcterms:identifier", NF))
        field("size", fmt_bytes(size))
        field("additional metadata", metadata)
    if not any_input:
        field("name", NF)
        field("description", NF)
        field("reference", NF)
        field("size", NF)
        field("additional metadata", NF)

    h3("Output Artifacts")
    seen_outputs = set()
    any_output = False
    for ident, ent in output_entities:
        if ident == "": continue

        if ident in seen_outputs:
            continue
        seen_outputs.add(ident)
        any_output = True
        lines.append(f"\n**Artifact: `{ident}`**")
        field("identifier", ident)
        typ = ent.get("prov:type", "file").replace("provml:", "")
        size_raw = ent.get("yprov:file_size", NF)
        size = size_raw.split("^^")[0] if "^^" in str(size_raw) else str(size_raw)
        tm = ent.get("prov:generatedAtTime", NF).replace("prov:", "")
        metadata = get_metadata(ent)
        field("description", f"Output artifact of type '{typ}', generated at time {tm}.")
        field("reference", ent.get("dcterms:identifier", NF))
        field("size", fmt_bytes(size))
        field("additional metadata", metadata)
    if not any_output:
        field("name", NF)
        field("description", NF)
        field("reference", NF)
        field("size", NF)
        field("additional metadata", NF)

    lines.append("")
    return "\n".join(lines)



def convert(json_file : str, output_dir : str):
    json_file = Path(json_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if json_file.is_file():
        json_file = str(json_file)
        if not json_file.endswith(".jsonl"): 
            exit("Not supported")
        wf_name = json_file.split("/")[-1].replace("_flowcept.jsonl", "")
        with open(json_file) as f:
            fc_lines = [json.loads(l) for l in f if l.strip()]
        metrics_dir = None
        artifacts_dir = None
    elif json_file.is_dir(): 
        json_ = [f for f in json_file.iterdir() if str(f).endswith(".json")][0]
        wf_name = json_file.name.replace("_yprov", "")
        with open(json_) as f:
            fc_lines = json.load(f)
        metrics_dir = os.path.join(json_file, "metrics_GR0")
        artifacts_dir = os.path.join(json_file, "artifacts_GR0")
    else: 
        exit("Not supported")

    card = build_card(
        wf_name=wf_name, 
        fc_lines=fc_lines,
        metrics_dir=metrics_dir,
        artifacts_dir=artifacts_dir,
    )

    out_path = os.path.join(output_dir, f"card_{wf_name}.md")
    with open(out_path, "w") as f:
        f.write(card)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a PROV-JSON file to a DOT file')
    parser.add_argument('--path', type=str, help='The PROV directory or JSONL to convert')
    parser.add_argument('--output_dir', type=str, help='The output directory for the card file', default=None)
    args = parser.parse_args()

    convert(args.path, args.output_dir or "./")
